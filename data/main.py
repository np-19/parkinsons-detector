import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import synapseclient
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

# Put these two CSV files in the same directory as this script.
VOICE_CSV = Path("Job-10649242714125674294653066972.csv")
DEMOGRAPHICS_CSV = Path("Job-1124303654762487256360746899.csv")

SYNAPSE_TABLE_ID = "syn5511444"
AUDIO_COLUMN = "audio_audio.m4a"

OUTPUT_DIR = Path(__file__).resolve().parent / "mPower_Audio"
TARGET_PD = 5_000
TARGET_CONTROL = 5_000
CHUNK_SIZE = 500
RANDOM_SEED = 42

# Conservative control definition:
# professional-diagnosis == False AND no PD-related year is reported.
#
# Set this to False if you want to use professional-diagnosis == False
# literally, without the additional consistency check.
EXCLUDE_CONTRADICTORY_CONTROLS = True


# ============================================================
# Load metadata
# ============================================================

def load_metadata():
    if not VOICE_CSV.exists():
        raise FileNotFoundError(f"Voice CSV not found: {VOICE_CSV}")

    if not DEMOGRAPHICS_CSV.exists():
        raise FileNotFoundError(
            f"Demographics CSV not found: {DEMOGRAPHICS_CSV}"
        )

    voice = pd.read_csv(VOICE_CSV)
    demo = pd.read_csv(DEMOGRAPHICS_CSV)

    required_voice = {
        "ROW_ID",
        "healthCode",
        AUDIO_COLUMN,
    }

    required_demo = {
        "healthCode",
        "professional-diagnosis",
        "diagnosis-year",
        "onset-year",
        "medication-start-year",
    }

    missing_voice = required_voice - set(voice.columns)
    missing_demo = required_demo - set(demo.columns)

    if missing_voice:
        raise ValueError(
            f"Voice CSV is missing columns: {sorted(missing_voice)}"
        )

    if missing_demo:
        raise ValueError(
            f"Demographics CSV is missing columns: {sorted(missing_demo)}"
        )

    return voice, demo


# ============================================================
# Build participant-level labels
# ============================================================

def build_labels(demo):
    # The demographics file currently has one row per healthCode.
    # Still check this so a future CSV cannot silently duplicate recordings.
    duplicate_health_codes = demo["healthCode"].duplicated().sum()

    if duplicate_health_codes:
        raise ValueError(
            f"Demographics CSV contains {duplicate_health_codes} duplicate "
            "healthCode rows. Resolve these before selecting recordings."
        )

    demo = demo.copy()

    # Normalize the diagnosis column to real booleans.
    demo["professional-diagnosis"] = demo["professional-diagnosis"].map(
        lambda x: (
            True if str(x).strip().lower() == "true"
            else False if str(x).strip().lower() == "false"
            else np.nan
        )
    )

    # We cannot responsibly label missing diagnosis as either class.
    labeled = demo.dropna(
        subset=["healthCode", "professional-diagnosis"]
    ).copy()

    if EXCLUDE_CONTRADICTORY_CONTROLS:
        # A participant saying "no professional PD diagnosis" but also
        # supplying a PD diagnosis/onset/medication-start year is treated
        # as ambiguous rather than automatically healthy.
        year_columns = [
            "diagnosis-year",
            "onset-year",
            "medication-start-year",
        ]

        for col in year_columns:
            labeled[col] = pd.to_numeric(
                labeled[col], errors="coerce"
            )

        contradictory_control = (
            (labeled["professional-diagnosis"] == False)
            & labeled[year_columns].notna().any(axis=1)
        )

        labeled = labeled[~contradictory_control].copy()

    labeled["label"] = np.where(
        labeled["professional-diagnosis"],
        "PD",
        "CONTROL",
    )

    return labeled[["healthCode", "label"]]


# ============================================================
# Select 10,000 recordings
# ============================================================

def select_recordings(voice, labels):
    rng = np.random.default_rng(RANDOM_SEED)

    # Only recordings with a known, usable diagnosis are eligible.
    data = voice[
        ["ROW_ID", "healthCode", AUDIO_COLUMN]
    ].dropna().copy()

    data["ROW_ID"] = data["ROW_ID"].astype(int)

    # Join by participant, not by recording.
    data = data.merge(
        labels,
        on="healthCode",
        how="inner",
        validate="many_to_one",
    )

    pd_data = data[data["label"] == "PD"].copy()
    control_data = data[data["label"] == "CONTROL"].copy()

    pd_participants = pd_data["healthCode"].nunique()
    control_participants = control_data["healthCode"].nunique()

    print(f"Eligible PD participants:      {pd_participants:,}")
    print(f"Eligible control participants: {control_participants:,}")
    print(f"Eligible PD recordings:        {len(pd_data):,}")
    print(f"Eligible control recordings:   {len(control_data):,}")

    if pd_participants == 0 or control_participants == 0:
        raise RuntimeError("Both PD and control participants are required.")

    if len(pd_data) < TARGET_PD:
        raise RuntimeError(
            f"Only {len(pd_data):,} eligible PD recordings exist; "
            f"{TARGET_PD:,} are required."
        )

    if len(control_data) < TARGET_CONTROL:
        raise RuntimeError(
            f"Only {len(control_data):,} eligible control recordings exist; "
            f"{TARGET_CONTROL:,} are required."
        )

    # We deliberately maximize participant diversity.
    #
    # PD: 5,000 recordings / 970 participants means:
    #      820 participants get 5 recordings
    #      150 participants get 6 recordings
    #
    # Controls: 5,000 / 4,107 participants means:
    #      3,214 participants get 1 recording
    #        893 participants get 2 recordings
    #
    # The exact participant counts depend on the input metadata.

    def balanced_round_robin(group, target):
        participants = group["healthCode"].drop_duplicates().tolist()
        rng.shuffle(participants)

        by_participant = {
            p: g.sample(frac=1, random_state=RANDOM_SEED + i)
            for i, (p, g) in enumerate(
                group.groupby("healthCode", sort=False)
            )
        }

        selected_rows = []

        # Round-robin guarantees that we use every participant once
        # before giving anyone a second recording, then every participant
        # again before a third, etc.
        while len(selected_rows) < target:
            made_progress = False

            for participant in participants:
                candidate = by_participant[participant]

                if candidate.empty:
                    continue

                selected_rows.append(candidate.iloc[0])
                by_participant[participant] = candidate.iloc[1:]
                made_progress = True

                if len(selected_rows) == target:
                    break

            if not made_progress:
                raise RuntimeError(
                    f"Could not select {target:,} recordings."
                )

        return pd.DataFrame(selected_rows)

    selected_pd = balanced_round_robin(pd_data, TARGET_PD)
    selected_control = balanced_round_robin(
        control_data, TARGET_CONTROL
    )

    selected = pd.concat(
        [selected_pd, selected_control],
        ignore_index=True,
    )

    # Shuffle the final query order.
    selected = selected.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    # Hard validation.
    assert len(selected) == 10_000
    assert selected["ROW_ID"].nunique() == 10_000
    assert selected["healthCode"].isna().sum() == 0
    assert selected["label"].value_counts()["PD"] == TARGET_PD
    assert selected["label"].value_counts()["CONTROL"] == TARGET_CONTROL

    print("\nFinal selection:")
    print(selected["label"].value_counts())
    print(
        f"Unique participants: "
        f"{selected['healthCode'].nunique():,}"
    )
    print(
        f"Maximum recordings from one participant: "
        f"{selected['healthCode'].value_counts().max()}"
    )

    return selected


# ============================================================
# Download from Synapse
# ============================================================

def download_audio(selected):
    load_dotenv()

    access_token = os.getenv("ACCESS_TOKEN")

    if not access_token:
        raise RuntimeError(
            "ACCESS_TOKEN is not set. Put it in your .env file."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.login(authToken=access_token)

    row_ids = selected["ROW_ID"].astype(int).tolist()

    print("\nStarting Synapse download...")
    print(f"Recordings to download: {len(row_ids):,}")
    print(f"Output directory: {OUTPUT_DIR}")

    downloaded = 0
    failed = []

    for start in range(0, len(row_ids), CHUNK_SIZE):
        chunk = row_ids[start:start + CHUNK_SIZE]
        id_list = ",".join(str(row_id) for row_id in chunk)

        query = (
            f"SELECT * FROM {SYNAPSE_TABLE_ID} "
            f"WHERE ROW_ID IN ({id_list})"
        )

        print(
            f"\nChunk {start // CHUNK_SIZE + 1}: "
            f"{len(chunk)} recordings"
        )

        try:
            results = syn.tableQuery(query)

            downloaded_files = syn.downloadTableColumns(
                results,
                [AUDIO_COLUMN],
            )

            for file_path in downloaded_files.values():
                if file_path and os.path.exists(file_path):
                    shutil.copy2(file_path, OUTPUT_DIR)
                    downloaded += 1

        except Exception as exc:
            failed.extend(chunk)
            print(f"Chunk failed: {exc}")

        print(f"Downloaded so far: {downloaded:,}")

    print("\n==============================")
    print("Download complete")
    print("==============================")
    print(f"Requested: 10,000")
    print(f"Downloaded: {downloaded:,}")
    print(f"Failed:     {len(failed):,}")

    if failed:
        failed_file = OUTPUT_DIR / "failed_row_ids.txt"
        failed_file.write_text(
            "\n".join(map(str, failed)),
            encoding="utf-8",
        )
        print(f"Failed ROW_IDs saved to: {failed_file}")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading mPower metadata...")
    voice, demo = load_metadata()

    print(f"Voice recordings in CSV: {len(voice):,}")
    print(f"Voice participants:       {voice['healthCode'].nunique():,}")
    print(f"Demographic participants: {demo['healthCode'].nunique():,}")

    labels = build_labels(demo)

    selected = select_recordings(voice, labels)

    download_audio(selected)


if __name__ == "__main__":
    main()
