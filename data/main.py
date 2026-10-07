import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import synapseclient
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
VOICE_CSV = BASE_DIR / "Job-10649242714125674294653066972.csv"
DEMOGRAPHICS_CSV = BASE_DIR / "Job-1124303654762487256360746899.csv"

SYNAPSE_TABLE_ID = "syn5511444"
AUDIO_COLUMN = "audio_audio.m4a"
OUTPUT_DIR = BASE_DIR / "mPower_Audio"
METADATA_DIR = BASE_DIR / "metadata"

TARGET_PD = 5_000
TARGET_CONTROL = 5_000
CHUNK_SIZE = 100
RANDOM_SEED = 42
EXCLUDE_CONTRADICTORY_CONTROLS = True


def load_metadata():
    if not VOICE_CSV.exists():
        raise FileNotFoundError(f"Voice CSV not found: {VOICE_CSV}")
    if not DEMOGRAPHICS_CSV.exists():
        raise FileNotFoundError(f"Demographics CSV not found: {DEMOGRAPHICS_CSV}")

    voice = pd.read_csv(VOICE_CSV)
    demo = pd.read_csv(DEMOGRAPHICS_CSV)

    required_voice = {"ROW_ID", "healthCode", AUDIO_COLUMN}
    required_demo = {
        "healthCode", "professional-diagnosis", "diagnosis-year",
        "onset-year", "medication-start-year"
    }

    missing = required_voice - set(voice.columns)
    if missing:
        raise ValueError(f"Voice CSV is missing columns: {sorted(missing)}")

    missing = required_demo - set(demo.columns)
    if missing:
        raise ValueError(f"Demographics CSV is missing columns: {sorted(missing)}")

    return voice, demo


def build_labels(demo):
    demo = demo.copy()

    if demo["healthCode"].duplicated().any():
        raise ValueError("Demographics CSV contains duplicate healthCode rows.")

    demo["professional-diagnosis"] = (
        demo["professional-diagnosis"]
        .astype("string").str.strip().str.lower()
        .map({"true": True, "false": False})
    )

    labeled = demo.dropna(
        subset=["healthCode", "professional-diagnosis"]
    ).copy()

    if EXCLUDE_CONTRADICTORY_CONTROLS:
        year_columns = [
            "diagnosis-year", "onset-year", "medication-start-year"
        ]
        for col in year_columns:
            labeled[col] = pd.to_numeric(labeled[col], errors="coerce")

        contradictory = (
            (labeled["professional-diagnosis"] == False)
            & labeled[year_columns].notna().any(axis=1)
        )
        removed = int(contradictory.sum())
        if removed:
            print(f"Excluding {removed:,} contradictory controls.")
        labeled = labeled.loc[~contradictory].copy()

    labeled["label"] = np.where(
        labeled["professional-diagnosis"], "PD", "CONTROL"
    )
    return labeled[["healthCode", "label"]]


def balanced_round_robin(group, target, rng):
    participants = group["healthCode"].drop_duplicates().tolist()
    rng.shuffle(participants)

    by_participant = {}
    for i, (participant, rows) in enumerate(group.groupby("healthCode", sort=False)):
        by_participant[participant] = rows.sample(
            frac=1, random_state=RANDOM_SEED + i
        ).reset_index(drop=True)

    selected = []
    while len(selected) < target:
        progress = False
        for participant in participants:
            rows = by_participant[participant]
            if rows.empty:
                continue
            selected.append(rows.iloc[0])
            by_participant[participant] = rows.iloc[1:]
            progress = True
            if len(selected) == target:
                break
        if not progress:
            raise RuntimeError(f"Could not select {target:,} recordings.")

    return pd.DataFrame(selected)


def select_recordings(voice, labels):
    rng = np.random.default_rng(RANDOM_SEED)

    data = voice[["ROW_ID", "healthCode", AUDIO_COLUMN]].dropna().copy()
    data["ROW_ID"] = pd.to_numeric(data["ROW_ID"], errors="raise").astype(int)
    data[AUDIO_COLUMN] = data[AUDIO_COLUMN].astype(str).str.strip()
    data = data[
        (data[AUDIO_COLUMN] != "")
        & (data[AUDIO_COLUMN].str.lower() != "nan")
    ].copy()

    if data["ROW_ID"].duplicated().any():
        raise ValueError("Voice CSV contains duplicate ROW_ID values.")

    data = data.merge(labels, on="healthCode", how="inner", validate="many_to_one")

    pd_data = data[data["label"] == "PD"].copy()
    control_data = data[data["label"] == "CONTROL"].copy()

    print(f"Eligible PD participants:      {pd_data['healthCode'].nunique():,}")
    print(f"Eligible control participants: {control_data['healthCode'].nunique():,}")
    print(f"Eligible PD recordings:        {len(pd_data):,}")
    print(f"Eligible control recordings:   {len(control_data):,}")

    if len(pd_data) < TARGET_PD:
        raise RuntimeError("Not enough eligible PD recordings.")
    if len(control_data) < TARGET_CONTROL:
        raise RuntimeError("Not enough eligible control recordings.")

    selected = pd.concat([
        balanced_round_robin(pd_data, TARGET_PD, rng),
        balanced_round_robin(control_data, TARGET_CONTROL, rng),
    ], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    if len(selected) != TARGET_PD + TARGET_CONTROL:
        raise RuntimeError("Final recording count is incorrect.")
    if selected["ROW_ID"].nunique() != len(selected):
        raise RuntimeError("Duplicate ROW_ID in final selection.")

    counts = selected["label"].value_counts()
    if counts.get("PD", 0) != TARGET_PD:
        raise RuntimeError("Final PD count is incorrect.")
    if counts.get("CONTROL", 0) != TARGET_CONTROL:
        raise RuntimeError("Final control count is incorrect.")

    participant_counts = selected["healthCode"].value_counts()
    print("\nFinal selection:")
    print(counts)
    print(f"Unique participants: {selected['healthCode'].nunique():,}")
    print(f"Maximum recordings from one participant: {participant_counts.max()}")
    return selected


def save_manifest(rows, path):
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates("ROW_ID", keep="last")
    new["ROW_ID"] = new["ROW_ID"].astype(str)
    new = new.sort_values("ROW_ID", key=lambda col: col.astype(str))
    new.to_csv(path, index=False)


def download_audio(selected):
    load_dotenv()
    token = os.getenv("ACCESS_TOKEN")
    if not token:
        raise RuntimeError("ACCESS_TOKEN is not set in .env.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.Synapse()
    print("\nLogging into Synapse...")
    syn.login(authToken=token)

    selected.to_csv(METADATA_DIR / "selected_recordings.csv", index=False)
    manifest_path = METADATA_DIR / "manifest.csv"
    failed_path = METADATA_DIR / "failed_row_ids.txt"

    completed = set()
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype={"ROW_ID": str})
        if "ROW_ID" in manifest.columns:
            completed = set(manifest["ROW_ID"].astype(str))
        print(f"Previously completed recordings: {len(completed):,}")

    failed = []
    downloaded = 0

    for start in range(0, len(selected), CHUNK_SIZE):
        chunk = selected.iloc[start:start + CHUNK_SIZE].copy()
        chunk_no = start // CHUNK_SIZE + 1
        pending = chunk[~chunk["ROW_ID"].astype(str).isin(completed)].copy()

        print(f"\nChunk {chunk_no}: {len(chunk)} recordings ({len(pending)} pending)")
        if pending.empty:
            continue

        batch_dir = OUTPUT_DIR / f".batch_{chunk_no:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        row_ids = pending["ROW_ID"].astype(int).tolist()
        row_id_sql = ",".join(map(str, row_ids))

        # Do NOT put audio_audio.m4a in the SELECT list. The dot in this
        # FILEHANDLEID column name is interpreted by Synapse SQL. The
        # documented downloadTableColumns API receives the file column
        # separately from the SELECT * query.
        query = (
            f"SELECT * FROM {SYNAPSE_TABLE_ID} "
            f"WHERE ROW_ID IN ({row_id_sql})"
        )

        try:
            results = syn.tableQuery(query)
            file_map = syn.downloadTableColumns(
                results,
                [AUDIO_COLUMN],
                downloadLocation=str(batch_dir),
            )
        except KeyboardInterrupt:
            shutil.rmtree(batch_dir, ignore_errors=True)
            raise
        except Exception as exc:
            print(f"CHUNK {chunk_no} FAILED: {type(exc).__name__}: {exc}")
            failed.extend(row_ids)
            shutil.rmtree(batch_dir, ignore_errors=True)
            continue

        handle_to_row = {
            str(row[AUDIO_COLUMN]).strip(): int(row["ROW_ID"])
            for _, row in pending.iterrows()
        }

        manifest_rows = []
        for file_handle, downloaded_path in file_map.items():
            file_handle = str(file_handle)
            row_id = handle_to_row.get(file_handle)
            if row_id is None:
                print(f"WARNING: file handle {file_handle} not in selected metadata.")
                continue

            source = Path(downloaded_path)
            destination = OUTPUT_DIR / f"{row_id}.m4a"

            try:
                if not source.exists():
                    raise FileNotFoundError(f"Downloaded file does not exist: {source}")
                if source.stat().st_size == 0:
                    raise RuntimeError("Downloaded file is empty.")

                if destination.exists():
                    destination.unlink()
                shutil.move(str(source), str(destination))

                row = pending[pending["ROW_ID"] == row_id].iloc[0]
                manifest_rows.append({
                    "ROW_ID": row_id,
                    "healthCode": row["healthCode"],
                    "label": row["label"],
                    "file_handle_id": file_handle,
                    "filename": destination.name,
                })
                completed.add(str(row_id))
                downloaded += 1
            except Exception as exc:
                print(f"FAILED ROW_ID {row_id}: {type(exc).__name__}: {exc}")
                failed.append(row_id)

        if manifest_rows:
            save_manifest(manifest_rows, manifest_path)

        shutil.rmtree(batch_dir, ignore_errors=True)
        print(f"Downloaded this run: {downloaded:,}")
        print(f"Failed so far: {len(set(failed)):,}")

    audio_files = list(OUTPUT_DIR.glob("*.m4a"))
    manifest_count = 0
    if manifest_path.exists():
        manifest_count = len(pd.read_csv(manifest_path))

    failed = sorted(set(failed))
    if failed:
        failed_path.write_text("\n".join(map(str, failed)), encoding="utf-8")
    elif failed_path.exists():
        failed_path.unlink()

    print("\n==============================")
    print("DOWNLOAD SUMMARY")
    print("==============================")
    print(f"Requested recordings: {len(selected):,}")
    print(f"Downloaded this run:  {downloaded:,}")
    print(f"Final .m4a files:     {len(audio_files):,}")
    print(f"Manifest entries:     {manifest_count:,}")
    print(f"Failed recordings:    {len(failed):,}")
    print(f"Output directory:     {OUTPUT_DIR}")
    if failed:
        print(f"Failed ROW_IDs:       {failed_path}")
    else:
        print("All requested recordings are present.")


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
