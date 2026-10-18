"""Test participant-level isolation and absence of data leakage across dataset splits."""

from pathlib import Path
import pandas as pd
import pytest

METADATA_DIR = Path(__file__).resolve().parent.parent / "data" / "metadata"


def test_participant_split_disjointness():
    """Verify 100% disjoint participant sets (zero leakage) between train, validation, and test splits."""
    train_path = METADATA_DIR / "train.csv"
    val_path = METADATA_DIR / "validation.csv"
    test_path = METADATA_DIR / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Metadata CSVs not found in data/metadata/")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_participants = set(df_train["healthCode"].dropna().unique())
    val_participants = set(df_val["healthCode"].dropna().unique())
    test_participants = set(df_test["healthCode"].dropna().unique())

    # Assert completely disjoint sets
    train_val_overlap = train_participants.intersection(val_participants)
    train_test_overlap = train_participants.intersection(test_participants)
    val_test_overlap = val_participants.intersection(test_participants)

    assert len(train_val_overlap) == 0, f"Leakage found between train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Leakage found between train and test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage found between val and test: {val_test_overlap}"


def test_participant_label_consistency():
    """Verify that no participant has conflicting labels within any split."""
    for split_name in ["train.csv", "validation.csv", "test.csv"]:
        path = METADATA_DIR / split_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "label" in df.columns and "healthCode" in df.columns:
            grouped = df.groupby("healthCode")["label"].nunique()
            inconsistent = grouped[grouped > 1]
            assert len(inconsistent) == 0, f"Inconsistent labels for participants in {split_name}: {inconsistent.index.tolist()}"
