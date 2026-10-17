#!/usr/bin/env python3
"""Local Clinical Evaluation Script for the 'records' Dataset.

Evaluates Parkinson's Disease (PD), Healthy Controls (HC), and Atypical
Parkinsonism (MSA, PSP) without Docker.
Supports CPU mode (default) and NVIDIA GPU mode (with automatic fallback).

Usage:
    python3 scripts/evaluate_records_dataset.py [--device cpu|gpu] [--vowels a|all] [--audio-type clean|raw]
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Ensure repository root and venv site-packages are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
venv_site = REPO_ROOT / "data" / ".venv" / "lib" / "python3.14" / "site-packages"
if venv_site.exists():
    sys.path.insert(0, str(venv_site))



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate records dataset locally without Docker.")
    parser.add_argument(
        "--csv",
        type=str,
        default="test_samples/records/dataset.csv",
        help="Path to dataset.csv",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="test_samples/records",
        help="Directory containing the WAV audio files",
    )
    parser.add_argument(
        "--vowels",
        choices=["a", "all"],
        default="a",
        help="Vowel phonation to evaluate: 'a' (sustained /a/, recommended model protocol) or 'all' (/a/ and /i/)",
    )
    parser.add_argument(
        "--audio-type",
        choices=["clean", "raw"],
        default="clean",
        help="Use cleaned/denoised audio ('clean') or raw recordings ('raw')",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "cuda"],
        default="cpu",
        help="Hardware acceleration: 'cpu' (default) or 'gpu'/'cuda' (with automatic CPU fallback)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/records_dataset_evaluation_results.csv",
        help="Output CSV path for detailed evaluation metrics",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/deep_learning/wav2vec2_v2/best_model.pt",
        help="Path to best_model.pt checkpoint to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of subjects to evaluate (useful for fast smoke tests)",
    )
    return parser.parse_args()


def load_dataset_metadata(csv_path: str, vowels_filter: str) -> Dict[str, Dict]:
    """Parse dataset.csv and group recording names by subject ID."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    subjects = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row["subject"].strip()
            group = row["group"].strip()
            vowel = row["vowel"].strip()
            record_id = row["record"].strip()

            if vowels_filter == "a" and vowel != "a":
                continue

            if subject_id not in subjects:
                subjects[subject_id] = {
                    "subject": subject_id,
                    "group": group,
                    "records": [],
                    "vowels": [],
                }
            subjects[subject_id]["records"].append(record_id)
            subjects[subject_id]["vowels"].append(vowel)

    return subjects


def resolve_audio_paths(records: List[str], audio_dir: str, audio_type: str) -> List[str]:
    """Resolve file paths for either clean or raw audio files."""
    paths = []
    suffix = "_clean.wav" if audio_type == "clean" else ".wav"
    for rec in records:
        candidate = os.path.join(audio_dir, f"{rec}{suffix}")
        if os.path.exists(candidate):
            paths.append(candidate)
        else:
            fallback = os.path.join(audio_dir, f"{rec}.wav")
            if os.path.exists(fallback):
                paths.append(fallback)
    return paths


def main():
    args = parse_args()

    # Lazy load heavy neural network dependencies
    import torch
    from src.deep_learning.predict_patient import PatientPredictor, META_THRESHOLD

    # Hardware detection with graceful fallback
    device = "cpu"
    if args.device in ("gpu", "cuda"):
        if torch.cuda.is_available():
            device = "cuda"
            print(f">>> [HARDWARE] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] GPU requested, but CUDA is not accessible. Falling back to CPU.")
            device = "cpu"
    else:
        print(">>> [HARDWARE] Running in CPU mode (Default)")

    print("================================================================================")
    print(" NEUROVOCAL CLINICAL EVALUATION: 'RECORDS' PHONATION DATASET")
    print("================================================================================")
    print(f" Dataset CSV   : {args.csv}")
    print(f" Audio Dir     : {args.audio_dir}")
    print(f" Target Vowels : {'Sustained /a/ (Model Standard Protocol)' if args.vowels == 'a' else 'All Vowels (/a/ and /i/)'}")
    print(f" Audio Type    : {args.audio_type.upper()} WAVs")
    print(f" Inference Dev : {device.upper()}")
    print("--------------------------------------------------------------------------------")

    # Load subjects metadata
    subjects = load_dataset_metadata(args.csv, args.vowels)
    cohort_counts = Counter(s["group"] for s in subjects.values())
    print(f" Cohorts Loaded ({len(subjects)} total subjects):")
    for group, count in sorted(cohort_counts.items()):
        print(f"   • {group:<6} : {count:>2} subjects")
    print("--------------------------------------------------------------------------------")

    # Initialize PatientPredictor
    print(f" Loading model from: {args.model_path}")
    predictor = PatientPredictor(model_path=args.model_path, device=device)

    results = []
    print(f"\n{'Subject ID':<12} | {'Cohort':<6} | {'Files':<5} | {'Mean Rec P':<10} | {'Meta Score':<10} | {'Triage':<11} | {'Result':<12}")
    print("-" * 80)

    items_to_eval = sorted(subjects.items(), key=lambda x: (x[1]["group"], x[0]))
    if args.limit:
        items_to_eval = items_to_eval[:args.limit]

    for sub_id, data in items_to_eval:
        group = data["group"]
        audio_files = resolve_audio_paths(data["records"], args.audio_dir, args.audio_type)

        if not audio_files:
            print(f"{sub_id:<12} | {group:<6} | {0:<5} | {'N/A':<10} | {'N/A':<10} | {'NO_AUDIO':<11} | {'SKIPPED':<12}")
            continue

        try:
            report = predictor.predict_patient(audio_files)
            meta_score = report.get("meta_classifier_score", 0.0)
            mean_p = report.get("distribution_summary", {}).get("mean_probability", 0.0)
            triage = report.get("triage_level", "UNKNOWN")
            is_pos = meta_score >= META_THRESHOLD

            # Result determination
            if group == "PD":
                result = "PASS" if is_pos else "MISS"
            elif group == "HC":
                result = "PASS" if not is_pos else "FALSE_ALARM"
            else:
                # Atypical (MSA, PSP)
                result = "PD_FLAGGED" if is_pos else "CLEARED"

            row = {
                "subject_id": sub_id,
                "cohort": group,
                "num_files": len(audio_files),
                "mean_recording_p": f"{mean_p:.4f}",
                "meta_score": f"{meta_score:.4f}",
                "triage_tier": triage,
                "diagnosis": report.get("participant_verdict", "N/A"),
                "result": result,
            }
            results.append(row)
            print(f"{sub_id:<12} | {group:<6} | {len(audio_files):<5} | {mean_p:<10.4f} | {meta_score:<10.4f} | {triage:<11} | {result:<12}", flush=True)

        except Exception as e:
            print(f"{sub_id:<12} | {group:<6} | ERROR: {e}", flush=True)

    # Compute Clinical Metrics for PD vs HC
    pd_rows = [r for r in results if r["cohort"] == "PD"]
    hc_rows = [r for r in results if r["cohort"] == "HC"]
    msa_rows = [r for r in results if r["cohort"] == "MSA"]
    psp_rows = [r for r in results if r["cohort"] == "PSP"]

    tp = sum(1 for r in pd_rows if r["result"] == "PASS")
    fn = sum(1 for r in pd_rows if r["result"] == "MISS")
    tn = sum(1 for r in hc_rows if r["result"] == "PASS")
    fp = sum(1 for r in hc_rows if r["result"] == "FALSE_ALARM")

    sens = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    bacc = (sens + spec) / 2.0
    total_eval = tp + fn + tn + fp
    acc = ((tp + tn) / total_eval * 100.0) if total_eval > 0 else 0.0

    print("\n" + "=" * 80)
    print(" CLINICAL PERFORMANCE EVALUATION REPORT (PD vs HEALTHY CONTROLS)")
    print("=" * 80)
    print(f" Parkinson's Subjects (PD)  : {len(pd_rows):>2}")
    print(f" Healthy Controls (HC)      : {len(hc_rows):>2}")
    print(f" Total Evaluated            : {total_eval:>2}")
    print("--------------------------------------------------------------------------------")
    print(f" Sensitivity (PD Recall)    : {sens:>6.2f}% ({tp}/{tp + fn} Parkinson's correctly identified)")
    print(f" Specificity (HC Clearance) : {spec:>6.2f}% ({tn}/{tn + fp} Healthy controls correctly cleared)")
    print(f" Balanced Accuracy          : {bacc:>6.2f}%")
    print(f" Overall Accuracy           : {acc:>6.2f}%")
    print("--------------------------------------------------------------------------------")
    print(" Confusion Matrix:")
    print(f"   • True Positives  (TP) : {tp:<2}  (Correctly flagged PD)")
    print(f"   • False Negatives (FN) : {fn:<2}  (Missed PD)")
    print(f"   • True Negatives  (TN) : {tn:<2}  (Correctly cleared HC)")
    print(f"   • False Positives (FP) : {fp:<2}  (False alarms on HC)")
    print("--------------------------------------------------------------------------------")

    if msa_rows or psp_rows:
        print(" Atypical Parkinsonism Acoustic Manifestation:")
        if msa_rows:
            msa_flagged = sum(1 for r in msa_rows if r["result"] == "PD_FLAGGED")
            print(f"   • Multiple System Atrophy (MSA)        : {msa_flagged}/{len(msa_rows)} ({msa_flagged/len(msa_rows)*100:.1f}%) manifest dysphonia")
        if psp_rows:
            psp_flagged = sum(1 for r in psp_rows if r["result"] == "PD_FLAGGED")
            print(f"   • Progressive Supranuclear Palsy (PSP) : {psp_flagged}/{len(psp_rows)} ({psp_flagged/len(psp_rows)*100:.1f}%) manifest dysphonia")
        print("--------------------------------------------------------------------------------")

    # Save to output CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "subject_id",
            "cohort",
            "num_files",
            "mean_recording_p",
            "meta_score",
            "triage_tier",
            "diagnosis",
            "result",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[OK] Detailed results saved to: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()
