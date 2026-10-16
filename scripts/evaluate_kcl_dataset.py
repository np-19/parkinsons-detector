"""Evaluation script for the King's College London (KCL) Parkinson's Speech Dataset.

Dataset structure: test_samples/26-29_09_2017_KCL
Tasks: ReadText (reading passage) & SpontaneousDialogue (free speech)
Subjects: 38 participants (22 Healthy Controls, 16 Parkinson's Patients)
Recordings: 73 long continuous speech audio files (~150 seconds each at 44.1 kHz)

The pipeline extracts temporal windows across the speech tasks, resamples to 16 kHz,
evaluates using the fine-tuned Wav2Vec2 checkpoint, and performs clinical subject-level
aggregation to evaluate cross-dataset, cross-task generalization.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.deep_learning.model import Wav2Vec2ForParkinsons


def evaluate_kcl(
    dataset_dir: str = "test_samples/26-29_09_2017_KCL",
    model_path: str = "artifacts/models/deep_learning/wav2vec2_v2/best_model.pt",
    output_dir: str = "artifacts/models/deep_learning/wav2vec2_v2",
    duration: float = 8.0,
    target_sr: int = 16000,
    crops_per_recording: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    device = torch.device(device)
    print("=" * 65)
    print("   EVALUATING KING'S COLLEGE LONDON (KCL) SPEECH DATASET")
    print("=" * 65)
    print(f"Dataset Directory:     {dataset_dir}")
    print(f"Model Checkpoint:      {model_path}")
    print(f"Target Window:         {duration}s at {target_sr} Hz")
    print(f"Crops per Audio File:  {crops_per_recording} windows across speech")
    print(f"Inference Device:      {device}")
    print("=" * 65 + "\n")

    base_dir = Path(dataset_dir)
    wav_files = sorted(list(base_dir.rglob("*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {dataset_dir}")

    items = []
    for f in wav_files:
        stem = f.stem
        # Filename format e.g. ID00_hc_0_0_0.wav or ID02_pd_2_0_0.wav
        parts = stem.split("_")
        subject_id = parts[0]
        is_pd = 1 if "_pd_" in stem.lower() else 0
        task_name = f.parent.parent.name  # 'ReadText' or 'SpontaneousDialogue'

        items.append({
            "path": str(f),
            "filename": f.name,
            "subject_id": subject_id,
            "task": task_name,
            "true_label": is_pd,
            "class_name": "Parkinsons" if is_pd == 1 else "Healthy",
        })

    unique_subjects = set((item["subject_id"], item["true_label"]) for item in items)
    n_hc = sum(1 for _, l in unique_subjects if l == 0)
    n_pd = sum(1 for _, l in unique_subjects if l == 1)

    print(f"Found {len(items)} audio recordings across {len(unique_subjects)} unique participants.")
    print(f"  - Healthy Control (HC) subjects: {n_hc}")
    print(f"  - Parkinson's Disease (PD) subjects: {n_pd}\n")

    # Load Model
    print(f"Loading Wav2Vec2 model checkpoint from: {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    num_frozen = config.get("num_frozen_layers", checkpoint.get("num_frozen_layers", 4))
    dropout = config.get("dropout", checkpoint.get("dropout", 0.3))

    model = Wav2Vec2ForParkinsons(
        num_frozen_layers=num_frozen,
        dropout=dropout,
        gradient_checkpointing=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    target_samples = int(duration * target_sr)

    # Run inference across audio files
    recording_records = []
    print(f"\nEvaluating recordings with {crops_per_recording} temporal windows each...")

    with torch.no_grad():
        for item in tqdm(items, desc="Evaluating KCL Speech"):
            path = item["path"]
            try:
                waveform, sr = torchaudio.load(path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                    waveform = resampler(waveform)

                waveform = waveform.squeeze(0)
                n_samples = waveform.shape[0]
                dur_sec = n_samples / target_sr

                crops = []
                if n_samples <= target_samples:
                    padded = F.pad(waveform, (0, target_samples - n_samples))
                    crops = [padded] * crops_per_recording
                else:
                    # Space out crops evenly across speech duration
                    max_start = n_samples - target_samples
                    step = max_start / max(crops_per_recording - 1, 1)
                    for i in range(crops_per_recording):
                        start = int(i * step)
                        crops.append(waveform[start : start + target_samples])

                batch_crops = torch.stack(crops, dim=0).to(device)
                outputs = model(input_values=batch_crops)
                probs = outputs["probabilities"][:, 1].cpu().numpy()

                rec_mean_prob = float(np.mean(probs))
                rec_top2_prob = float(np.mean(sorted(probs)[-2:]))
                rec_max_prob = float(np.max(probs))

                recording_records.append({
                    "subject_id": item["subject_id"],
                    "filename": item["filename"],
                    "task": item["task"],
                    "duration_seconds": round(dur_sec, 1),
                    "true_label": item["true_label"],
                    "class_name": item["class_name"],
                    "mean_prob_PD": round(rec_mean_prob, 4),
                    "top2_prob_PD": round(rec_top2_prob, 4),
                    "max_prob_PD": round(rec_max_prob, 4),
                    "window_probs": [round(float(p), 4) for p in probs],
                })
            except Exception as e:
                print(f"Error reading {path}: {e}")

    df_recs = pd.DataFrame(recording_records)

    # 1. Recording-Level Metrics
    y_true_rec = df_recs["true_label"].values
    y_prob_rec = df_recs["top2_prob_PD"].values

    def compute_stats(y_t, y_p, thresh=0.360):
        preds = (y_p >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, preds).ravel()
        return {
            "threshold": round(thresh, 3),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_t, preds) * 100), 2),
            "accuracy": round(float(accuracy_score(y_t, preds) * 100), 2),
            "sensitivity": round(float(tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0, 2),
            "specificity": round(float(tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0, 2),
            "f1_score": round(float(f1_score(y_t, preds)), 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        }

    # 2. Subject-Level Metrics (Clinical Aggregation across tasks)
    subject_df = df_recs.groupby("subject_id").agg({
        "true_label": "first",
        "top2_prob_PD": "mean",
        "mean_prob_PD": "mean",
        "max_prob_PD": "max",
    }).reset_index()

    y_true_subj = subject_df["true_label"].values
    y_prob_subj = subject_df["top2_prob_PD"].values

    auc_subj = round(float(roc_auc_score(y_true_subj, y_prob_subj)), 4)
    auc_rec = round(float(roc_auc_score(y_true_rec, y_prob_rec)), 4)

    # Best threshold search
    best_thresh = 0.36
    best_bal = 0.0
    for t_val in np.linspace(0.15, 0.85, 71):
        p_t = (y_prob_subj >= t_val).astype(int)
        bal = balanced_accuracy_score(y_true_subj, p_t)
        if bal > best_bal:
            best_bal = bal
            best_thresh = float(t_val)

    metrics_subj_default = compute_stats(y_true_subj, y_prob_subj, thresh=0.360)
    metrics_subj_opt = compute_stats(y_true_subj, y_prob_subj, thresh=best_thresh)

    print("\n" + "=" * 65)
    print("      KCL SPEECH DATASET EVALUATION RESULTS")
    print("=" * 65)
    print("--- 1. Subject-Level Metrics (Primary Benchmark: 38 Participants) ---")
    print(f"  Participants:             {len(y_true_subj)} (22 Healthy, 16 Parkinson's)")
    print(f"  Subject ROC-AUC:          {auc_subj:.4f}")
    print(f"  Balanced Accuracy:        {metrics_subj_default['balanced_accuracy']}%  (Optimal Thresh: {metrics_subj_opt['balanced_accuracy']}%)")
    print(f"  Raw Accuracy:             {metrics_subj_default['accuracy']}%")
    print(f"  Sensitivity (PD Recall):  {metrics_subj_default['sensitivity']}% (Catches {metrics_subj_default['confusion_matrix']['TP']}/{n_pd} PD Patients)")
    print(f"  Specificity (HC Recall):  {metrics_subj_default['specificity']}% ({metrics_subj_default['confusion_matrix']['TN']}/{n_hc} Controls Correct)")
    print(f"  Confusion Matrix:         TN={metrics_subj_default['confusion_matrix']['TN']}, FP={metrics_subj_default['confusion_matrix']['FP']}, "
          f"FN={metrics_subj_default['confusion_matrix']['FN']}, TP={metrics_subj_default['confusion_matrix']['TP']}")
    print("-" * 65)
    print("--- 2. Recording-Level Metrics (73 Long Speech Files) ---")
    print(f"  Recording ROC-AUC:        {auc_rec:.4f}")
    print(f"  Balanced Accuracy:        {compute_stats(y_true_rec, y_prob_rec, thresh=0.360)['balanced_accuracy']}%")
    print("=" * 65 + "\n")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "kcl_speech_evaluation_report.json"
    preds_path = out_dir / "kcl_speech_predictions.csv"

    report = {
        "dataset": "King's College London (KCL) Speech Dataset",
        "total_recordings": len(df_recs),
        "total_subjects": len(subject_df),
        "subject_level": {
            "roc_auc": auc_subj,
            "metrics_at_0_360": metrics_subj_default,
            "metrics_at_optimal_threshold": metrics_subj_opt,
        },
        "recording_level": {
            "roc_auc": auc_rec,
            "metrics_at_0_360": compute_stats(y_true_rec, y_prob_rec, thresh=0.360),
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    df_recs.to_csv(preds_path, index=False)
    print(f"Predictions saved to: {preds_path}")
    print(f"Summary JSON saved to:  {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate KCL speech dataset")
    parser.add_argument("--dataset-dir", default="test_samples/26-29_09_2017_KCL")
    parser.add_argument("--model-path", default="artifacts/models/deep_learning/wav2vec2_v2/best_model.pt")
    parser.add_argument("--output-dir", default="artifacts/models/deep_learning/wav2vec2_v2")
    parser.add_argument("--crops", type=int, default=10)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    evaluate_kcl(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        crops_per_recording=args.crops,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()

