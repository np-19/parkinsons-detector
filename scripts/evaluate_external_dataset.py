"""Evaluation script for external Parkinson's voice datasets (e.g. test_samples/Voice_Dataset).

Loads the trained Wav2Vec2 v2 checkpoint, iterates through external audio samples,
automatically resamples them to 16 kHz mono, runs inference, and computes complete
classification metrics (Balanced Accuracy, Sensitivity, Specificity, ROC-AUC, F1).
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


def evaluate_external(
    dataset_dir: str = "test_samples/Voice_Dataset",
    model_path: str = "artifacts/models/deep_learning/wav2vec2_v2/best_model.pt",
    output_dir: str = "artifacts/models/deep_learning/wav2vec2_v2",
    duration: float = 8.0,
    target_sr: int = 16000,
    num_crops: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8,
) -> Dict:
    device = torch.device(device)
    print(f"============================================================")
    print(f"   EVALUATING EXTERNAL VOICE DATASET WITH WAV2VEC2 V2")
    print(f"============================================================")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Model Checkpoint:  {model_path}")
    print(f"Target SR:         {target_sr} Hz | Duration: {duration}s")
    print(f"Inference Device:  {device}")
    print(f"============================================================\n")

    # Discover audio files
    base_dir = Path(dataset_dir)
    healthy_files = sorted(list((base_dir / "Healthy").glob("*.wav")))
    pd_files = sorted(list((base_dir / "Parkinsons").glob("*.wav")))

    if not healthy_files and not pd_files:
        # Fallback to general rglob if folder names differ
        all_wavs = list(base_dir.rglob("*.wav"))
        print(f"Found {len(all_wavs)} WAV files in total.")
    else:
        print(f"Found {len(healthy_files)} Healthy files and {len(pd_files)} Parkinson's files.")

    items = []
    for f in healthy_files:
        items.append({"path": str(f), "filename": f.name, "label": 0, "class_name": "Healthy"})
    for f in pd_files:
        items.append({"path": str(f), "filename": f.name, "label": 1, "class_name": "Parkinsons"})

    if not items:
        raise FileNotFoundError(f"No audio files found in {dataset_dir}")

    # Load Model
    print(f"\nLoading model checkpoint from {model_path}...")
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

    # Run inference
    predictions = []
    print(f"\nRunning Multi-Crop TTA ({num_crops} crops) across {len(items)} audio recordings...")

    with torch.no_grad():
        for item in tqdm(items, desc="Evaluating"):
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

                # Multi-crop TTA
                crops = []
                if n_samples <= target_samples:
                    pad_amount = target_samples - n_samples
                    padded = F.pad(waveform, (0, pad_amount))
                    for _ in range(num_crops):
                        crops.append(padded)
                else:
                    max_start = n_samples - target_samples
                    step = max_start / max(num_crops - 1, 1)
                    for i in range(num_crops):
                        start = int(i * step)
                        crops.append(waveform[start : start + target_samples])

                batch_crops = torch.stack(crops, dim=0).to(device)
                outputs = model(input_values=batch_crops)
                probs = outputs["probabilities"][:, 1]
                rec_prob = float(probs.mean().item())

                predictions.append({
                    "filename": item["filename"],
                    "path": item["path"],
                    "true_label": item["label"],
                    "class_name": item["class_name"],
                    "prob_PD": rec_prob,
                    "pred_label_default": 1 if rec_prob >= 0.50 else 0,
                    "pred_label_opt": 1 if rec_prob >= 0.360 else 0,
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")

    df_preds = pd.DataFrame(predictions)
    y_true = df_preds["true_label"].values
    y_prob = df_preds["prob_PD"].values

    # Find optimal threshold on this dataset
    best_thresh = 0.50
    best_bal = 0.0
    for t_val in np.linspace(0.10, 0.90, 81):
        preds_t = (y_prob >= t_val).astype(int)
        bal = balanced_accuracy_score(y_true, preds_t)
        if bal > best_bal:
            best_bal = bal
            best_thresh = float(t_val)

    def calc_metrics_at_thresh(thresh):
        preds = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        return {
            "threshold": round(thresh, 3),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds) * 100), 2),
            "accuracy": round(float(accuracy_score(y_true, preds) * 100), 2),
            "sensitivity": round(float(tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0, 2),
            "specificity": round(float(tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0, 2),
            "f1_score": round(float(f1_score(y_true, preds)), 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        }

    auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    metrics_360 = calc_metrics_at_thresh(0.360)
    metrics_500 = calc_metrics_at_thresh(0.500)
    metrics_opt = calc_metrics_at_thresh(best_thresh)

    print("\n" + "=" * 65)
    print("      EXTERNAL DATASET EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Audio Samples Evaluated: {len(y_true)}")
    print(f"  - Healthy Controls:          {int(sum(y_true == 0))}")
    print(f"  - Parkinson's Patients:      {int(sum(y_true == 1))}")
    print(f"Area Under ROC Curve (ROC-AUC): {auc:.4f}")
    print("-" * 65)
    print(f"--- At Model Default Validation Threshold (0.360) ---")
    print(f"  Balanced Accuracy:           {metrics_360['balanced_accuracy']}%")
    print(f"  Raw Accuracy:                {metrics_360['accuracy']}%")
    print(f"  Sensitivity (PD Recall):     {metrics_360['sensitivity']}% (TP={metrics_360['confusion_matrix']['TP']}, FN={metrics_360['confusion_matrix']['FN']})")
    print(f"  Specificity (Ctrl Recall):   {metrics_360['specificity']}% (TN={metrics_360['confusion_matrix']['TN']}, FP={metrics_360['confusion_matrix']['FP']})")
    print(f"  F1 Score:                    {metrics_360['f1_score']}")
    print("-" * 65)
    print(f"--- At Standard 0.500 Threshold ---")
    print(f"  Balanced Accuracy:           {metrics_500['balanced_accuracy']}%")
    print(f"  Raw Accuracy:                {metrics_500['accuracy']}%")
    print(f"  Sensitivity:                 {metrics_500['sensitivity']}%")
    print(f"  Specificity:                 {metrics_500['specificity']}%")
    print("-" * 65)
    print(f"--- At Optimal Dataset Threshold ({best_thresh:.3f}) ---")
    print(f"  Balanced Accuracy:           {metrics_opt['balanced_accuracy']}%")
    print(f"  Raw Accuracy:                {metrics_opt['accuracy']}%")
    print(f"  Sensitivity:                 {metrics_opt['sensitivity']}%")
    print(f"  Specificity:                 {metrics_opt['specificity']}%")
    print("=" * 65 + "\n")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "external_voice_dataset_report.json"
    preds_path = out_dir / "external_voice_dataset_predictions.csv"

    full_report = {
        "dataset_directory": dataset_dir,
        "model_checkpoint": model_path,
        "total_samples": len(y_true),
        "class_counts": {"healthy": int(sum(y_true == 0)), "parkinsons": int(sum(y_true == 1))},
        "roc_auc": auc,
        "metrics_at_val_threshold_0_360": metrics_360,
        "metrics_at_standard_0_500": metrics_500,
        "metrics_at_optimal_threshold": metrics_opt,
    }

    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    df_preds.to_csv(preds_path, index=False)
    print(f"Detailed predictions saved to: {preds_path}")
    print(f"Summary JSON report saved to:  {report_path}")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate external voice dataset with Wav2Vec2")
    parser.add_argument("--dataset-dir", default="test_samples/Voice_Dataset")
    parser.add_argument("--model-path", default="artifacts/models/deep_learning/wav2vec2_v2/best_model.pt")
    parser.add_argument("--output-dir", default="artifacts/models/deep_learning/wav2vec2_v2")
    parser.add_argument("--num-crops", type=int, default=5)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    evaluate_external(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_crops=args.num_crops,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()

