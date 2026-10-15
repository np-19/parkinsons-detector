"""One-time evaluation script for the fine-tuned Parkinson's deep learning model.

Evaluates the frozen best model checkpoint on the held-out test split,
producing recording-level and participant-aggregated clinical metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

from .dataset import MPowerAudioDataset, create_dataloader
from .model import Wav2Vec2ForParkinsons


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on held-out test split")
    parser.add_argument("--model-dir", type=str, default="artifacts/models/deep_learning/wav2vec2_finetuned", help="Directory containing best_model.pt")
    parser.add_argument("--test-metadata", type=str, default="data/metadata/test.csv", help="Path to test.csv")
    parser.add_argument("--audio-root", type=str, default="data/processed_audio", help="Audio root directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--duration", type=float, default=5.0, help="Audio segment duration in seconds")
    parser.add_argument("--num-crops", type=int, default=1, help="Number of overlapping crops per recording for TTA (1=center crop only)")
    parser.add_argument("--crop-stride", type=float, default=1.5, help="Stride in seconds between crops for multi-crop TTA")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Cap test samples (optional for sanity check)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Held-Out Test Evaluation ===")
    print(f"Device: {device}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    optimal_threshold = checkpoint.get("optimal_threshold", 0.50)
    best_epoch = checkpoint.get("epoch", -1)

    print(f"Checkpoint was trained for {best_epoch} epochs.")
    print(f"Loaded validation-selected optimal threshold: {optimal_threshold:.3f}")

    # Initialize model
    model = Wav2Vec2ForParkinsons(
        model_name=config.get("model_name", "facebook/wav2vec2-base"),
        num_classes=2,
        num_frozen_layers=config.get("num_frozen_layers", 6),
        freeze_feature_extractor=True,
        gradient_checkpointing=False,  # No checkpointing needed for eval
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Dataset
    failures_file = Path("data/metadata/processed_audio_failures.csv")
    test_ds = MPowerAudioDataset(
        metadata_file=args.test_metadata,
        audio_root=args.audio_root,
        target_duration=args.duration,
        is_training=False,
        augment=False,
        failures_file=failures_file if failures_file.exists() else None,
        max_samples=args.max_test_samples,
    )
    test_loader = create_dataloader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Evaluating on {len(test_ds)} test recordings (Multi-crop TTA: {args.num_crops})...")

    all_probs = []
    all_labels = []
    all_health_codes = []
    all_row_ids = []

    with torch.no_grad():
        if args.num_crops > 1:
            target_samples = int(args.duration * 16000)
            for rec in tqdm(test_ds.records, desc=f"Testing (TTA {args.num_crops}-crops)"):
                data, _ = sf.read(str(rec["path"]), dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)

                crops = []
                total_len = len(data)
                if total_len <= target_samples:
                    pad_len = target_samples - total_len
                    crop = np.pad(data, (0, pad_len), mode="constant")
                    crops.append(crop)
                else:
                    max_offset = total_len - target_samples
                    offsets = np.linspace(0, max_offset, num=args.num_crops, dtype=int)
                    for start_idx in offsets:
                        crop = data[start_idx : start_idx + target_samples]
                        crops.append(crop)

                # Normalize each crop
                norm_crops = []
                for c in crops:
                    max_v = np.max(np.abs(c))
                    if max_v > 1e-6:
                        c = c / max_v
                    norm_crops.append(c.astype(np.float32))

                crop_tensor = torch.from_numpy(np.stack(norm_crops)).to(device)
                outputs = model(input_values=crop_tensor)
                # Average probability across all crops of this recording
                rec_prob = outputs["probabilities"][:, 1].mean().item()

                all_probs.append(rec_prob)
                all_labels.append(rec["label"])
                all_health_codes.append(rec["health_code"])
                all_row_ids.append(rec["row_id"])

            y_true = np.array(all_labels, dtype=int)
            y_prob = np.array(all_probs, dtype=float)
        else:
            for batch in tqdm(test_loader, desc="Testing"):
                input_values = batch["input_values"].to(device)
                outputs = model(input_values=input_values)
                probs = outputs["probabilities"][:, 1]  # PD positive class probability

                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
                all_health_codes.extend(batch["health_codes"])
                all_row_ids.extend(batch["row_ids"])

            y_true = np.concatenate(all_labels)
            y_prob = np.concatenate(all_probs)

    # Recording-level metrics
    rec_metrics_opt = compute_metrics(y_true, y_prob, threshold=optimal_threshold)
    rec_metrics_default = compute_metrics(y_true, y_prob, threshold=0.50)

    # Participant-level aggregation
    pred_df = pd.DataFrame({
        "ROW_ID": all_row_ids,
        "healthCode": all_health_codes,
        "label": y_true,
        "prob_PD": y_prob,
        "recording_pred": (y_prob >= optimal_threshold).astype(int),
    })

    part_df = pred_df.groupby("healthCode").agg({
        "label": "first",
        "prob_PD": "mean",
    }).reset_index()

    p_true = part_df["label"].values.astype(int)
    p_prob = part_df["prob_PD"].values.astype(float)

    part_metrics_opt = compute_metrics(p_true, p_prob, threshold=optimal_threshold)
    part_metrics_default = compute_metrics(p_true, p_prob, threshold=0.50)

    # Attach participant prediction back to pred_df
    part_prob_map = dict(zip(part_df["healthCode"], part_df["prob_PD"]))
    pred_df["participant_prob_PD"] = pred_df["healthCode"].map(part_prob_map)
    pred_df["participant_pred"] = (pred_df["participant_prob_PD"] >= optimal_threshold).astype(int)

    # Display results
    print("\n" + "=" * 60)
    print("           HELD-OUT TEST SPLIT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Optimal Threshold (selected on val): {optimal_threshold:.3f}\n")
    print("--- Participant-Level Metrics (Primary Benchmark) ---")
    print(f"  Participants Evaluated:   {len(p_true)}")
    print(f"  Balanced Accuracy:        {part_metrics_opt['balanced_accuracy'] * 100:.2f}%")
    print(f"  Accuracy:                 {part_metrics_opt['accuracy'] * 100:.2f}%")
    print(f"  Sensitivity (PD Recall):  {part_metrics_opt['sensitivity'] * 100:.2f}%")
    print(f"  Specificity (Ctrl Rec):   {part_metrics_opt['specificity'] * 100:.2f}%")
    print(f"  F1 Score:                 {part_metrics_opt['f1']:.4f}")
    print(f"  ROC-AUC:                  {part_metrics_opt['roc_auc']:.4f}")
    print(f"  Confusion Matrix:         TN={part_metrics_opt['confusion_matrix']['tn']}, FP={part_metrics_opt['confusion_matrix']['fp']}, "
          f"FN={part_metrics_opt['confusion_matrix']['fn']}, TP={part_metrics_opt['confusion_matrix']['tp']}")

    print("\n--- Recording-Level Metrics ---")
    print(f"  Recordings Evaluated:     {len(y_true)}")
    print(f"  Balanced Accuracy:        {rec_metrics_opt['balanced_accuracy'] * 100:.2f}%")
    print(f"  Accuracy:                 {rec_metrics_opt['accuracy'] * 100:.2f}%")
    print(f"  Sensitivity:              {rec_metrics_opt['sensitivity'] * 100:.2f}%")
    print(f"  Specificity:              {rec_metrics_opt['specificity'] * 100:.2f}%")
    print(f"  ROC-AUC:                  {rec_metrics_opt['roc_auc']:.4f}")
    print("=" * 60)

    # Save reports
    report = {
        "model_dir": str(model_dir),
        "best_epoch": best_epoch,
        "optimal_threshold": optimal_threshold,
        "participant_level": {
            "optimal_threshold_metrics": part_metrics_opt,
            "default_threshold_metrics": part_metrics_default,
            "total_participants": len(p_true),
        },
        "recording_level": {
            "optimal_threshold_metrics": rec_metrics_opt,
            "default_threshold_metrics": rec_metrics_default,
            "total_recordings": len(y_true),
        },
    }

    report_path = model_dir / "test_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    predictions_path = model_dir / "test_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    print(f"\nEvaluation report written to {report_path}")
    print(f"Detailed predictions written to {predictions_path}")


if __name__ == "__main__":
    main()

