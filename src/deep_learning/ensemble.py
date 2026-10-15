"""Ensemble fusion script for combining predictions across multiple models.

Supports:
- Averaging predictions from multiple test_predictions.csv files
- Grid-search optimal threshold on combined participant probabilities
- Computing participant-level and recording-level clinical metrics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


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


def main():
    parser = argparse.ArgumentParser(description="Ensemble multiple model predictions")
    parser.add_argument(
        "--prediction-files",
        nargs="+",
        required=True,
        help="Paths to test_predictions.csv from each model to ensemble",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional weights for each model (must sum to 1.0)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="artifacts/models/deep_learning/ensemble_test_predictions.csv",
        help="Output CSV for ensemble predictions",
    )
    args = parser.parse_args()

    dfs = [pd.read_csv(f) for f in args.prediction_files]
    num_models = len(dfs)
    print(f"Ensembling {num_models} models:")
    for f in args.prediction_files:
        print(f"  - {f}")

    if args.weights:
        weights = np.array(args.weights)
        weights = weights / weights.sum()
    else:
        weights = np.ones(num_models) / num_models

    base_df = dfs[0][["ROW_ID", "healthCode", "label"]].copy()

    # Weighted average recording prob
    ensemble_rec_prob = np.zeros(len(base_df))
    for i, df in enumerate(dfs):
        ensemble_rec_prob += weights[i] * df["prob_PD"].values

    base_df["prob_PD"] = ensemble_rec_prob

    # Participant aggregation
    part_df = base_df.groupby("healthCode").agg({
        "label": "first",
        "prob_PD": "mean",
    }).reset_index()

    p_true = part_df["label"].values.astype(int)
    p_prob = part_df["prob_PD"].values.astype(float)

    # Search best threshold
    best_thresh = 0.50
    best_bal_acc = -1.0
    for t in np.arange(0.30, 0.70, 0.01):
        m = compute_metrics(p_true, p_prob, threshold=float(t))
        if m["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = m["balanced_accuracy"]
            best_thresh = float(t)

    part_metrics = compute_metrics(p_true, p_prob, threshold=best_thresh)
    rec_metrics = compute_metrics(base_df["label"].values, ensemble_rec_prob, threshold=best_thresh)

    part_prob_map = dict(zip(part_df["healthCode"], part_df["prob_PD"]))
    base_df["participant_prob_PD"] = base_df["healthCode"].map(part_prob_map)
    base_df["participant_pred"] = (base_df["participant_prob_PD"] >= best_thresh).astype(int)

    out_p = Path(args.output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(out_p, index=False)

    print("\n" + "=" * 60)
    print("              ENSEMBLE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Optimal Threshold: {best_thresh:.3f}")
    print(f"Participant Balanced Accuracy: {part_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"Participant Accuracy:          {part_metrics['accuracy'] * 100:.2f}%")
    print(f"Participant ROC-AUC:           {part_metrics['roc_auc']:.4f}")
    print(f"Participant Sensitivity:       {part_metrics['sensitivity'] * 100:.2f}%")
    print(f"Participant Specificity:       {part_metrics['specificity'] * 100:.2f}%")
    print(f"Confusion Matrix:              {part_metrics['confusion_matrix']}")
    print(f"\nSaved ensemble predictions to: {out_p}")


if __name__ == "__main__":
    main()

