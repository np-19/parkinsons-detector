"""Training and Cross-Validation Pipeline for Gated Attention-MIL.

Evaluates Gated Attention-MIL against the count-invariant arithmetic mean
baseline on participant bags under 5-Fold Stratified Cross-Validation,
computing participant-level metrics with 1,000-resample bootstrap 95% CIs.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mil.model import GatedAttentionMIL


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
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
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find threshold in [0.05, 0.95] maximizing balanced accuracy."""
    best_thresh = 0.5
    best_bal = 0.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= t).astype(int)
        bal = balanced_accuracy_score(y_true, y_pred)
        if bal > best_bal:
            best_bal = bal
            best_thresh = t
    return float(best_thresh), float(best_bal)


def compute_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_bootstraps: int = 1000) -> Dict:
    """Compute 95% bootstrap confidence intervals for participant-level metrics."""
    n = len(y_true)
    rng = np.random.RandomState(42)
    boot_bal = []
    boot_auc = []
    boot_sens = []
    boot_spec = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_b = y_true[idx]
        yp_b = y_prob[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        m = compute_metrics(yt_b, yp_b, threshold)
        boot_bal.append(m["balanced_accuracy"])
        boot_auc.append(m["roc_auc"])
        boot_sens.append(m["sensitivity"])
        boot_spec.append(m["specificity"])

    def get_ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "balanced_accuracy_ci": get_ci(boot_bal),
        "roc_auc_ci": get_ci(boot_auc),
        "sensitivity_ci": get_ci(boot_sens),
        "specificity_ci": get_ci(boot_spec),
    }


def run_cross_validation(
    bags_path: Path,
    output_dir: Path,
    device: torch.device,
    n_splits: int = 5,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.25,
    pos_weight_val: float = 2.0,
    use_residual_mean: bool = False,
) -> Dict:
    """Run 5-Fold Stratified Participant Cross-Validation."""
    print(f"\n=======================================================")
    print(f" Loading Participant Bags: {bags_path}")
    print(f"=======================================================")

    bags_dict = torch.load(bags_path, map_location="cpu", weights_only=False)
    participant_list = list(bags_dict.values())
    print(f"Total participants: {len(participant_list)}")

    labels = np.array([p["label"] for p in participant_list])
    n_pd = int(np.sum(labels == 1))
    n_ctrl = int(np.sum(labels == 0))
    print(f"Class distribution: {n_pd} PD ({n_pd/len(labels)*100:.1f}%), {n_ctrl} Control ({n_ctrl/len(labels)*100:.1f}%)")

    # Baseline: Count-Invariant Arithmetic Mean of WaveLM probabilities
    baseline_probs = []
    for p in participant_list:
        # p["crop_probabilities"]: (N_crops,)
        mean_p = float(p["crop_probabilities"].mean().item())
        baseline_probs.append(mean_p)
    baseline_probs = np.array(baseline_probs)

    base_thresh, _ = find_optimal_threshold(labels, baseline_probs)
    base_metrics = compute_metrics(labels, baseline_probs, threshold=base_thresh)
    base_ci = compute_bootstrap_ci(labels, baseline_probs, threshold=base_thresh)
    base_metrics.update(base_ci)

    print("\n-------------------------------------------------------")
    print(f"BASELINE: Count-Invariant Arithmetic Mean (WaveLM)")
    print(f"Balanced Accuracy: {base_metrics['balanced_accuracy']*100:.2f}% [95% CI: {base_ci['balanced_accuracy_ci'][0]*100:.1f}%, {base_ci['balanced_accuracy_ci'][1]*100:.1f}%]")
    print(f"ROC-AUC:           {base_metrics['roc_auc']:.4f} [95% CI: {base_ci['roc_auc_ci'][0]:.3f}, {base_ci['roc_auc_ci'][1]:.3f}]")
    print(f"Sensitivity:       {base_metrics['sensitivity']*100:.2f}%")
    print(f"Specificity:       {base_metrics['specificity']*100:.2f}%")
    print(f"Threshold:         {base_metrics['threshold']:.3f}")
    print("-------------------------------------------------------")

    # Stratified 5-Fold Cross-Validation for Gated Attention-MIL
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(participant_list))
    oof_attention_weights = {}

    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    fold_metrics = []
    fold_states = []
    best_overall_bal = 0.0
    best_overall_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(participant_list, labels)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        train_bags = [participant_list[i] for i in train_idx]
        val_bags = [participant_list[i] for i in val_idx]

        model = GatedAttentionMIL(
            in_dim=768,
            attention_dim=128,
            hidden_dim=128,
            dropout=dropout,
            use_residual_mean=use_residual_mean,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_bal = 0.0
        best_val_state = None

        for epoch in range(epochs):
            model.train()
            random.shuffle(train_bags)
            optimizer.zero_grad()

            acc_loss = 0.0
            accum_steps = 8
            step_count = 0

            for p_bag in train_bags:
                # Shape: (N_crops, 768)
                h = p_bag["embeddings"].to(device)
                target = torch.tensor([float(p_bag["label"])], device=device)

                out = model(h)
                loss = criterion(out["logit"], target) / accum_steps
                loss.backward()
                acc_loss += loss.item() * accum_steps
                step_count += 1

                if step_count % accum_steps == 0 or step_count == len(train_bags):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()

            # Validation evaluation
            model.eval()
            val_probs = []
            val_true = []
            with torch.no_grad():
                for p_bag in val_bags:
                    h = p_bag["embeddings"].to(device)
                    out = model(h)
                    val_probs.append(float(out["probability"].item()))
                    val_true.append(p_bag["label"])

            val_probs = np.array(val_probs)
            val_true = np.array(val_true)
            _, val_bal = find_optimal_threshold(val_true, val_probs)

            if val_bal > best_val_bal:
                best_val_bal = val_bal
                best_val_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best fold model
        model.load_state_dict(best_val_state)
        model.to(device)
        model.eval()

        # Generate OOF predictions
        with torch.no_grad():
            for i, p_bag in zip(val_idx, val_bags):
                h = p_bag["embeddings"].to(device)
                out = model(h)
                oof_predictions[i] = float(out["probability"].item())
                oof_attention_weights[p_bag["health_code"]] = out["attention_weights"].cpu().numpy().tolist()

        fold_true = labels[val_idx]
        fold_probs = oof_predictions[val_idx]
        fold_thresh, fold_bal = find_optimal_threshold(fold_true, fold_probs)
        m_fold = compute_metrics(fold_true, fold_probs, threshold=fold_thresh)
        fold_metrics.append(m_fold)
        print(f"Fold {fold + 1} Best BalAcc: {fold_bal*100:.2f}% (ROC-AUC: {m_fold['roc_auc']:.4f})")
        fold_states.append(best_val_state)
        if fold_bal > best_overall_bal:
            best_overall_bal = fold_bal
            best_overall_state = best_val_state

    # Overall OOF Evaluation across all 579 participants
    opt_thresh, _ = find_optimal_threshold(labels, oof_predictions)
    mil_metrics = compute_metrics(labels, oof_predictions, threshold=opt_thresh)
    mil_ci = compute_bootstrap_ci(labels, oof_predictions, threshold=opt_thresh)
    mil_metrics.update(mil_ci)

    print("\n=======================================================")
    print(" GATED ATTENTION-MIL: Overall 5-Fold OOF Results")
    print(f"Balanced Accuracy: {mil_metrics['balanced_accuracy']*100:.2f}% [95% CI: {mil_ci['balanced_accuracy_ci'][0]*100:.1f}%, {mil_ci['balanced_accuracy_ci'][1]*100:.1f}%]")
    print(f"ROC-AUC:           {mil_metrics['roc_auc']:.4f} [95% CI: {mil_ci['roc_auc_ci'][0]:.3f}, {mil_ci['roc_auc_ci'][1]:.3f}]")
    print(f"Sensitivity:       {mil_metrics['sensitivity']*100:.2f}%")
    print(f"Specificity:       {mil_metrics['specificity']*100:.2f}%")
    print(f"Threshold:         {mil_metrics['threshold']:.3f}")
    print(f"Confusion Matrix:  {mil_metrics['confusion_matrix']}")
    print("=======================================================")

    report = {
        "dataset_participants": len(participant_list),
        "n_pd": n_pd,
        "n_control": n_ctrl,
        "baseline_arithmetic_mean": base_metrics,
        "gated_attention_mil": mil_metrics,
        "fold_metrics": fold_metrics,
        "delta": {
            "balanced_accuracy_diff": mil_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"],
            "roc_auc_diff": mil_metrics["roc_auc"] - base_metrics["roc_auc"],
            "sensitivity_diff": mil_metrics["sensitivity"] - base_metrics["sensitivity"],
            "specificity_diff": mil_metrics["specificity"] - base_metrics["specificity"],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "mil_evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Save OOF predictions CSV
    pred_df = pd.DataFrame({
        "healthCode": [p["health_code"] for p in participant_list],
        "label": labels,
        "baseline_prob": baseline_probs,
        "mil_prob": oof_predictions,
        "baseline_pred": (baseline_probs >= base_thresh).astype(int),
        "mil_pred": (oof_predictions >= opt_thresh).astype(int),
        "num_recordings": [p["num_recordings"] for p in participant_list],
        "num_instances": [p["num_instances"] for p in participant_list],
    })
    pred_df.to_csv(output_dir / "mil_oof_predictions.csv", index=False)

    # Save attention weights JSON
    with open(output_dir / "mil_attention_weights.json", "w") as f:
        json.dump(oof_attention_weights, f)

    # Export production Attention-MIL checkpoint
    mil_export = {
        "model_state_dict": best_overall_state,
        "fold_states": fold_states,
        "optimal_threshold": round(opt_thresh, 4),
        "screening_cutoff": round(max(0.08, opt_thresh - 0.05), 4),
        "diagnostic_cutoff": round(min(0.90, opt_thresh + 0.05), 4),
        "aggregation": "gated_attention_mil",
        "config": {
            "in_dim": 768,
            "attention_dim": 128,
            "hidden_dim": 128,
            "dropout": dropout,
            "use_residual_mean": use_residual_mean,
        },
        "metrics": mil_metrics,
    }
    torch.save(mil_export, output_dir / "mil_head.pt")
    target_dl_dir = Path("artifacts/models/deep_learning/wavelm")
    target_dl_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mil_export, target_dl_dir / "mil_head.pt")
    print(f"Exported production mil_head.pt to {output_dir / 'mil_head.pt'} and {target_dl_dir / 'mil_head.pt'}")

    print(f"Saved reports and predictions to {output_dir}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Gated Attention-MIL")
    parser.add_argument("--bags", type=str, default="artifacts/mil/bags_test.pt")
    parser.add_argument("--output-dir", type=str, default="artifacts/mil")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument("--use-residual-mean", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    dev = torch.device(args.device)
    run_cross_validation(
        bags_path=Path(args.bags),
        output_dir=Path(args.output_dir),
        device=dev,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        pos_weight_val=args.pos_weight,
        use_residual_mean=args.use_residual_mean,
    )

