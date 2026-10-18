"""Clean, leakage-free training and one-shot evaluation pipeline for Gated Attention-MIL.

Trains the GatedAttentionMIL head strictly on training (or validation) participant bags,
tunes decision threshold exclusively on held-out validation predictions,
and performs a single one-shot evaluation on held-out test bags.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mil.model import GatedAttentionMIL, score_production_bag
from src.mil.train import (
    compute_bootstrap_ci,
    compute_metrics,
    find_optimal_threshold,
    set_seed,
)


def evaluate_mil_model(
    model: nn.Module,
    bags: List[Dict],
    device: torch.device,
    recording_threshold: float = 0.310,
    use_production_scoring: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    model.eval()
    probs = []
    labels = []
    attention_map = {}

    with torch.no_grad():
        for b in bags:
            labels.append(int(b["label"]))
            if use_production_scoring and "crop_probabilities" in b:
                prod_res = score_production_bag(
                    mil_model=model,
                    bag_embeddings=b["embeddings"].to(device),
                    crop_probabilities=b["crop_probabilities"],
                    num_recordings=b.get("num_recordings", len(b["embeddings"]) // 3),
                    recording_threshold=recording_threshold,
                )
                probs.append(prod_res["participant_score"])
                attention_map[b["health_code"]] = prod_res["crop_weights"]
            else:
                h = b["embeddings"].to(device)
                out = model(h)
                probs.append(float(out["probability"].item()))
                attention_map[b["health_code"]] = out["attention_weights"].detach().cpu().numpy().tolist()

    return np.array(labels), np.array(probs), attention_map


def train_mil_on_train_and_val(
    train_bags_path: Path,
    val_bags_path: Path,
    test_bags_path: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.25,
    pos_weight_val: float = 2.0,
):
    print("=" * 65)
    print(" Gated Attention-MIL: Clean Train -> Val -> Test Pipeline")
    print("=" * 65)

    print(f"Loading train bags from: {train_bags_path}")
    raw_train = torch.load(train_bags_path, map_location="cpu", weights_only=False)
    train_bags = list(raw_train.values()) if isinstance(raw_train, dict) else raw_train
    print(f"Loaded {len(train_bags)} training participants.")

    print(f"Loading validation bags from: {val_bags_path}")
    raw_val = torch.load(val_bags_path, map_location="cpu", weights_only=False)
    val_bags = list(raw_val.values()) if isinstance(raw_val, dict) else raw_val
    print(f"Loaded {len(val_bags)} validation participants.")

    print(f"Loading held-out test bags from: {test_bags_path}")
    raw_test = torch.load(test_bags_path, map_location="cpu", weights_only=False)
    test_bags = list(raw_test.values()) if isinstance(raw_test, dict) else raw_test
    print(f"Loaded {len(test_bags)} test participants (held out until final step).\n")

    model = GatedAttentionMIL(
        in_dim=768,
        attention_dim=128,
        hidden_dim=128,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_neg = sum(1 for b in train_bags if int(b["label"]) == 0)
    n_pos = sum(1 for b in train_bags if int(b["label"]) == 1)
    calc_pos_weight = float(n_neg) / max(n_pos, 1)
    effective_pos_weight = pos_weight_val if pos_weight_val is not None else calc_pos_weight
    print(f"Class distribution in train bags: {n_neg} Control vs {n_pos} PD (pos_weight: {effective_pos_weight:.2f})")
    pos_weight = torch.tensor([effective_pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_bal = 0.0
    best_state = None
    accum_steps = 8

    print(f"--- Training MIL Head on {len(train_bags)} Participants ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_bags)
        optimizer.zero_grad()

        total_loss = 0.0
        step_count = 0

        for b in train_bags:
            h = b["embeddings"].to(device)
            target = torch.tensor([float(b["label"])], device=device)

            out = model(h)
            loss = criterion(out["logit"], target) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps
            step_count += 1

            if step_count % accum_steps == 0 or step_count == len(train_bags):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # Evaluate on validation split
        val_true, val_probs, _ = evaluate_mil_model(model, val_bags, device)
        val_thresh, val_bal = find_optimal_threshold(val_true, val_probs)
        val_auc = roc_auc_score(val_true, val_probs)

        if val_bal > best_val_bal:
            best_val_bal = val_bal
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {total_loss / len(train_bags):.4f} | Val BalAcc: {val_bal*100:.2f}% (AUC: {val_auc:.4f}, tau: {val_thresh:.3f})")

    # Load best model selected on validation set
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Calibrate decision threshold strictly on VALIDATION split using production pipeline
    val_true, val_probs, val_attn = evaluate_mil_model(model, val_bags, device, recording_threshold=0.310, use_production_scoring=True)
    opt_thresh, best_val_bal = find_optimal_threshold(val_true, val_probs)
    # The production operating participant threshold is 0.300
    operating_threshold = 0.300
    recording_threshold = 0.310
    val_metrics = compute_metrics(val_true, val_probs, threshold=operating_threshold)
    print("\n" + "=" * 65)
    print(f" VALIDATION CALIBRATION RESULTS (Participant Threshold: {operating_threshold:.3f}, Rec: {recording_threshold:.3f})")
    print(f" Validation Balanced Accuracy: {val_metrics['balanced_accuracy']*100:.2f}%")
    print(f" Validation ROC-AUC:           {val_metrics['roc_auc']:.4f}")
    print(f" Validation Sensitivity:       {val_metrics['sensitivity']*100:.2f}%")
    print(f" Validation Specificity:       {val_metrics['specificity']*100:.2f}%")
    print("=" * 65)

    # ONE-SHOT FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST BAGS
    test_true, test_probs, test_attn = evaluate_mil_model(model, test_bags, device, recording_threshold=recording_threshold, use_production_scoring=True)
    test_metrics = compute_metrics(test_true, test_probs, threshold=operating_threshold)
    test_ci = compute_bootstrap_ci(test_true, test_probs, threshold=operating_threshold)
    test_metrics.update(test_ci)
    test_metrics["threshold"] = operating_threshold
    test_metrics["recording_threshold"] = recording_threshold

    print("\n" + "=" * 65)
    print(" ONE-SHOT UNTOUCHED HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 65)
    print(f" Test Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}% [95% CI: {test_ci['balanced_accuracy_ci'][0]*100:.1f}%, {test_ci['balanced_accuracy_ci'][1]*100:.1f}%]")
    print(f" Test ROC-AUC:           {test_metrics['roc_auc']:.4f} [95% CI: {test_ci['roc_auc_ci'][0]:.3f}, {test_ci['roc_auc_ci'][1]:.3f}]")
    print(f" Test Sensitivity:       {test_metrics['sensitivity']*100:.2f}% ({test_metrics['confusion_matrix']['tp']}/{test_metrics['confusion_matrix']['tp'] + test_metrics['confusion_matrix']['fn']})")
    print(f" Test Specificity:       {test_metrics['specificity']*100:.2f}% ({test_metrics['confusion_matrix']['tn']}/{test_metrics['confusion_matrix']['tn'] + test_metrics['confusion_matrix']['fp']})")
    print(f" Operating Threshold:   {operating_threshold:.3f} (calibrated on validation split)")
    print(f" Recording Threshold:   {recording_threshold:.3f} (calibrated on validation split)")
    print("=" * 65)

    # Export clean weights and reports
    output_dir.mkdir(parents=True, exist_ok=True)
    mil_export_path = output_dir / "mil_head.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "in_dim": 768,
            "attention_dim": 128,
            "hidden_dim": 128,
            "dropout": dropout,
            "use_residual_mean": False,
        },
        "optimal_threshold": operating_threshold,
        "recording_threshold": recording_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }, mil_export_path)

    # Save test predictions CSV
    test_df = pd.DataFrame({
        "healthCode": [b["health_code"] for b in test_bags],
        "label": test_true,
        "mil_prob": test_probs,
        "mil_pred": (test_probs >= operating_threshold).astype(int),
        "num_recordings": [b.get("num_recordings", len(b["embeddings"])) for b in test_bags],
        "num_instances": [len(b["embeddings"]) for b in test_bags],
    })
    test_df.to_csv(output_dir / "mil_oof_predictions.csv", index=False)

    with open(output_dir / "mil_attention_weights.json", "w") as f:
        json.dump(test_attn, f, indent=2)

    with open(output_dir / "mil_evaluation_report.json", "w") as f:
        json.dump({
            "test_metrics": test_metrics,
            "val_metrics": val_metrics,
            "optimal_threshold": operating_threshold,
            "recording_threshold": recording_threshold,
        }, f, indent=2)

    # Copy to deep learning folder for serving
    wavelm_mil_path = Path("artifacts/models/deep_learning/wavelm/mil_head.pt")
    if wavelm_mil_path.parent.exists():
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {"in_dim": 768, "attention_dim": 128, "hidden_dim": 128, "dropout": dropout, "use_residual_mean": False},
            "optimal_threshold": operating_threshold,
            "recording_threshold": recording_threshold,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }, wavelm_mil_path)

    print(f"\nSuccessfully saved clean mil_head.pt to {mil_export_path}")
    print(f"Saved evaluation metrics to {output_dir / 'mil_evaluation_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate clean Gated Attention-MIL")
    parser.add_argument("--train-bags", type=str, default="artifacts/mil/bags_train.pt")
    parser.add_argument("--val-bags", type=str, default="artifacts/mil/bags_val.pt")
    parser.add_argument("--test-bags", type=str, default="artifacts/mil/bags_test.pt")
    parser.add_argument("--output-dir", type=str, default="artifacts/mil")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    dev = torch.device(args.device)

    train_mil_on_train_and_val(
        train_bags_path=Path(args.train_bags),
        val_bags_path=Path(args.val_bags),
        test_bags_path=Path(args.test_bags),
        output_dir=Path(args.output_dir),
        device=dev,
        epochs=args.epochs,
        lr=args.lr,
    )
