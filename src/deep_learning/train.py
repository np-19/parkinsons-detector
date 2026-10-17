"""Training script for fine-tuning Wav2Vec2 for Parkinson's Disease detection.

Supports:
- Local GTX 1650 (4 GB VRAM) with mixed precision, gradient checkpointing, and gradient accumulation.
- High-throughput cloud GPUs (GridShare / RunPod) with larger physical batch sizes.
- Participant-level validation aggregation and optimal threshold selection.
"""

import argparse
import json
import os
import random
import sys
import time
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .dataset import MPowerAudioDataset, create_dataloader
from .model import Wav2Vec2ForParkinsons, WaveLMForParkinsons, build_speech_model


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def aggregate_by_participant(
    health_codes: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate recording predictions by healthCode via probability mean."""
    df = pd.DataFrame({
        "health_code": health_codes,
        "y_true": y_true,
        "y_prob": y_prob,
    })
    grouped = df.groupby("health_code").agg({
        "y_true": "first",
        "y_prob": "mean",
    }).reset_index()

    return grouped["y_true"].values.astype(int), grouped["y_prob"].values.astype(float)


def find_best_participant_threshold(
    p_true: np.ndarray,
    p_prob: np.ndarray,
    min_thresh: float = 0.30,
    max_thresh: float = 0.70,
    step: float = 0.01,
) -> Tuple[float, float, Dict[str, float]]:
    """Grid search optimal threshold on validation participants to maximize balanced accuracy."""
    best_thresh = 0.50
    best_bal_acc = -1.0
    best_metrics = {}

    thresholds = np.arange(min_thresh, max_thresh + step / 2, step)
    for t in thresholds:
        m = compute_metrics(p_true, p_prob, threshold=float(t))
        if m["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = m["balanced_accuracy"]
            best_thresh = float(t)
            best_metrics = m

    return best_thresh, best_bal_acc, best_metrics


def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_fp16: bool = False,
) -> Dict:
    """Run model evaluation over a dataset and compute recording and participant metrics."""
    model.eval()
    all_probs = []
    all_labels = []
    all_health_codes = []
    all_row_ids = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(input_values=input_values, labels=labels)
                loss = outputs["loss"]
                probs = outputs["probabilities"][:, 1]  # PD positive class probability

            total_loss += loss.item()
            num_batches += 1

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_health_codes.extend(batch["health_codes"])
            all_row_ids.extend(batch["row_ids"])

    avg_loss = total_loss / max(num_batches, 1)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # Participant aggregation
    p_true, p_prob = aggregate_by_participant(all_health_codes, y_true, y_prob)

    # Optimize threshold on participants
    opt_thresh, opt_bal_acc, p_metrics = find_best_participant_threshold(p_true, p_prob)
    r_metrics = compute_metrics(y_true, y_prob, threshold=opt_thresh)

    return {
        "loss": avg_loss,
        "optimal_threshold": opt_thresh,
        "participant_metrics": p_metrics,
        "recording_metrics": r_metrics,
        "recording_probs": y_prob.tolist(),
        "recording_labels": y_true.tolist(),
        "health_codes": all_health_codes,
        "row_ids": all_row_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for Parkinson's Detection")
    parser.add_argument("--data-root", type=str, default="data/processed_audio", help="Root directory with processed audio")
    parser.add_argument("--metadata-root", type=str, default="data/metadata", help="Metadata directory containing train.csv/val.csv")
    parser.add_argument("--output-root", type=str, default="artifacts/models/deep_learning/wav2vec2_finetuned", help="Output directory")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base", help="Hugging Face pretrained model name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step (4 for GTX 1650, 16+ for cloud)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps (8 for GTX 1650, 1-2 for cloud)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for Wav2Vec2 backbone")
    parser.add_argument("--head-learning-rate", type=float, default=1e-4, help="Learning rate for classification head")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for cosine scheduler")
    parser.add_argument("--duration", type=float, default=5.0, help="Audio segment duration in seconds")
    parser.add_argument("--num-frozen-layers", type=int, default=6, help="Number of lower Transformer layers to freeze (0-11)")
    parser.add_argument("--fp16", action="store_true", help="Enable PyTorch mixed precision (AMP fp16)")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing (recommended for GTX 1650)")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate in classification head")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing factor")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup interpolation alpha (0=disabled)")
    parser.add_argument("--mask-time-prob", type=float, default=0.05, help="SpecAugment time mask probability")
    parser.add_argument("--mask-feature-prob", type=float, default=0.05, help="SpecAugment feature mask probability")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap train samples (e.g. 100 for benchmark)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Cap val samples (e.g. 30 for benchmark)")
    parser.add_argument("--pos-weight", type=float, default=None, help="Loss weight for positive class (PD), e.g. 2.5 to counter class imbalance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Parkinson's Deep Learning Fine-Tuning Pipeline ===")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Batch Size: {args.batch_size} (Acc Steps: {args.gradient_accumulation_steps}, Effective: {args.batch_size * args.gradient_accumulation_steps})")
    print(f"Mixed Precision (fp16): {args.fp16}")
    print(f"Gradient Checkpointing: {args.gradient_checkpointing}")
    print(f"Dropout: {args.dropout}, Label Smoothing: {args.label_smoothing}, Mixup Alpha: {args.mixup_alpha}")
    print(f"Output Directory: {output_dir}")

    # Paths
    meta_root = Path(args.metadata_root)
    train_meta = meta_root / "train.csv"
    val_meta = meta_root / "validation.csv"
    failures_file = meta_root / "processed_audio_failures.csv"

    # Datasets
    print("\nLoading datasets...")
    train_ds = MPowerAudioDataset(
        metadata_file=train_meta,
        audio_root=args.data_root,
        target_duration=args.duration,
        is_training=True,
        augment=True,
        failures_file=failures_file,
        max_samples=args.max_train_samples,
    )
    val_ds = MPowerAudioDataset(
        metadata_file=val_meta,
        audio_root=args.data_root,
        target_duration=args.duration,
        is_training=False,
        augment=False,
        failures_file=failures_file,
        max_samples=args.max_val_samples,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    train_loader = create_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Class weighting for loss function
    class_weights = None
    if args.pos_weight is not None:
        class_weights = torch.tensor([1.0, float(args.pos_weight)], device=device)
        print(f"Applying class weights: [1.0, {args.pos_weight}] to counter class imbalance.")

    # Model
    print(f"\nInitializing model '{args.model_name}'...")
    model = build_speech_model(
        model_name=args.model_name,
        num_classes=2,
        num_frozen_layers=args.num_frozen_layers,
        freeze_feature_extractor=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dropout=args.dropout,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        mask_time_prob=args.mask_time_prob,
        mask_feature_prob=args.mask_feature_prob,
    ).to(device)

    # Optimizer & Scheduler
    param_groups = model.get_parameter_groups(
        backbone_lr=args.learning_rate,
        head_lr=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    total_training_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device.type == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    # Training state
    best_val_bal_acc = -1.0
    best_threshold = 0.50
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    start_epoch = 1

    if args.resume_from and Path(args.resume_from).exists():
        print(f"\n[RESUME] Loading checkpoint from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[RESUME WARNING] Could not load optimizer state dict: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_bal_acc = ckpt.get("best_val_balanced_accuracy", -1.0)
        best_threshold = ckpt.get("optimal_threshold", 0.50)
        best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", -1))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        history = ckpt.get("history", [])
        # Advance scheduler steps for previous epochs
        steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
        prior_steps = (start_epoch - 1) * steps_per_epoch
        for _ in range(prior_steps):
            scheduler.step()
        print(f"[RESUME] Resuming at Epoch {start_epoch} (Prior Best Bal Acc: {best_val_bal_acc * 100:.2f}% from Epoch {best_epoch})")

    print(f"\nStarting training loop (Epoch {start_epoch} to {args.epochs})...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            # Mixup augmentation
            if args.mixup_alpha > 0:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                batch_size = input_values.size(0)
                index = torch.randperm(batch_size, device=device)
                mixed_input = lam * input_values + (1 - lam) * input_values[index]
                with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                    outputs = model(input_values=mixed_input, labels=labels)
                    loss_a = outputs["loss"]
                    outputs_b = model(input_values=mixed_input, labels=labels[index])
                    loss_b = outputs_b["loss"]
                    loss = (lam * loss_a + (1 - lam) * loss_b) / args.gradient_accumulation_steps
            else:
                with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                    outputs = model(input_values=input_values, labels=labels)
                    loss = outputs["loss"] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

                pbar.set_postfix({
                    "loss": f"{running_loss / (step + 1):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        train_loss = running_loss / max(len(train_loader), 1)

        # Validation evaluation
        val_results = evaluate_dataset(
            model=model,
            dataloader=val_loader,
            device=device,
            use_fp16=(args.fp16 and device.type == "cuda"),
        )

        val_loss = val_results["loss"]
        val_bal_acc = val_results["participant_metrics"]["balanced_accuracy"]
        val_acc = val_results["participant_metrics"]["accuracy"]
        val_auc = val_results["participant_metrics"]["roc_auc"]
        val_sens = val_results["participant_metrics"]["sensitivity"]
        val_spec = val_results["participant_metrics"]["specificity"]
        opt_thresh = val_results["optimal_threshold"]
        epoch_duration = time.time() - epoch_start

        print(
            f"\nEpoch {epoch:02d} Summary [{epoch_duration:.1f}s]: "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Part Bal Acc: {val_bal_acc * 100:.2f}% (Acc: {val_acc * 100:.2f}%, AUC: {val_auc:.4f}, Sens: {val_sens * 100:.1f}%, Spec: {val_spec * 100:.1f}%) | "
            f"Optimal Threshold: {opt_thresh:.3f}"
        )

        if args.max_train_samples:
            sec_per_rec = epoch_duration / len(train_ds)
            est_full_epoch_min = (sec_per_rec * 7006) / 60
            est_10_epochs_hr = (est_full_epoch_min * 10) / 60
            print(
                f"\n[BENCHMARK TIME PROJECTION FOR FULL DATASET]"
                f"\n  Processing speed: {sec_per_rec:.2f} seconds per recording"
                f"\n  Projected Full Epoch (7,006 recordings): ~{est_full_epoch_min:.1f} minutes"
                f"\n  Projected Full 10 Epochs: ~{est_10_epochs_hr:.2f} hours\n"
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_participant_balanced_accuracy": val_bal_acc,
            "val_participant_accuracy": val_acc,
            "val_participant_roc_auc": val_auc,
            "val_participant_sensitivity": val_sens,
            "val_participant_specificity": val_spec,
            "optimal_threshold": opt_thresh,
            "duration_seconds": epoch_duration,
        }
        history.append(epoch_record)

        # Always save latest checkpoint for seamless resuming
        latest_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_balanced_accuracy": best_val_bal_acc,
            "optimal_threshold": best_threshold,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "config": vars(args),
            "history": history,
        }
        torch.save(latest_checkpoint, output_dir / "checkpoint_latest.pt")

        # Save running training history after every epoch
        with open(output_dir / "training_history.json", "w") as f:
            json.dump({
                "config": vars(args),
                "best_epoch": best_epoch,
                "best_val_balanced_accuracy": best_val_bal_acc,
                "optimal_threshold": best_threshold,
                "epochs": history,
            }, f, indent=2)

        # Checkpoint if best validation participant balanced accuracy
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_threshold = opt_thresh
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_balanced_accuracy": best_val_bal_acc,
                "optimal_threshold": best_threshold,
                "config": vars(args),
                "val_participant_metrics": val_results["participant_metrics"],
                "val_recording_metrics": val_results["recording_metrics"],
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  >>> New best model saved to {output_dir / 'best_model.pt'} (Bal Acc: {val_bal_acc * 100:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"  --- No improvement for {epochs_no_improve}/{args.patience} epochs.")
            if epochs_no_improve >= args.patience:
                print(f"\n[Early Stopping] Triggered after {epoch} epochs. Best epoch was {best_epoch}.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.1f} minutes.")
    print(f"Best Epoch: {best_epoch} with Participant Balanced Accuracy: {best_val_bal_acc * 100:.2f}% (Threshold: {best_threshold:.3f})")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump({
            "config": vars(args),
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_bal_acc,
            "optimal_threshold": best_threshold,
            "total_training_time_seconds": total_time,
            "epochs": history,
        }, f, indent=2)

    print(f"Training history written to {output_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()

