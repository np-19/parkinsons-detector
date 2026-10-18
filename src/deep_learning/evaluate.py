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
from .model import Wav2Vec2ForParkinsons, WaveLMForParkinsons, build_speech_model


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
    parser.add_argument("--model-dir", type=str, default="artifacts/models/deep_learning/wavelm", help="Directory containing best_model.pt")
    parser.add_argument("--test-metadata", type=str, default="data/metadata/test.csv", help="Path to test.csv")
    parser.add_argument("--audio-root", type=str, default="data/processed_audio", help="Audio root directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (0=main thread, avoids IPC stalls)")
    parser.add_argument("--duration", type=float, default=8.0, help="Audio segment duration in seconds")
    parser.add_argument("--num-crops", type=int, default=1, help="Number of overlapping crops per recording for TTA (1=center crop only)")
    parser.add_argument("--crop-stride", type=float, default=1.5, help="Stride in seconds between crops for multi-crop TTA")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Cap test samples (optional for sanity check)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 mixed precision")
    parser.add_argument("--mil-path", type=str, default=None, help="Path to mil_head.pt checkpoint")
    parser.add_argument("--bags-path", type=str, default="artifacts/mil/bags_test.pt", help="Path to pre-extracted participant bags")
    parser.add_argument("--eval-bags", action="store_true", default=False, help="Fast evaluation on pre-extracted participant bags (seconds instead of 1 hour)")
    parser.add_argument("--run-meta-classifier", action="store_true", default=False, help="Run exploratory meta-classifier (Note: confounded by recording counts)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    use_fp16 = args.fp16 and (device.type == "cuda")
    print(f"=== Held-Out Test Evaluation ===")
    print(f"Device: {device} (FP16: {use_fp16})")
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    optimal_threshold = checkpoint.get("optimal_threshold", 0.50)
    best_epoch = checkpoint.get("epoch", -1)

    print(f"Checkpoint was trained for {best_epoch} epochs.")
    print(f"Loaded validation-selected optimal threshold: {optimal_threshold:.3f}")

    # Initialize model
    model_name = config.get("model_name", "facebook/wav2vec2-base")
    print(f"Instantiating architecture for backbone: '{model_name}'...")
    model = build_speech_model(
        model_name=model_name,
        num_classes=2,
        num_frozen_layers=config.get("num_frozen_layers", 4),
        freeze_feature_extractor=True,
        gradient_checkpointing=False,  # No checkpointing needed for eval
    )
    raw_state_dict = checkpoint["model_state_dict"]
    state_dict = {k: v for k, v in raw_state_dict.items() if not k.startswith("loss_fn.")}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Load Gated Attention-MIL head if available
    mil_path = Path(args.mil_path) if args.mil_path else (model_dir / "mil_head.pt")
    if not mil_path.exists():
        for candidate in [
            Path("artifacts/models/deep_learning/wavelm/mil_head.pt"),
            Path("artifacts/mil/mil_head.pt"),
        ]:
            if candidate.exists():
                mil_path = candidate
                break

    mil_model = None
    mil_threshold = 0.300
    if mil_path and mil_path.exists():
        try:
            from src.mil.model import GatedAttentionMIL
            mil_ckpt = torch.load(mil_path, map_location=device, weights_only=False)
            valid_keys = {"in_dim", "attention_dim", "hidden_dim", "dropout", "use_residual_mean"}
            clean_cfg = {k: v for k, v in mil_ckpt.get("config", {}).items() if k in valid_keys}
            mil_model = GatedAttentionMIL(**clean_cfg)
            mil_model.load_state_dict(mil_ckpt["model_state_dict"])
            mil_model.to(device).eval()
            mil_threshold = float(mil_ckpt.get("optimal_threshold", 0.300))
            print(f">>> [ATTENTION-MIL] Gated Attention Head Loaded: {mil_path}")
            print(f"    - Calibrated Operating Threshold: {mil_threshold:.3f}")
        except Exception as e:
            print(f"[WARNING] Failed to load Attention-MIL head ({e}); falling back to arithmetic mean.")

    # Check for pre-extracted test bags fallback if requested or if local audio files are partial
    bags_path = Path(args.bags_path)
    use_bags = bool(args.eval_bags and bags_path.exists())
    if not use_bags and bags_path.exists():
        try:
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
            if len(test_ds) < 100:
                use_bags = True
        except Exception:
            use_bags = True

    if use_bags:
        print(f"\n[INFO] Fast evaluation mode using pre-extracted participant bags: {bags_path}")
        bags = torch.load(bags_path, map_location=device, weights_only=False)
        print(f"Loaded {len(bags)} held-out test participant bags.")

        from src.mil.model import score_production_bag
        from src.mil.train import compute_bootstrap_ci

        p_true = []
        p_prob_base = []
        p_prob_mil = []
        all_health_codes = []

        with torch.no_grad():
            for hc, b in bags.items():
                all_health_codes.append(hc)
                p_true.append(int(b["label"]))
                base_p = float(b["crop_probabilities"].mean().item())
                p_prob_base.append(base_p)
                if mil_model is not None:
                    prod_res = score_production_bag(
                        mil_model=mil_model,
                        bag_embeddings=b["embeddings"].to(device),
                        crop_probabilities=b["crop_probabilities"],
                        num_recordings=b.get("num_recordings", len(b["embeddings"]) // 3),
                        recording_threshold=optimal_threshold,
                        participant_threshold=mil_threshold,
                    )
                    p_prob_mil.append(prod_res["participant_score"])

        p_true = np.array(p_true, dtype=int)
        p_prob_base = np.array(p_prob_base, dtype=float)
        part_metrics_opt = compute_metrics(p_true, p_prob_base, threshold=optimal_threshold)
        part_metrics_default = compute_metrics(p_true, p_prob_base, threshold=0.50)

        part_metrics_mil = None
        if mil_model is not None:
            p_prob_mil = np.array(p_prob_mil, dtype=float)
            part_metrics_mil = compute_metrics(p_true, p_prob_mil, threshold=mil_threshold)
            ci = compute_bootstrap_ci(p_true, p_prob_mil, threshold=mil_threshold)
            part_metrics_mil.update(ci)

        print("\n" + "=" * 65)
        print("           HELD-OUT TEST SPLIT EVALUATION RESULTS")
        print("=" * 65)
        print(f"Recording Threshold (selected on val): {optimal_threshold:.3f}\n")
        print("--- 1. Primary Baseline: Participant Mean (Arithmetic Mean) ---")
        print(f"  Participants Evaluated:   {len(p_true)}")
        print(f"  Balanced Accuracy:        {part_metrics_opt['balanced_accuracy'] * 100:.2f}%")
        print(f"  Accuracy:                 {part_metrics_opt['accuracy'] * 100:.2f}%")
        print(f"  Sensitivity (PD Recall):  {part_metrics_opt['sensitivity'] * 100:.2f}% ({part_metrics_opt['confusion_matrix']['tp']}/{part_metrics_opt['confusion_matrix']['tp'] + part_metrics_opt['confusion_matrix']['fn']})")
        print(f"  Specificity (Ctrl Rec):   {part_metrics_opt['specificity'] * 100:.2f}% ({part_metrics_opt['confusion_matrix']['tn']}/{part_metrics_opt['confusion_matrix']['tn'] + part_metrics_opt['confusion_matrix']['fp']})")
        print(f"  F1 Score:                 {part_metrics_opt['f1']:.4f}")
        print(f"  ROC-AUC:                  {part_metrics_opt['roc_auc']:.4f}")
        print(f"  Confusion Matrix:         TN={part_metrics_opt['confusion_matrix']['tn']}, FP={part_metrics_opt['confusion_matrix']['fp']}, "
              f"FN={part_metrics_opt['confusion_matrix']['fn']}, TP={part_metrics_opt['confusion_matrix']['tp']}")

        if part_metrics_mil is not None:
            delta_bal = (part_metrics_mil['balanced_accuracy'] - part_metrics_opt['balanced_accuracy']) * 100
            print(f"\n--- 2. Production: Gated Attention-MIL Aggregation ---")
            print(f"  Operating Threshold:      {mil_threshold:.3f}")
            print(f"  Recording Threshold:      {optimal_threshold:.3f}")
            print(f"  Balanced Accuracy:        {part_metrics_mil['balanced_accuracy'] * 100:.2f}% (Delta: {delta_bal:+.2f}%)")
            if "balanced_accuracy_ci" in part_metrics_mil:
                b_ci = part_metrics_mil["balanced_accuracy_ci"]
                print(f"  95% Bootstrap CI:         [{b_ci[0]*100:.1f}%, {b_ci[1]*100:.1f}%]")
            print(f"  Accuracy:                 {part_metrics_mil['accuracy'] * 100:.2f}%")
            print(f"  Sensitivity (PD Recall):  {part_metrics_mil['sensitivity'] * 100:.2f}% ({part_metrics_mil['confusion_matrix']['tp']}/{part_metrics_mil['confusion_matrix']['tp'] + part_metrics_mil['confusion_matrix']['fn']})")
            print(f"  Specificity (Ctrl Rec):   {part_metrics_mil['specificity'] * 100:.2f}% ({part_metrics_mil['confusion_matrix']['tn']}/{part_metrics_mil['confusion_matrix']['tn'] + part_metrics_mil['confusion_matrix']['fp']})")
            print(f"  F1 Score:                 {part_metrics_mil['f1']:.4f}")
            print(f"  ROC-AUC:                  {part_metrics_mil['roc_auc']:.4f}")
            if "roc_auc_ci" in part_metrics_mil:
                a_ci = part_metrics_mil["roc_auc_ci"]
                print(f"  ROC-AUC 95% CI:           [{a_ci[0]:.3f}, {a_ci[1]:.3f}]")
            print(f"  Confusion Matrix:         TN={part_metrics_mil['confusion_matrix']['tn']}, FP={part_metrics_mil['confusion_matrix']['fp']}, "
                  f"FN={part_metrics_mil['confusion_matrix']['fn']}, TP={part_metrics_mil['confusion_matrix']['tp']}")

        print("=" * 65)

        # Save reports
        report = {
            "model_dir": str(model_dir),
            "best_epoch": best_epoch,
            "optimal_threshold": optimal_threshold,
            "recording_threshold": optimal_threshold,
            "participant_level": {
                "optimal_threshold_metrics": part_metrics_opt,
                "default_threshold_metrics": part_metrics_default,
                "total_participants": len(p_true),
            },
        }
        if part_metrics_mil is not None:
            report["gated_attention_mil"] = {
                "threshold": mil_threshold,
                "recording_threshold": optimal_threshold,
                "metrics": part_metrics_mil,
            }

        report_path = model_dir / "test_evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        pred_df = pd.DataFrame({
            "healthCode": all_health_codes,
            "label": p_true,
            "prob_PD_baseline": p_prob_base,
            "prob_PD_mil": p_prob_mil if mil_model is not None else p_prob_base,
        })
        predictions_path = model_dir / "test_predictions.csv"
        pred_df.to_csv(predictions_path, index=False)

        print(f"\nEvaluation report written to {report_path}")
        print(f"Detailed predictions written to {predictions_path}\n")
        return

    # Standard Dataset path when all raw audio files are present locally
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
                with torch.cuda.amp.autocast(enabled=use_fp16):
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
                with torch.cuda.amp.autocast(enabled=use_fp16):
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

    # Clinical Top-2 Aggregation
    def calc_top2(probs):
        s = sorted(list(probs))
        return float(np.mean(s[-2:]))

    part_top2_df = pred_df.groupby("healthCode").agg({
        "label": "first",
        "prob_PD": calc_top2,
    }).reset_index()
    p_true_top2 = part_top2_df["label"].values.astype(int)
    p_prob_top2 = part_top2_df["prob_PD"].values.astype(float)
    part_metrics_top2 = compute_metrics(p_true_top2, p_prob_top2, threshold=optimal_threshold)

    # Display results
    print("\n" + "=" * 60)
    print("           HELD-OUT TEST SPLIT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Optimal Threshold (selected on val): {optimal_threshold:.3f}\n")
    print("--- 1. Primary Clinical Metric: Participant Mean (Count-Invariant & Leak-Free) ---")
    print(f"  Participants Evaluated:   {len(p_true)}")
    print(f"  Balanced Accuracy:        {part_metrics_opt['balanced_accuracy'] * 100:.2f}%")
    print(f"  Accuracy:                 {part_metrics_opt['accuracy'] * 100:.2f}%")
    print(f"  Sensitivity (PD Recall):  {part_metrics_opt['sensitivity'] * 100:.2f}%")
    print(f"  Specificity (Ctrl Rec):   {part_metrics_opt['specificity'] * 100:.2f}%")
    print(f"  F1 Score:                 {part_metrics_opt['f1']:.4f}")
    print(f"  ROC-AUC:                  {part_metrics_opt['roc_auc']:.4f}")
    print(f"  Confusion Matrix:         TN={part_metrics_opt['confusion_matrix']['tn']}, FP={part_metrics_opt['confusion_matrix']['fp']}, "
          f"FN={part_metrics_opt['confusion_matrix']['fn']}, TP={part_metrics_opt['confusion_matrix']['tp']}")

    print("\n--- 2. Exploratory: Top-2 Mean Aggregation ---")
    print("  [DISCLAIMER: Confounded by recording count imbalance (5.16 PD vs 1.72 Control); exploratory only]")
    print(f"  Balanced Accuracy:        {part_metrics_top2['balanced_accuracy'] * 100:.2f}%")
    tp_t2 = part_metrics_top2['confusion_matrix']['tp']
    fn_t2 = part_metrics_top2['confusion_matrix']['fn']
    tn_t2 = part_metrics_top2['confusion_matrix']['tn']
    fp_t2 = part_metrics_top2['confusion_matrix']['fp']
    print(f"  Sensitivity (PD Recall):  {part_metrics_top2['sensitivity'] * 100:.2f}% (Catches {tp_t2} / {tp_t2 + fn_t2} true PD patients)")
    print(f"  Specificity (Ctrl Rec):   {part_metrics_top2['specificity'] * 100:.2f}% ({tn_t2} / {tn_t2 + fp_t2} controls correct)")
    print(f"  F1 Score:                 {part_metrics_top2['f1']:.4f}")
    print(f"  ROC-AUC:                  {part_metrics_top2['roc_auc']:.4f}")
    print(f"  Confusion Matrix:         TN={tn_t2}, FP={fp_t2}, FN={fn_t2}, TP={tp_t2}")

    print("\n--- 3. Recording-Level Metrics ---")
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

    # Step 4: Run Clinical Participant Meta-Classifier only if explicitly requested
    if args.run_meta_classifier:
        try:
            from src.deep_learning.meta_classifier import run_meta_classifier
            meta_report = model_dir / "meta_classifier_report.json"
            v1_preds = model_dir.parent / "wav2vec2_finetuned" / "test_predictions.csv"
            print("\n--- 4. Participant-Level Distribution Meta-Classifier (Exploratory Ablation) ---")
            print("  [WARNING: Confounded by recording count imbalance in mPower dataset; exploratory only]")
            run_meta_classifier(
                v2_preds_path=predictions_path,
                v1_preds_path=v1_preds if v1_preds.exists() else None,
                output_report=meta_report,
            )
        except Exception as e:
            print(f"Meta-classifier evaluation skipped: {e}")


if __name__ == "__main__":
    main()


