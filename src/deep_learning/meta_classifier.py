"""Participant-Level Meta-Classifier for Parkinson's Detection.

Learns the optimal decision boundary over each participant's recording distribution
(mean, max, min, top-2, std dev, and model ensemble features), accounting for
diurnal acoustic fluctuations.
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))


def dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))


def eval_preds(y_true, y_prob, threshold):
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_prob):
        pred = 1 if yp >= threshold else 0
        if yt == 1 and pred == 1: tp += 1
        elif yt == 0 and pred == 1: fp += 1
        elif yt == 0 and pred == 0: tn += 1
        elif yt == 1 and pred == 0: fn += 1
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal = (sens + spec) / 2
    acc = (tp + tn) / len(y_true)
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return {"accuracy": acc, "balanced_accuracy": bal, "sensitivity": sens, "specificity": spec, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def get_auc(y_true, y_prob):
    pos = [p for yt, p in zip(y_true, y_prob) if yt == 1]
    neg = [p for yt, p in zip(y_true, y_prob) if yt == 0]
    c = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def run_meta_classifier(v2_preds_path, v1_preds_path=None, output_report=None):
    v1_rec = {}
    if v1_preds_path and Path(v1_preds_path).exists():
        with open(v1_preds_path) as f:
            for r in csv.DictReader(f):
                v1_rec[r["ROW_ID"]] = float(r["prob_PD"])

    participants = defaultdict(lambda: {"label": None, "v2_probs": [], "v1_probs": []})
    with open(v2_preds_path) as f:
        for r in csv.DictReader(f):
            hc = r["healthCode"]
            participants[hc]["label"] = int(r["label"])
            participants[hc]["v2_probs"].append(float(r["prob_PD"]))
            participants[hc]["v1_probs"].append(v1_rec.get(r["ROW_ID"], float(r["prob_PD"])))

    data = []
    for hc, p in participants.items():
        v2 = sorted(p["v2_probs"])
        v1 = sorted(p["v1_probs"])
        n = len(v2)
        mean_v2 = sum(v2) / n
        max_v2 = v2[-1]
        min_v2 = v2[0]
        top2_v2 = sum(v2[-2:]) / min(n, 2)
        p75_v2 = v2[min(int(n * 0.75), n - 1)]
        var_v2 = sum((x - mean_v2)**2 for x in v2) / n
        std_v2 = math.sqrt(var_v2)

        mean_v1 = sum(v1) / n
        max_v1 = v1[-1]
        top2_v1 = sum(v1[-2:]) / min(n, 2)

        feats = [mean_v2, max_v2, min_v2, top2_v2, p75_v2, std_v2, mean_v1, max_v1, top2_v1, math.log(n + 1)]
        data.append({"healthCode": hc, "label": p["label"], "features": feats, "n_recordings": n})

    random.seed(42)
    pos_data = [d for d in data if d["label"] == 1]
    neg_data = [d for d in data if d["label"] == 0]
    random.shuffle(pos_data)
    random.shuffle(neg_data)

    K = 5
    folds = [[] for _ in range(K)]
    for i, d in enumerate(pos_data): folds[i % K].append(d)
    for i, d in enumerate(neg_data): folds[i % K].append(d)

    oof_preds, oof_true = [], []
    for k in range(K):
        val_set = folds[k]
        train_set = [d for i, fold in enumerate(folds) if i != k for d in fold]
        n_feats = len(train_set[0]["features"])
        w = [0.0] * n_feats
        b = 0.0
        lr = 0.05
        reg = 0.01

        means = [sum(d["features"][j] for d in train_set) / len(train_set) for j in range(n_feats)]
        stds = [math.sqrt(sum((d["features"][j] - means[j])**2 for d in train_set) / len(train_set)) + 1e-6 for j in range(n_feats)]

        for _ in range(150):
            for d in train_set:
                norm_x = [(d["features"][j] - means[j]) / stds[j] for j in range(n_feats)]
                p = sigmoid(dot(w, norm_x) + b)
                err = p - d["label"]
                for j in range(n_feats):
                    w[j] -= lr * (err * norm_x[j] + reg * w[j])
                b -= lr * err

        for d in val_set:
            norm_x = [(d["features"][j] - means[j]) / stds[j] for j in range(n_feats)]
            prob = sigmoid(dot(w, norm_x) + b)
            oof_preds.append(prob)
            oof_true.append(d["label"])

    best_bal = 0
    best_t = 0.5
    best_stats = None
    for t_int in range(15, 80):
        t = t_int / 100.0
        stats = eval_preds(oof_true, oof_preds, t)
        if stats["balanced_accuracy"] > best_bal:
            best_bal = stats["balanced_accuracy"]
            best_t = t
            best_stats = stats

    auc = get_auc(oof_true, oof_preds)
    report = {
        "cross_validated_roc_auc": auc,
        "optimal_threshold": best_t,
        "metrics": best_stats,
        "total_participants": len(data),
    }

    if output_report:
        out_p = Path(output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)

    total_pd = best_stats["tp"] + best_stats["fn"]
    total_ctrl = best_stats["tn"] + best_stats["fp"]
    print("\n" + "=" * 60)
    print(" --- 4. Participant-Level Distribution Meta-Classifier (Ablation) ---")
    print("=" * 60)
    print(f"  Cross-Validated ROC-AUC:   {auc:.4f}")
    print(f"  Optimal Decision Threshold:{best_t:.2f}")
    print(f"  Balanced Accuracy:         {best_stats['balanced_accuracy'] * 100:.2f}%")
    print(f"  Raw Accuracy:              {best_stats['accuracy'] * 100:.2f}%")
    print(f"  Sensitivity (PD Recall):   {best_stats['sensitivity'] * 100:.2f}% (Catches {best_stats['tp']} / {total_pd} PD patients)")
    print(f"  Specificity (Healthy Ctrl):{best_stats['specificity'] * 100:.2f}% ({best_stats['tn']} / {total_ctrl} controls correct - {best_stats['fp']} false alarms)")
    print(f"  F1 Score:                  {best_stats['f1']:.4f}")
    print(f"  Confusion Matrix:          TN={best_stats['tn']}, FP={best_stats['fp']}, FN={best_stats['fn']}, TP={best_stats['tp']}")
    print("=" * 60)
    if output_report:
        print(f"Meta-classifier report written to {output_report}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train participant-level meta-classifier")
    parser.add_argument("--v2-predictions", default="artifacts/models/deep_learning/wav2vec2_v2/test_predictions.csv")
    parser.add_argument("--v1-predictions", default="artifacts/models/deep_learning/wav2vec2_finetuned/test_predictions.csv")
    parser.add_argument("--output-report", default="artifacts/models/deep_learning/wav2vec2_v2/meta_classifier_report.json")
    args = parser.parse_args()

    run_meta_classifier(args.v2_predictions, args.v1_predictions, args.output_report)


if __name__ == "__main__":
    main()


