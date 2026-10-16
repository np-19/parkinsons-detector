#!/usr/bin/env python3
"""Empirical Verification of Sensitivity Across Operating Thresholds.

Evaluates out-of-fold cross-validated predictions over all 579 mPower
test participants to confirm exact sensitivity gains at different cutoffs.
"""
import csv
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
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def get_auc(y_true, y_prob):
    pos = [p for yt, p in zip(y_true, y_prob) if yt == 1]
    neg = [p for yt, p in zip(y_true, y_prob) if yt == 0]
    c = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def main():
    v2_preds_path = "artifacts/models/deep_learning/wav2vec2_v2/test_predictions.csv"
    v1_preds_path = "artifacts/models/deep_learning/wav2vec2_finetuned/test_predictions.csv"

    v1_rec = {}
    if Path(v1_preds_path).exists():
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

    auc = get_auc(oof_true, oof_preds)
    total_pd = sum(oof_true)
    total_ctrl = len(oof_true) - total_pd

    print("\n" + "=" * 88)
    print(f"  CROSS-VALIDATED OPERATING SPECTRUM (Total Cohort: {len(oof_true)} | PD: {total_pd} | Control: {total_ctrl})")
    print(f"  Overall ROC-AUC: {auc:.4f}")
    print("=" * 88)
    print(f"  {'Cutoff (t)':<12} | {'Clinical Role':<24} | {'Sensitivity':<13} | {'Specificity':<13} | {'PD Caught':<10}")
    print("  " + "-" * 84)

    test_thresholds = [
        (0.12, "Ultra-Sensitive Gate"),
        (0.15, "Low Screening Cutoff"),
        (0.18, "Early Detection Gate"),
        (0.20, "3-Tier Screening Cutoff"),
        (0.22, "Conservative Screening"),
        (0.26, "Youden Peak Cutoff"),
        (0.29, "Balanced Diagnostic Cutoff"),
        (0.35, "3-Tier High-Risk Cutoff"),
        (0.50, "Standard Naive Cutoff"),
    ]

    for t, role in test_thresholds:
        st = eval_preds(oof_true, oof_preds, t)
        sens = st["sensitivity"] * 100
        spec = st["specificity"] * 100
        tp = st["tp"]
        print(f"  {t:<12.2f} | {role:<24} | {sens:>6.2f}%       | {spec:>6.2f}%       | {tp:>3}/{total_pd} PD")

    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
