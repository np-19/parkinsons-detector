"""Compare all fine-tuned models side-by-side on the held-out test split.

Reads test_evaluation_report.json across all model runs and prints
an honest, unconfounded comparison table using the Participant Mean metric.
"""

import json
from pathlib import Path


def main():
    models_dir = Path("artifacts/models/deep_learning")
    model_keys = ["wav2vec2_v2", "wavelm_v1", "wavelm_v2", "wavelm_v3", "wavelm_v4"]

    print("\n" + "=" * 90)
    print("      HELD-OUT TEST SET COMPARISON: ALL MODELS (579 PARTICIPANTS)")
    print("      Primary Metric: Participant Mean (Count-Invariant & Leak-Free)")
    print("=" * 90)
    print(f"{'Model Name':<15} | {'Bal Acc':<9} | {'Accuracy':<9} | {'Sens (Recall)':<14} | {'Specificity':<12} | {'ROC-AUC':<8} | {'Threshold':<9}")
    print("-" * 90)

    for m in model_keys:
        report_path = models_dir / m / "test_evaluation_report.json"
        if not report_path.exists():
            print(f"{m:<15} | [Not yet evaluated or report missing at {report_path}]")
            continue

        try:
            with open(report_path) as f:
                rep = json.load(f)
            thresh = rep.get("optimal_threshold", 0.50)
            part_metrics = rep.get("participant_level", {}).get("optimal_threshold_metrics", {})

            bacc = part_metrics.get("balanced_accuracy", 0.0) * 100
            acc = part_metrics.get("accuracy", 0.0) * 100
            sens = part_metrics.get("sensitivity", 0.0) * 100
            spec = part_metrics.get("specificity", 0.0) * 100
            auc = part_metrics.get("roc_auc", 0.0)

            print(f"{m:<15} | {bacc:>7.2f}% | {acc:>7.2f}% | {sens:>12.2f}% | {spec:>10.2f}% | {auc:>8.4f} | {thresh:>9.3f}")
        except Exception as e:
            print(f"{m:<15} | Error reading report: {e}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
