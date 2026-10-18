"""Compare all fine-tuned models side-by-side on the held-out test split.

Reads test_evaluation_report.json across all model runs and prints
an honest, unconfounded comparison table using the Participant Mean metric.
"""

import json
from pathlib import Path


def main():
    models_dir = Path("artifacts/models/deep_learning")
    benchmark_targets = [
        ("wav2vec2", "Wav2Vec2 Base", False),
        ("wavelm", "WaveLM Base+", False),
        ("wavelm", "WaveLM + MIL", True),
    ]

    print("\n" + "=" * 90)
    print("      HELD-OUT TEST SET BENCHMARKS: ACOUSTIC FOUNDATION MODELS (579 PARTICIPANTS)")
    print("      Strict Participant Isolation (Zero Leakage) | Count-Invariant Evaluation")
    print("=" * 90)
    print(f"{'Model Architecture':<20} | {'Bal Acc':<9} | {'Accuracy':<9} | {'Sens (Recall)':<14} | {'Specificity':<12} | {'ROC-AUC':<8} | {'Threshold':<9}")
    print("-" * 90)

    for dir_name, display_name, use_mil in benchmark_targets:
        report_path = models_dir / dir_name / "test_evaluation_report.json"
        if not report_path.exists():
            print(f"{display_name:<20} | [Report missing at {report_path}]")
            continue

        try:
            with open(report_path) as f:
                rep = json.load(f)

            if use_mil:
                mil_data = rep.get("gated_attention_mil", {})
                m = mil_data.get("metrics", {})
                thresh = mil_data.get("threshold", 0.300)
            else:
                m = rep.get("participant_level", {}).get("optimal_threshold_metrics", {})
                thresh = rep.get("optimal_threshold", 0.500)

            bacc = m.get("balanced_accuracy", 0.0) * 100
            acc = m.get("accuracy", 0.0) * 100
            sens = m.get("sensitivity", 0.0) * 100
            spec = m.get("specificity", 0.0) * 100
            auc = m.get("roc_auc", 0.0)

            print(f"{display_name:<20} | {bacc:>7.2f}% | {acc:>7.2f}% | {sens:>12.2f}% | {spec:>10.2f}% | {auc:>8.4f} | {thresh:>9.3f}")
        except Exception as e:
            print(f"{display_name:<20} | Error reading report: {e}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

