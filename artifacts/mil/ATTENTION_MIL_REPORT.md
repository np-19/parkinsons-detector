# Gated Attention Multiple Instance Learning (Attention-MIL) Evaluation Report

**Execution Hardware:** NVIDIA GeForce GTX 1650 (CUDA)<br>
**Evaluation Population:** 579 Held-Out Participants (146 PD, 433 Control, 1,480 total audio recordings)<br>
**Protocol:** Single train → validation → one-shot held-out test (zero test leakage). The separate 5-fold OOF ablation is not the source of these test metrics.<br>

---

## 1. Head-to-Head Performance Comparison

| Metric | WaveLM Baseline (Arithmetic Mean) | Gated Attention-MIL (Adaptive Pooling) | Absolute Change ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy** | **79.11%** (CI not reported) | **82.54%** [79.0%, 86.1%] | **+3.44%** |
| **ROC-AUC** | **0.8585** (CI not reported) | **0.8791** [0.845, 0.912] | **+0.0206** |
| **Sensitivity (PD Recall)** | **78.77%** (115/146) | **83.56%** (122/146) | **+4.79%** |
| **Specificity (Control Recall)**| **79.45%** (344/433) | **81.52%** (353/433) | **+2.08%** |
| **Operating Threshold** | **0.310** | **0.300** | - |
| **Confusion Matrix (TN / FP / FN / TP)** | 344 / 89 / 31 / 115 | 353 / 80 / 24 / 122 | - |

---

## 2. Participant-Level Error Breakdown

- **Rescued PD Cases**: **8 participants** previously missed by arithmetic mean were successfully detected by Attention-MIL.
- **Degraded PD Cases**: **1 participant**.
- **Net PD Change**: **+7 participants** (TP 115 → 122).
- **Cleared False Positives**: **12 healthy participants** falsely alerted by arithmetic mean were cleared by Attention-MIL.
- **New False Positives**: **3 participants**.
- **Net False-Positive Change**: **-9 participants** (FP 89 → 80).

---

## 3. Attention Weight Dynamics & Interpretability

- **Average Peak Attention Weight**: **39.3%** per bag (mean bag size: 7.7 instances; uniform baseline: ~18.4%).
- **Average Attention Std Dev**: **0.108**.
- **Clinical Implication**: The gated attention network selectively concentrates probability mass on specific dysphonic time windows rather than diluting confidence across clean phonation segments.
