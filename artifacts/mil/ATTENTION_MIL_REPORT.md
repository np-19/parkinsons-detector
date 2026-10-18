# Gated Attention Multiple Instance Learning (Attention-MIL) Evaluation Report

**Branch:** `feature/attention-mil-pooling` (Local, unpushed)  
**Execution Hardware:** NVIDIA GeForce GTX 1650 (CUDA)  
**Evaluation Population:** 579 Held-Out Participants (146 PD, 433 Control, 1,480 total audio recordings)  
**Cross-Validation:** 5-Fold Stratified Participant Cross-Validation (Zero test leakage)  

---

## 1. Head-to-Head Performance Comparison

| Metric | WaveLM Baseline (Arithmetic Mean) | Gated Attention-MIL (Adaptive Pooling) | Absolute Change ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy** | **79.11%** [75.0%, 83.0%] | **82.54%** [79.0%, 86.1%] | **+3.44%** |
| **ROC-AUC** | **0.8585** [0.820, 0.890] | **0.8791** [0.845, 0.912] | **+0.0206** |
| **Sensitivity (PD Recall)** | **78.77%** (115/146) | **83.56%** (122/146) | **+4.79%** |
| **Specificity (Control Recall)**| **79.45%** (344/433) | **81.52%** (353/433) | **+2.08%** |
| **Operating Threshold** | **0.310** | **0.300** | - |
| **Confusion Matrix (TN / FP / FN / TP)** | 344 / 89 / 31 / 115 | 353 / 80 / 24 / 122 | - |

---

## 2. Participant-Level Error Breakdown

- **Rescued PD Cases**: **8 participants** previously missed by arithmetic mean were successfully detected by Attention-MIL.
- **Degraded PD Cases**: **1 participants**.
- **Cleared False Positives**: **12 healthy participants** falsely alerted by arithmetic mean were cleared by Attention-MIL.
- **New False Positives**: **3 participants**.

---

## 3. Attention Weight Dynamics & Interpretability

- **Average Peak Attention Weight**: **39.3%** per bag (Uniform baseline: ~16.7% for 6 instances).
- **Average Attention Std Dev**: **0.108**.
- **Clinical Implication**: The gated attention network selectively concentrates probability mass on specific dysphonic time windows rather than diluting confidence across clean phonation segments.
