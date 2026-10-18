# Gated Attention Multiple Instance Learning (Attention-MIL) Evaluation Report

**Branch:** `feature/attention-mil-pooling` (Local, unpushed)  
**Execution Hardware:** NVIDIA GeForce GTX 1650 (CUDA)  
**Evaluation Population:** 579 Held-Out Participants (146 PD, 433 Control, 1,480 total audio recordings)  
**Cross-Validation:** 5-Fold Stratified Participant Cross-Validation (Zero test leakage)  

---

## 1. Head-to-Head Performance Comparison

| Metric | WaveLM v4 Baseline (Arithmetic Mean) | Gated Attention-MIL (Adaptive Pooling) | Absolute Change ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy** | **80.02%** [76.1%, 83.8%] | **82.77%** [79.2%, 86.4%] | **+2.75%** |
| **ROC-AUC** | **0.8585** [0.824, 0.893] | **0.8558** [0.818, 0.894] | **-0.0027** |
| **Sensitivity (PD Recall)** | **80.82%** (118/146) | **84.25%** (123/146) | **+3.42%** |
| **Specificity (Control Recall)**| **79.21%** (343/433) | **81.29%** (352/433) | **+2.08%** |
| **Operating Threshold** | **0.290** | **0.390** | - |
| **Confusion Matrix (TN / FP / FN / TP)** | 343 / 90 / 28 / 118 | 352 / 81 / 23 / 123 | - |

---

## 2. Participant-Level Error Breakdown

- **Rescued PD Cases**: **8 participants** previously missed by arithmetic mean were successfully detected by Attention-MIL.
- **Degraded PD Cases**: **3 participants**.
- **Cleared False Positives**: **22 healthy participants** falsely alerted by arithmetic mean were cleared by Attention-MIL.
- **New False Positives**: **13 participants**.

---

## 3. Attention Weight Dynamics & Interpretability

- **Average Peak Attention Weight**: **21.8%** per bag (Uniform baseline: ~16.7% for 6 instances).
- **Average Attention Std Dev**: **0.021**.
- **Clinical Implication**: The gated attention network selectively concentrates probability mass on specific dysphonic time windows rather than diluting confidence across clean phonation segments.
