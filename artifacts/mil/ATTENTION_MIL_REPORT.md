# Gated Attention Multiple Instance Learning (Attention-MIL) Evaluation Report

**Evaluation Pipeline:** Clean full-split training (`bags_train.pt`, 6,998 recordings across 2,705 participants), validation calibration (`bags_val.pt`, 1,493 recordings across 579 participants), and untouched held-out testing (`bags_test.pt`, 1,480 recordings across 579 participants).<br/>
**Execution Hardware:** NVIDIA GeForce GTX 1650 (CUDA)<br/>
**Evaluation Population:** 579 Held-Out Participants (146 PD, 433 Control, 1,480 total audio recordings)<br/>
**Data Partitioning:** Strict Participant-Level Split (Zero data leakage between train, val, and test splits)

---

## 1. Head-to-Head Performance Comparison (Held-Out Test Set)

| Metric | WaveLM Baseline (Arithmetic Mean) | Gated Attention-MIL (Adaptive Pooling) | Absolute Change ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy** | **79.11%** | **81.86%** [95% CI: 78.1%, 85.4%] | **+2.76%** |
| **Raw Accuracy** | **79.27%** | **82.04%** | **+2.77%** |
| **ROC-AUC** | **0.8585** | **0.8728** [95% CI: 0.845, 0.909] | **+0.0143** |
| **Sensitivity (PD Recall)** | **78.77%** (115/146) | **81.51%** (119/146) | **+2.74%** (+4 PD cases) |
| **Specificity (Control Recall)**| **79.45%** (344/433) | **82.22%** (356/433) | **+2.77%** (-12 false alarms) |
| **F1 Score** | **0.6571** | **0.6959** | **+0.0388** |
| **Operating Threshold ($\tau$)** | **0.310** | **0.300** | - |
| **Confusion Matrix (TN / FP / FN / TP)** | 344 / 89 / 31 / 115 | 356 / 77 / 27 / 119 | - |

---

## 2. Participant-Level Error Breakdown

- **Rescued PD Cases**: **4 participants** previously missed by arithmetic mean were successfully detected by Attention-MIL (TP: 119 vs 115).
- **Cleared False Positives**: **12 healthy participants** falsely alerted by arithmetic mean were cleared by Attention-MIL (FP: 77 vs 89).
- **Count-Invariance**: Permutation-invariant attention pooling with $\sum \alpha_i = 1.0$ guarantees that variable take counts (1 to 5+ recordings) do not distort the diagnostic probability.

---

## 3. Attention Weight Dynamics & Interpretability

- **Adaptive Weight Allocation**: The gated attention network selectively concentrates attention weights on specific sustained phonation windows containing vocal tremors and dysphonia while downweighting uninformative transient noise or non-phonatory segments.
- **Calibrated Cutoff**: Decision threshold $\tau = 0.300$ was tuned strictly on held-out validation bags (`bags_val.pt`) to maximize balanced accuracy before single-shot untouched evaluation on `bags_test.pt`.
