# Minor Project Viva & Defense Guide: Voice-Based Parkinson's Detection

This guide provides clear, direct, and technically sound answers for college project evaluators and viva examiners. All metrics and figures are grounded in the actual codebase outputs and test results.

---

## 1. Project Overview & Quick Reference Sheet

| Parameter | Specification |
| :--- | :--- |
| **Project Title** | Voice-Based Parkinson's Disease Detection Using Deep Multiple Instance Learning |
| **Objective** | Classify voice recordings (`/aaa/`) as Parkinson's Disease vs. Healthy Control. |
| **Dataset** | mPower Mobile Parkinson Study (3,868 participants, 9,986 recordings). |
| **Data Partitioning** | **Strict Participant-Level Splitting**: Train (2,707), Validation (580), Held-Out Test (579). Zero participant overlap. |
| **Backbone Model** | Fine-Tuned **Microsoft WaveLM Base+** (8 unfrozen Transformer layers, 90M parameters). |
| **Aggregation Head** | **Gated Attention Multiple Instance Learning (Attention-MIL)** (`mil_head.pt`). |
| **Hardware** | Trained & evaluated locally on an **NVIDIA GeForce GTX 1650 (4 GB VRAM)**. |
| **Single-Model Accuracy** | **Balanced Accuracy: `81.86%`** (95% CI: `[78.1%, 85.4%]`), **Sensitivity: `81.51%`** (119/146), **Specificity: `82.22%`** (356/433), **ROC-AUC: `0.8728`**, Threshold $\tau = 0.300$. |
| **Primary Baseline** | **Balanced Accuracy: `79.11%`**, ROC-AUC: `0.8585` (Arithmetic Mean, $\tau = 0.310$). |
| **Inference Latency** | **< 1 second** per audio recording on GTX 1650. |

---

## 2. Key Figures to Show Evaluators

### Figure 1: Receiver Operating Characteristic (ROC) Curve
![ROC Curves](figures/roc_curves.png)
* **What to say**: *"Our WaveLM + Gated Attention-MIL model achieves an ROC-AUC of 0.8728 on 579 unseen test participants. At our validation-calibrated operating threshold ($\tau = 0.300$), it achieves 81.51% Sensitivity and 82.22% Specificity."*

---

### Figure 2: Confusion Matrices (Baseline vs. Proposed Model)
![Confusion Matrices](figures/confusion_matrices.png)
* **What to say**: *"This figure compares our Baseline model against our Proposed WaveLM + Gated Attention-MIL model on all 579 held-out test participants (433 Healthy, 146 Parkinson's).
  - The vertical axis displays Ground Truth Status (Actual Healthy vs. Actual Parkinson's), and the horizontal axis shows the Model Prediction.
  - Slices are clearly labeled with counts, category names (True Healthy, False Alarm, Missed Case, Detected PD), and row percentages.
  - Notice that Attention-MIL reduced false alarms from 89 down to 77 (clearing 12 healthy individuals falsely flagged by arithmetic averaging).
  - It also correctly detected 119 out of 146 Parkinson's patients (up from 115, catching 4 additional cases missed by standard averaging).
  - This boosted balanced accuracy from **79.11% to 81.86%** (+2.76% gain) with zero test leakage."*

---

### Figure 3: Prediction Score Distribution
![Score Distribution](figures/clinical_triage_distribution.png)
* **What to say**: *"This histogram shows the distribution of prediction probabilities. Healthy controls cluster toward low probabilities, while Parkinson's participants cluster toward high probabilities. The middle region represents borderline takes where asking for a repeat recording provides clear confirmation."*

---

## 3. Top 10 Viva / Evaluator Questions (With Direct Answers)

### Q1: "Why did you choose WaveLM over standard Wav2Vec 2.0?"
> **Answer**:
> "Wav2Vec 2.0 was pretrained purely on clean audiobook speech (LibriSpeech). WaveLM Base+ was pretrained by Microsoft on **94,000 hours of noisy speech** using **Masked Speech Denoising**.
>
> During pretraining, WaveLM was trained to reconstruct clean speech states from noisy and reverberant audio. Because mobile phone voice recordings have ambient room noise and microphone differences, WaveLM is naturally robust to these variations. On our held-out test data, WaveLM improved test balanced accuracy from **74.72% up to 79.11%** (arithmetic mean) compared to Wav2Vec 2.0."

---

### Q2: "Why didn't you use an ensemble of multiple models?"
> **Answer**:
> "We tested combining WaveLM with Wav2Vec 2.0 in an ensemble. While the ensemble gave a tiny improvement in ROC-AUC (+0.0027, from 0.8560 to 0.8587), it **tripled GPU memory usage and tripled inference time** because it has to load multiple 90M-parameter models.
>
> For our project, we decided that a 300% increase in compute cost for a negligible +0.0027 AUC gain is not practical. Instead, keeping single-model WaveLM and adding a lightweight Gated Attention-MIL head gave a solid +2.76% balanced accuracy jump, reaching **81.86%** (ROC-AUC: **0.8728**) while keeping inference under 1 second."

---

### Q3: "What is Gated Attention-MIL and why is it better than simple averaging?"
> **Answer**:
> "In our project, each participant has multiple audio takes, and each take is sliced into 8-second crops. This is a **Multiple Instance Learning (MIL)** problem where one participant is a 'bag' containing multiple audio slices.
>
> Simple arithmetic averaging gives equal weight to all slices.
>
> We implemented Ilse et al.'s **Gated Attention-MIL**:
> $$a_i = \mathbf{w}^T (\tanh(\mathbf{V} \mathbf{h}_i) \odot \sigma(\mathbf{U} \mathbf{h}_i)) \cdot \frac{1}{\sqrt{L}}, \quad \alpha_i = \text{softmax}(a_i)$$
>
> The attention head automatically learns which audio instance representations provide the strongest acoustic evidence for classification. This eliminated 12 false alarms on healthy controls (from 89 down to 77) and detected 4 additional Parkinson's patients (up to 119)."

---

### Q4: "Why did late fusion with handcrafted acoustic features (jitter, shimmer) fail?"
> **Answer**:
> "We tested extracting 88 openSMILE acoustic features (jitter, shimmer, pitch perturbation) and training gradient-boosted trees. While they reached 81% on validation data, their accuracy collapsed to **72% on unseen test data**.
>
> Handcrafted features like jitter and shimmer assume high-end studio microphones. On smartphones, phone casing acoustics, room echoes, and automatic gain control distort local perturbation math. Fusing them with WaveLM injected noise and degraded performance. Deep self-supervised representations from WaveLM are vastly more reliable on mobile audio."

---

### Q5: "How did you prevent data leakage and cheating in your evaluation?"
> **Answer**:
> "We followed strict machine learning practices:
> 1. **Participant-Level Splitting**: Grouped strictly by participant ID (`healthCode`). No participant in the test set ever appeared in training or validation.
> 2. **Count-Invariance**: In mPower, Parkinson's patients submitted more recordings than controls (5.0 vs 1.7 average). If a model uses `count` or order statistics (`max`, `min`), it cheats by predicting who used the app more rather than voice quality. Our Attention-MIL weights strictly sum to 1.0 ($\sum \alpha_i = 1.0$), making the output count-invariant.
> 3. **Clean Training & Thresholding**: The MIL head was trained strictly on 2,705 training participant bags (`bags_train.pt`), the operating threshold ($\tau = 0.300$) was calibrated on validation bags (`bags_val.pt`), and the held-out test set was evaluated once without test peeking."

---

### Q6: "How is audio preprocessed and normalized before model input?"
> **Answer**:
> "Raw voice recordings from mobile phones arrive in various sampling rates and volume levels:
> 1. **Resampling**: All recordings are resampled to 16 kHz Mono float32 waveforms.
> 2. **Duration Slicing**: Sliced into 8.0-second sliding windows (128,000 samples). Recordings under 1.0s or containing pure silence are rejected.
> 3. **Peak Normalization**: Normalized to $[-1.0, 1.0]$ so differences in microphone recording distance or gain do not bias predictions.
> We deliberately avoid aggressive digital high-pass or bandpass filters that could alter natural vocal fold harmonics."

---

### Q7: "How did you train and run this on an NVIDIA GTX 1650 (4 GB VRAM)?"
> **Answer**:
> "To fit the 90M-parameter model into 4 GB VRAM without out-of-memory errors:
> 1. **Frozen CNN Encoder**: Freezing the 7-layer CNN encoder saved ~1.2 GB of VRAM.
> 2. **Mixed Precision (AMP FP16)**: Using 16-bit floating point cut memory by 50%.
> 3. **Gradient Accumulation**: Batch size 4 with 8 accumulation steps achieved an effective batch size of 32 while keeping peak memory at **~2.8–3.1 GB**, comfortably within our 4 GB limit."

---

### Q8: "How does the system achieve high diagnostic accuracy?"
> **Answer**:
> "Through grounded architectural design and rigorous evaluation stages:
> 1. **Gated Attention-MIL Pooling**: Achieves **81.86% Balanced Accuracy [78.1%, 85.4% 95% CI]**, **81.51% Sensitivity** (119/146), **82.22% Specificity** (356/433), and **0.8728 ROC-AUC** across all 579 held-out test participants with zero data leakage.
> 2. **High-Confidence Triage Bands**: For prediction probabilities outside the borderline band ($P < 0.20$ or $P \ge 0.50$, representing 84% of participants), balanced accuracy reaches **83.55% to 85.34%**, providing decisive separation for clinical review.
> 3. **Acoustic Waveform Integrity**: Standardized 16 kHz resampling and peak normalization preserves vocal fold harmonics and micro-tremor dynamics without distorting filtering."

---

### Q9: "What voice symptoms does Parkinson's Disease cause?"
> **Answer**:
> "Parkinson's causes **hypokinetic dysarthria** due to reduced dopamine in the brain affecting vocal cord motor control:
> 1. **Vocal Tremor (4–7 Hz)**: Rhythmic involuntary micro-fluctuations in pitch and loudness.
> 2. **Reduced Loudness (Hypophonia)** and breathiness due to incomplete vocal cord closure.
> 3. **Acoustic Micro-Breaks**: Brief interruptions where vocal fold vibration loses stability.
> Our 8.0-second audio window was chosen because it captures 32–56 complete tremor cycles, allowing the attention heads to detect them reliably."

---

### Q10: "What are the limitations and future scope of your project?"
> **Answer**:
> "1. **Current Scope**: The model currently evaluates sustained vowels (`/aaa/`). Adding running speech (reading a sentence) could provide additional speech cadence biomarkers.
> 2. **Future Scope**: We plan to export the Attention-MIL head to ONNX so the entire model can run locally on an Android/iOS smartphone app in under 500 milliseconds without requiring cloud servers."

---

## 4. Defense Presentation Checklist

- [x] **Problem Statement**: Voice changes appear early in Parkinson's disease; non-invasive smartphone voice testing can assist early screening.
- [x] **Dataset**: mPower dataset (3,868 participants, 9,986 recordings) with 100% strict participant-level splitting.
- [x] **Single-Model Architecture**: Fine-tuned WaveLM Base+ (90M params) with Gated Attention-MIL pooling.
- [x] **Show Figure 1 (ROC)**: 0.8728 ROC-AUC on 579 unseen test participants.
- [x] **Show Figure 2 (Confusion Matrix)**: 81.51% Sensitivity, 82.22% Specificity ($\tau = 0.300$), clearing 12 false alarms (77 vs 89) and detecting 4 additional cases (119 vs 115).
- [x] **Engineering Highlights**: Standardized 16 kHz audio pipeline, GTX 1650 4GB VRAM optimization (AMP FP16), and <1 second inference latency.
