# Deep Learning Fine-Tuning & Docker Portability Implementation Plan (>85% Target)

This document contains the complete technical blueprint, mathematical justification, and execution plan for fine-tuning `facebook/wav2vec2-base` on the mPower dataset inside a portable Docker container for local GTX 1650 and GridShare cloud GPU environments.

---

## 1. Scientific & Mathematical Foundation for >85% Accuracy

### 1.1 The Generalizability Gap in Prior Baselines
Previous experiments achieved high validation accuracy that failed to transfer to unseen test participants:
- **Phase 3 (Frozen wav2vec2 mean-pooled embeddings)**:
  - Recording Accuracy: ~64.2%
  - Participant Accuracy: ~69.1%
- **Phase 3b (openSMILE eGeMAPS acoustic features + HistGradientBoosting)**:
  - Validation Participant Balanced Accuracy: **81.24%** (ROC-AUC: 0.8661)
  - Held-out Test Participant Balanced Accuracy: **69.49%** (Accuracy: **72.02%**, ROC-AUC: 0.7884)

The ~10% drop on held-out test participants was caused by:
1. **Pre-trained Phoneme Bias**: Wav2Vec2 self-attention heads pre-trained on clean LibriSpeech were optimized to discard vocal tract tremor, micro-pauses, and breathiness as background noise.
2. **Naïve Mean Pooling**: Collapsing time frames ($\sim 500$ frames for a 5-second audio snippet) into a uniform average dilutes localized phonation instabilities.
3. **Acoustic Background Confounding**: Fixed tabular acoustic features overfit to specific room reverberation and smartphone microphone responses present in the training participants.

### 1.2 Five Key Architectural Mechanisms to Exceed 85% Test Accuracy

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Raw Audio (16 kHz mono WAV)                                                            │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. 7-Layer Temporal Convolutional Feature Encoder (FROZEN)                             │
│    - Conv1D blocks converting raw waveform into 20ms acoustic frames                   │
│    - Frozen to preserve core signal processing and save ~1.2 GB VRAM                   │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. SpecAugment & Masking Layer (TRAINING ONLY)                                         │
│    - Time masking: up to 15% of frames masked                                          │
│    - Channel/feature masking: up to 15% of channels masked                             │
│    - Gaussian noise injection: SNR 15–30 dB                                            │
│    - Forces network to learn robust pathology invariant to smartphone mics             │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. 12-Layer Transformer Encoder (LAYERS 7–12 UNFROZEN, LAYERS 1–6 FROZEN)              │
│    - Layers 1–6 maintain generic acoustic primitives                                  │
│    - Layers 7–12 fine-tune to extract dysarthria, vocal instability, and jitter        │
│    - PyTorch Gradient Checkpointing enabled to save ~60% activation memory             │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Learnable Temporal Attention Pooling                                                │
│    - Projection: a_t = w^T * tanh(W * h_t + b)                                         │
│    - Normalized Weights: alpha_t = softmax(a_t)                                        │
│    - Pooled Representation: e = sum_t (alpha_t * h_t) [768-dimensional]                │
│    - Dynamically focuses on frames with dysphonia and pitch instability                │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. Classification Head                                                                 │
│    - LayerNorm(768) ──► Dropout(0.2) ──► Linear(768, 256) ──► GELU ──►                 │
│      Dropout(0.2) ──► Linear(256, 2) ──► CrossEntropyLoss / FocalLoss                  │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 6. Participant-Level Aggregation                                                       │
│    - Averaging probabilities across all recordings per participant:                    │
│      P(PD | healthCode) = (1/N) * sum_i P(PD | recording_i)                            │
│    - Confirmed +4% to +6% boost over recording-level predictions                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### A. Selective Unfreezing of Transformer Layers (Layers 7–12)
- Layers 1–6 remain frozen, preserving low-level phonetic structure and saving compute.
- Layers 7–12 are trained with a low learning rate ($\eta = 2 \times 10^{-5}$), adapting attention heads to vocal frequency modulation (4–7 Hz Parkinsonian tremor).

#### B. Learnable Multi-Head Temporal Attention Pooling
Instead of mean pooling, temporal attention learns which time steps contain diagnostic phonation breaks:
$$a_t = \mathbf{w}^T \tanh(\mathbf{W} h_t + \mathbf{b})$$
$$\alpha_t = \frac{\exp(a_t)}{\sum_{\tau=1}^T \exp(a_\tau)}$$
$$\mathbf{e} = \sum_{t=1}^T \alpha_t h_t$$
Where $h_t \in \mathbb{R}^{768}$, $\mathbf{W} \in \mathbb{R}^{128 \times 768}$, and $\mathbf{w} \in \mathbb{R}^{128}$.

#### C. SpecAugment & SNR Noise Injection
To prevent the model from memorizing participant background noise, during training we inject:
1. Time masking (randomly masking continuous blocks of time frames).
2. Channel masking (masking frequency channels).
3. Additive Gaussian noise at random SNR between 15 dB and 30 dB.
This forces the attention weights to latch onto intrinsic vocal tract mechanics.

#### D. Participant-Level Aggregation
Given $N$ recordings for a participant:
$$P(\text{PD} \mid \text{participant}) = \frac{1}{N} \sum_{i=1}^N P(\text{PD} \mid \text{recording}_i)$$
Decision threshold $\theta^*$ is selected strictly on the validation set to maximize participant balanced accuracy, then applied to the held-out test set.

---

## 2. Hardware Engineering & VRAM Safety Budget

### Local GTX 1650 (4 GB VRAM) vs GridShare Cloud GPU

| Parameter | Local GTX 1650 (4 GB) | GridShare Cloud GPU (A100 / RTX 3090) |
| :--- | :--- | :--- |
| **Precision** | PyTorch AMP `fp16` | PyTorch AMP `fp16` / `bf16` |
| **CNN Feature Extractor** | Frozen | Frozen |
| **Transformer Layers 1–6** | Frozen | Frozen |
| **Transformer Layers 7–12** | Trainable | Trainable |
| **Gradient Checkpointing** | **Enabled** (saves ~2.2 GB) | Optional |
| **Physical Batch Size** | 4 | 16 or 32 |
| **Gradient Accumulation** | 8 | 1 or 2 |
| **Effective Batch Size** | **32** | **32** |
| **Peak VRAM Usage** | **~2.8 – 3.1 GB** (Safe in 4GB) | **~6.5 – 8.0 GB** |
| **Estimated Epoch Time** | ~18–20 minutes | ~2.5–3.5 minutes |
| **Total Training Time (10 Epochs)** | ~3.0 hours | ~30 minutes |

---

## 3. Docker Containerization Architecture

### 3.1 Base Image & Dependencies
- **Base**: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- **Dependencies**: `transformers>=4.48.0`, `accelerate>=0.28.0`, `soundfile`, `scipy`, `scikit-learn`, `pandas`
- **Isolation**: Bypasses host Python 3.14 wheel constraints completely.

### 3.2 Volume Mounts
The container does not copy dataset files into the image. Instead, it uses volumes:
- `/workspace/data`: Read-only (`:ro`) mount of `data/`
- `/workspace/artifacts`: Read-write mount for checkpoints, metrics, and logs
- `/workspace/.cache/huggingface`: Mount to persist downloaded `facebook/wav2vec2-base` weights

### 3.3 Docker Compose Services
- `local-train`: Configured with batch size 4, accumulation 8, `fp16`, gradient checkpointing.
- `cloud-train`: Configured with batch size 16, accumulation 2, `fp16`.
- `evaluate`: Runs one-time evaluation on `data/metadata/test.csv`.

---

## 4. Codebase Structure to Implement

```
parkinsons-detector/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── scripts/
│   ├── run_local.sh
│   └── run_cloud.sh
└── src/
    └── deep_learning/
        ├── __init__.py
        ├── dataset.py
        ├── model.py
        ├── train.py
        └── evaluate.py
```

### Module Responsibilities:
1. **`src/deep_learning/dataset.py`**:
   - Loads 16 kHz WAV audio files mapped from metadata CSVs (`train.csv`, `validation.csv`, `test.csv`).
   - Fixes duration to 5.0 seconds (80,000 samples) via center-crop or zero-padding.
   - Implements dynamic time/channel masking and noise injection during training mode only.
   - Groups participant metadata for downstream aggregation.

2. **`src/deep_learning/model.py`**:
   - Loads `facebook/wav2vec2-base`.
   - Freezes CNN feature extractor (`feature_extractor.requires_grad_(False)`).
   - Freezes Transformer layers 0 through 5.
   - Enables gradient checkpointing on the Wav2Vec2 encoder.
   - Implements `TemporalAttentionPooling`.
   - Classification head: `LayerNorm(768) -> Dropout(0.2) -> Linear(768, 256) -> GELU -> Dropout(0.2) -> Linear(256, 2)`.

3. **`src/deep_learning/train.py`**:
   - Configures AdamW with differential learning rates ($\eta_{\text{backbone}} = 2 \times 10^{-5}$, $\eta_{\text{head}} = 1 \times 10^{-4}$).
   - Cosine annealing learning rate scheduler with linear warmup (first 10% of steps).
   - PyTorch `GradScaler` for AMP FP16.
   - Evaluates validation loss and validation participant balanced accuracy at the end of each epoch.
   - Saves `best_model.pt` based strictly on validation participant balanced accuracy.

4. **`src/deep_learning/evaluate.py`**:
   - Loads frozen `best_model.pt`.
   - Runs deterministic inference on `test.csv`.
   - Aggregates recording predictions by `healthCode`.
   - Computes:
     - Participant Accuracy & Balanced Accuracy
     - Sensitivity (PD recall) & Specificity (Control recall)
     - Participant F1 score & ROC-AUC
     - Confusion Matrix
   - Writes structured JSON and CSV reports into `artifacts/models/deep_learning/wav2vec2_finetuned/`.

---

## 5. Verification & Safety Protocol Before Full Execution

Before launching multi-hour training, we perform 4 mandatory verification checks:
1. **Synthetic Forward/Backward Pass**: Run a dummy tensor batch `(4, 80000)` through the model to verify tensor dimensions, loss calculation, backward pass, and optimizer step.
2. **Layer Gradient Verification**: Verify that CNN layers and Transformer layers 0–5 have `requires_grad=False`, and layers 6–11 + attention pooling have `requires_grad=True`.
3. **VRAM Footprint Check**: Ensure active GPU allocation stays below 3.2 GB during the synthetic backward pass.
4. **Single-Batch Overfit Test**: Overfit on a single batch of 4 audio files for 10 iterations to verify that cross-entropy loss monotonically decreases toward zero.

