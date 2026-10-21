# Changelog

All notable changes to the `parkinsons-detector` project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Phase 7] - 2026-10-21 - Documentation Accuracy Pass

### Fixed
- Corrected the documented backbone configuration to match the shipped checkpoint: **layers 1–2 frozen, layers 3–12 trainable (10 unfrozen)**, 15 epochs (best epoch 14), not 8 unfrozen layers / 20 epochs.
- Corrected participant-error wording to distinguish gross changes from net changes: **net +7 PD** (8 rescued, 1 degraded) and **net −9 false positives** (12 cleared, 3 new).
- Corrected dataset counts to separate the 10,000 selected / 9,986 decoded recordings (3,868 / 3,863 participants) from the metadata splits and the 579-participant, 1,480-recording evaluated test cohort.
- Aligned triage documentation with runtime thresholds: low-risk `<0.20`, borderline `0.20–0.30`, high-risk `≥0.30`; `0.40` is only a confidence marker.
- Replaced unsupported or stale statistics in the user and defense guides with values recomputed from `artifacts/models/deep_learning/wavelm/test_predictions.csv`.
- Marked the Phase 4c meta-classifier results as retracted/confounded and annotated historical planning documents.

---

## [Phase 6] - 2026-10-18 - WaveLM Backbone, Gated Attention-MIL & Clinical Telemetry Figures

### Added
- **Champion Single-Model Architecture (`WaveLM Base+` + Gated Attention-MIL)**:
  - Selected fine-tuned `microsoft/wavlm-base-plus` as champion acoustic foundation model, pre-trained with masked speech denoising across 94,000 hours of noisy audio.
  - Implemented `src/mil/model.py`: Gated Attention Multiple Instance Learning head (Ilse et al., ICML 2018) with scale factor $\frac{1}{\sqrt{L}}$ and LayerNorm.
  - Slices each participant take into **3 evenly spaced 8.0-second crops** (not a dense sliding window), dynamically downweighting uninformative transient noise and silent air while concentrating attention on sustained phonation plateaus.
  - Exported trained weights to `artifacts/mil/mil_head.pt` and `artifacts/models/deep_learning/wavelm/mil_head.pt`.
- **Grounded Benchmark Metrics (Held-Out Test Set, 579 Participants)**:
  - **Balanced Accuracy**: Elevated from 79.11% (arithmetic mean) to **`82.54%` [95% CI: 79.0%, 86.1%]** (+3.44% gain, $\tau = 0.300$, $\tau_{\text{rec}} = 0.310$).
  - **PD Sensitivity**: Increased from 78.77% to **`83.56%`** (122 of 146 PD participants; **net +7**, comprising 8 rescued and 1 degraded).
  - **Healthy Control Specificity**: Increased from 79.45% to **`81.52%`** (**net −9 false positives**, comprising 12 controls cleared and 3 new false positives; FP 89 → 80).
  - **Held-Out Test ROC-AUC**: **`0.8791` [95% CI: 0.845, 0.912]**.
  - **Clean Training Protocol**: Trained strictly on 2,705 training participant bags (`bags_train.pt`) with zero test leakage. Operating thresholds were fixed at $\tau = 0.300$ and $\tau_{\text{rec}} = 0.310$ from validation analysis and hard-coded in `scripts/train_mil_clean.py`; the held-out test set was scored once.
- **High-Resolution Evaluation Figures (`docs/figures/`)**:
  - `roc_curves.png`: High-resolution ROC curves for Baseline (AUC = 0.8585) vs. Attention-MIL (AUC = 0.8791).
  - `confusion_matrices.png`: Side-by-side confusion matrices for Baseline vs. Attention-MIL (N=579) with clear actual condition and model prediction labels.
  - `clinical_triage_distribution.png`: Score density distribution across decision regions.
- **Acoustic Signal Integrity (ADR-013)**:
  - Standardized preprocessing to 16 kHz mono resampling and peak normalization; avoided aggressive digital filtering to protect fundamental vocal pitch harmonics ($F_0$).

---

## [Phase 5] - 2026-10-17 - Count-Invariant Pipeline & Adherence Confounder Analysis

### Added
- **Count-Invariant Inference & Held-Out Test Evaluation**:
  - Achieved **74.72% Balanced Accuracy** (**0.8245 ROC-AUC**, 78.08% Sensitivity, 71.36% Specificity) on 579 held-out test participants with the **Wav2Vec 2.0 baseline** under count-invariant aggregation.
  - Achieved **81.83% Balanced Accuracy** (**0.8985 ROC-AUC**, 86.99% Sensitivity, 76.67% Specificity) under Top-2 aggregation for that baseline.
  - Retained the 10-moment distribution meta-classifier (**90.86% Balanced Accuracy**, **0.9346 ROC-AUC**) as an exploratory ablation. **This result is retracted as a performance claim:** the features leak recording-count/adherence information (see ADR-010).
- **Documentation**:
  - Documented acoustic impact of spectral noise reduction vs. raw audio preservation on early dysphonia detection.
  - Validated superiority of the standard sustained `/a/` open-vowel protocol over constricted front vowels like `/i/`.

---

## [Phase 4] - End-to-End Fine-Tuning & Docker Containerization

### Added
- **Docker-First Containerization**:
  - `Dockerfile` based on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` for environment isolation from host Python 3.14.
  - `docker-compose.yml` defining `local-train` (4GB VRAM optimization for GTX 1650), `cloud-train` (high-throughput for GridShare / RunPod), and `evaluate` services.
  - Shell launchers: the originally documented `scripts/run_local.sh` / `scripts/run_cloud.sh` names do not exist in the current tree; current equivalents are `scripts/run_docker.sh`, `scripts/run_docker_evaluate.sh`, and `scripts/train_wavelm_cloud.sh`.
- **End-to-End Speech Fine-Tuning Architecture**:
  - `src/deep_learning/dataset.py`: PyTorch dataset loading 16 kHz mono WAVs dynamically, applying padding/truncation to 5 seconds, and SpecAugment masking.
  - `src/deep_learning/model.py`: `Wav2Vec2ForParkinsons` neural network with frozen CNN encoder, frozen lower Transformer layers (1–6), unfrozen upper Transformer layers (7–12), learnable temporal attention pooling, and classification head.
  - `src/deep_learning/train.py`: PyTorch training engine featuring AMP `fp16`, gradient checkpointing, gradient accumulation, cosine annealing schedule with linear warmup, validation loss/balanced accuracy tracking, and model checkpointing.
  - `src/deep_learning/evaluate.py`: One-time test evaluation engine computing recording-level and participant-aggregated metrics, ROC-AUC, sensitivity, specificity, and confusion matrix.
- **Documentation & Standards**:
  - `docs/INSTRUCTIONS.md`: Mandatory instructions, hardware safety limits, and execution rules.
  - `docs/DECISIONS.md`: Architectural decisions ADR-001 through ADR-008.
  - `docs/IMPLEMENTATION_PLAN.md`: Comprehensive end-to-end plan, VRAM mathematics, and verification procedures.

---

## [Phase 4c] - 2026-10-15 - Clinical Meta-Classifier & Peak Benchmark Results

> **Retraction (2026-10-16, ADR-010):** The results in this section are confounded by participant recording-count/adherence. `count`, `max`, `min`, and `std` features encode how often the app was used, not just voice quality. They are retained here as historical exploratory results and must **not** be cited as validated diagnostic performance.

### Added
- **Participant-Level Distribution Analysis**:
  - Statistical aggregation engine over recording distributions (mean, max, min, top-2, std dev, count).
  - Initially interpreted as resolving the clinical "ON-medication" dilution phenomenon; later audit showed the gain was driven by the recording-count confounder.
- **Exploratory (confounded) Test Results**:
  - **ROC-AUC**: **0.9338** (not a clean benchmark; do not compare with published mPower results).
  - **Participant Balanced Accuracy**: **91.20%**.
  - **Participant Raw Accuracy**: **94.99%**.
  - **Sensitivity (PD Recall)**: **83.56%** (122 / 146).
  - **Specificity (Healthy Control)**: **98.85%** (428 / 433).
  - **F1 Score**: **0.8938**.

---

## [Phase 4b] - 2026-10-15 - Multi-Crop TTA & Enhanced Regularization

### Added
- **Multi-Crop Test-Time Augmentation (TTA)**:
  - Added `--num-crops` and `--crop-stride` in `src/deep_learning/evaluate.py`. Samples evenly spaced crops and averages predictions per recording to stabilize test inference (production uses 3 crops).
- **Enhanced Augmentation & Regularization**:
  - Waveform speed perturbation (0.9x–1.1x), multi-time masking (1–3 masks), wider gain scaling (0.7–1.3), and SNR noise injection (12–30 dB) in `dataset.py`.
  - Added Mixup augmentation (`--mixup-alpha 0.2`), higher dropout (0.3), label smoothing (0.1), and unfreezing of 8 transformer layers in `model.py` and `train.py` (the final shipped WaveLM checkpoint used 2 frozen / 10 unfrozen layers; see Phase 6 and ADR-009).
  - Created cloud execution scripts for one-command execution on cloud GPUs.

---

## [0.3.1] - 2026-10-15 - Phase 3b: Acoustic eGeMAPS & Gradient Boosting Baseline

### Added
- Feature extraction using openSMILE eGeMAPS v02 (88 acoustic features per recording) across 9,986 audio files.
- Evaluation suite comparing 24 candidate classifiers (Logistic Regression, RBF SVM, Random Forest, ExtraTrees, and HistGradientBoosting across acoustic and fused spaces).

### Results & Findings
- Best validation model: `acoustic__hist_gradient_boosting_reg` (Participant Balanced Accuracy: 81.24%, ROC-AUC: 0.8661).
- Held-out test evaluation: Participant accuracy dropped to **72.02%** (ROC-AUC: 0.7884).
- Established the empirical limitation: static tabular acoustic features overfit to participant backgrounds and cannot reach >85% on unseen test participants.

---

## [0.3.0] - 2026-10-14 - Phase 3: Classifier Baseline Comparison

### Added
- Standardized baseline comparison pipeline (`src/train_model.py`, `src/validate_model.py`, `src/evaluate_model.py`).
- Linear models, SVM, and Random Forest evaluated on frozen 768-dimensional wav2vec2 embeddings.
- Participant aggregation logic averaging recording probabilities per `healthCode`.

### Results & Findings
- Frozen wav2vec2 embeddings yielded ~64% recording accuracy and ~69% participant accuracy.
- Highlighted that frozen speech representations trained on LibriSpeech do not capture pathological voice tremor.

---

## [0.2.0] - 2026-10-14 - Phase 2: Frozen Feature Extraction

### Added
- Resumable feature extraction pipeline utilizing `facebook/wav2vec2-base`.
- Masked mean pooling across valid frame timestamps.
- Generation of 768-dimensional feature vectors stored in `artifacts/features/wav2vec2_base/`.

---

## [0.1.0] - 2026-10-14 - Phase 1: Audio Preprocessing & Dataset Validation

### Added
- Synapse metadata loader and deterministic participant split script (`split_dataset.py`, seed 42).
- FFmpeg audio conversion to standardized 16 kHz mono uncompressed PCM WAV files.
- SoundFile audio validation checking sample rates, channels, durations, and decode integrity.
- Manifest generation: 9,986 converted recordings (`data/metadata/processed_audio_manifest.csv`) and 14 cleanly tracked container/codec failures (`data/metadata/processed_audio_failures.csv`).
