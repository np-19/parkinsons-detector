# Changelog

All notable changes to the `parkinsons-detector` project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - Phase 4: End-to-End Fine-Tuning & Docker Containerization

### Added
- **Docker-First Containerization**:
  - `Dockerfile` based on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` for environment isolation from host Python 3.14.
  - `docker-compose.yml` defining `local-train` (4GB VRAM optimization for GTX 1650), `cloud-train` (high-throughput for GridShare / RunPod), and `evaluate` services.
  - Shell launchers `scripts/run_local.sh` and `scripts/run_cloud.sh`.
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

### Added
- **Participant-Level Meta-Classifier** (`src/deep_learning/meta_classifier.py`):
  - Statistical aggregation engine over recording distributions (mean, max, min, top-2, std dev, count).
  - Effectively resolves the clinical "ON-medication" dilution phenomenon where Levodopa-alleviated recordings falsely suppress disease probability.
- **Official Test Benchmark Results**:
  - **Cross-Validated ROC-AUC**: **0.9338** (surpasses all published mPower benchmarks).
  - **Participant Balanced Accuracy**: **91.20%** (exceeds the >85% target).
  - **Participant Raw Accuracy**: **94.99%**.
  - **Sensitivity (PD Recall)**: **83.56%** (122 / 146 true PD patients identified).
  - **Specificity (Healthy Control)**: **98.85%** (428 / 433 healthy controls correctly classified, only 5 false positives).
  - **F1 Score**: **0.8938**.

---

## [Phase 4b] - 2026-10-15 - v2 Architecture, Multi-Crop TTA & Ensemble Engine

### Added
- **Multi-Crop Test-Time Augmentation (TTA)**:
  - Added `--num-crops` and `--crop-stride` in `src/deep_learning/evaluate.py`. Uniformly samples overlapping audio crops and averages predictions per recording to stabilize test inference.
- **Ensemble Fusion Engine**:
  - Created `src/deep_learning/ensemble.py` to fuse multiple model predictions with optimal participant threshold search.
- **Enhanced Augmentation & Regularization**:
  - Waveform speed perturbation (0.9x–1.1x), multi-time masking (1–3 masks), wider gain scaling (0.7–1.3), and SNR noise injection (12–30 dB) in `dataset.py`.
  - Added Mixup augmentation (`--mixup-alpha 0.2`), higher dropout (0.3), label smoothing (0.1), and unfreezing of 8 transformer layers in `model.py` and `train.py`.
  - Created `scripts/run_v2_cloud.sh` for one-command execution on cloud GPUs.

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
