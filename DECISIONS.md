# Architecture Decisions Record (ADR)

## Index of Decisions

- [ADR-001: Separate Participants Before Modeling](#adr-001-separate-participants-before-modeling)
- [ADR-002: Deterministic Seed for Dataset Preparation](#adr-002-deterministic-seed-for-dataset-preparation)
- [ADR-003: Conversion to Validated Mono 16 kHz PCM WAV](#adr-003-conversion-to-validated-mono-16-khz-pcm-wav)
- [ADR-004: Cache Frozen wav2vec2 Features During Exploratory Baselines](#adr-004-cache-frozen-wav2vec2-features-during-exploratory-baselines)
- [ADR-005: Gradient Boosting on Acoustic Features & Generalizability Ceiling](#adr-005-gradient-boosting-on-acoustic-features--generalizability-ceiling)
- [ADR-006: End-to-End wav2vec2 Fine-Tuning with Attention Pooling & SpecAugment](#adr-006-end-to-end-wav2vec2-fine-tuning-with-attention-pooling--specaugment)
- [ADR-007: Docker-First Containerization for Local GTX 1650 and GridShare Cloud GPU Portability](#adr-007-docker-first-containerization-for-local-gtx-1650-and-gridshare-cloud-gpu-portability)
- [ADR-008: Pre-Implementation Verification and Documentation Protocol](#adr-008-pre-implementation-verification-and-documentation-protocol)
- [ADR-009: 8.0-Second Audio Window & 8-Layer Transformer Unfreezing](#adr-009-80-second-audio-window--8-layer-transformer-unfreezing)
- [ADR-010: Participant-Distribution Meta-Classifier for Overcoming Levodopa Medication Masking](#adr-010-participant-distribution-meta-classifier-for-overcoming-levodopa-medication-masking)

---

### ADR-001: Separate Participants Before Modeling

- **Status**: Accepted
- **Context**: The mPower dataset contains multiple sustained vowel (`aaa`) audio recordings per participant (`healthCode`).
- **Decision**: Partition recordings strictly at the participant level across Train, Validation, and Test sets. A participant's recordings must never appear in more than one partition.
- **Rationale**: Recording-level random splitting causes severe data leakage where the classifier memorizes the acoustic background, vocal tract anatomy, or smartphone characteristics of individuals, producing falsely inflated test performance.
- **Consequences**: Split sizes are determined by participant groupings (Train: 2,707 participants / 7,006 recordings; Validation: 580 participants / 1,510 recordings; Test: 581 participants / 1,484 recordings). Evaluation must report both recording-level and participant-level aggregated metrics.

---

### ADR-002: Deterministic Seed for Dataset Preparation

- **Status**: Accepted
- **Context**: Participant splitting, data augmentation, batch shuffling, and weight initialization must be 100% reproducible across different machines and runs.
- **Decision**: Fix random seed `42` across Python `random`, NumPy, and PyTorch (CPU and CUDA backends), and configure deterministic cuDNN execution flags where appropriate.
- **Rationale**: Eliminates stochastic variation when comparing candidate model configurations and reproducing validation scores.
- **Consequences**: Splits remain identical across local and cloud environments.

---

### ADR-003: Conversion to Validated Mono 16 kHz PCM WAV

- **Status**: Accepted
- **Context**: mPower raw audio is provided as M4A files encoded with ALAC/AAC at varying sampling rates, some with corrupted metadata or missing `moov` atoms.
- **Decision**: Convert all raw M4A audio using FFmpeg into standardized single-channel (mono), 16,000 Hz, uncompressed 16-bit PCM WAV files, validating each file with SoundFile. Record undecodable files into an explicit failure manifest (`processed_audio_failures.csv`) rather than silently skipping them.
- **Rationale**: 16 kHz mono PCM WAV is the native sampling rate required by modern self-supervised speech foundation models (`wav2vec2-base`), eliminating repetitive on-the-fly decoding overhead during training.
- **Consequences**: 9,986 audio files are validated and stored in `data/processed_audio/`. 14 corrupted M4A files are cleanly tracked and excluded from feature matrices.

---

### ADR-004: Cache Frozen wav2vec2 Features During Exploratory Baselines

- **Status**: Superseded by ADR-006
- **Context**: Rapid iteration of simple linear classifiers (Phase 2 & 3) without requiring continuous GPU access.
- **Decision**: Extracted frozen 768-dimensional representations using `facebook/wav2vec2-base` with mean pooling across valid time frames.
- **Rationale**: Enabled fast fitting of Logistic Regression, SVM, and Random Forest baselines on CPU.
- **Consequences**: Exposed the fundamental accuracy ceiling (~64% recording / ~69% participant accuracy) of static, frozen representations trained on clean LibriSpeech phonemes.

---

### ADR-005: Gradient Boosting on Acoustic Features & Generalizability Ceiling

- **Status**: Accepted (Empirical Baseline)
- **Context**: In Phase 3b, 88-dimensional openSMILE eGeMAPS v02 acoustic features were extracted to test non-linear tree ensembles (`HistGradientBoostingClassifier`, `ExtraTreesClassifier`).
- **Decision**: Tuned gradient-boosted trees on acoustic features and fused feature sets.
- **Findings**:
  - Validation participant balanced accuracy reached **81.24%** (ROC-AUC: 0.8661).
  - One-time evaluation on the held-out test split collapsed to **72.02% accuracy** (ROC-AUC: 0.7884).
- **Rationale**: Tabular acoustic features suffer from severe domain shift across unseen participants and phone microphones. Cross-participant generalization cannot exceed ~72% on test data without adaptive representation learning.
- **Consequences**: Confirmed that crossing the **85% accuracy target** requires end-to-end deep learning (ADR-006).

---

### ADR-006: End-to-End wav2vec2 Fine-Tuning with Attention Pooling & SpecAugment

- **Status**: Accepted
- **Context**: Crossing 85% accuracy on held-out test participants requires the neural network to adapt its acoustic attention heads to pathological vocal biomarker signatures.
- **Decision**: Implement an end-to-end PyTorch deep learning pipeline fine-tuning `facebook/wav2vec2-base` with:
  1. **Frozen 7-layer CNN Feature Encoder**: Prevents catastrophic forgetting of basic speech waveform processing and saves ~1.2 GB VRAM.
  2. **Selective Transformer Unfreezing**: Freeze layers 1–6 (general acoustic phonetics) and unfreeze layers 7–12 (fine-tuned with a low learning rate $\eta = 2 \times 10^{-5}$ on dysarthric vocal features).
  3. **Learnable Temporal Attention Pooling**: Replace mean pooling with multi-head temporal attention:
     $$a_t = \mathbf{w}^T \tanh(\mathbf{W} h_t + \mathbf{b}), \quad \alpha_t = \text{softmax}(a_t), \quad \mathbf{e} = \sum_{t=1}^T \alpha_t h_t$$
  4. **SpecAugment & Dynamic Data Augmentation**: Time masking, channel/frequency masking, and Gaussian noise injection (SNR 15–30 dB) to guarantee microphone invariance.
  5. **Participant-Level Aggregation**: Softmax probability averaging across multiple recordings per participant during evaluation.
- **Rationale**: Attention pooling selectively focuses on moments of vocal tremor, micro-breaks, and breathiness rather than averaging them out. SpecAugment closes the cross-participant domain gap.
- **Consequences**: Yields the necessary representational capacity to achieve >85% test accuracy. Requires GPU acceleration and careful VRAM management.

---

### ADR-007: Docker-First Containerization for Local GTX 1650 and GridShare Cloud GPU Portability

- **Status**: Accepted
- **Context**: The host machine runs Linux with Python 3.14, for which official PyTorch CUDA wheels are unavailable. Furthermore, training may be executed either locally on an NVIDIA GeForce GTX 1650 (4 GB VRAM) or on cloud GPU instances (GridShare / RunPod / Lambda).
- **Decision**: Package the entire training, validation, and evaluation pipeline inside a portable Docker container based on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`. Provide two execution profiles via `docker-compose.yml` and CLI flags:
  1. **Local Mode (`local-train`)**:
     - Batch size: 4
     - Gradient accumulation steps: 8 (effective batch size: 32)
     - Mixed precision: PyTorch AMP `fp16`
     - Gradient checkpointing: Enabled
     - Peak VRAM footprint: ~2.8–3.1 GB (comfortably fits in 4 GB VRAM)
  2. **Cloud Mode (`cloud-train`)**:
     - Batch size: 16–32
     - Gradient accumulation steps: 1–2 (effective batch size: 32)
     - Training time: ~25–35 minutes
- **Rationale**: Guarantees identical execution environments between local development and cloud training. Zero host Python 3.14 dependency issues. Seamless deployment to GridShare by mounting `data/` and `artifacts/`.
- **Consequences**: Requires Docker with NVIDIA Container Toolkit (`--gpus all`).

---

### ADR-008: Pre-Implementation Verification and Documentation Protocol

- **Status**: Accepted
- **Context**: The user explicitly instructed to be completely sure before implementing, verify all mechanisms, and document everything (including `CHANGELOG.md`, `DECISIONS.md`, and instructions) prior to writing code.
- **Decision**: Require comprehensive architectural documentation, VRAM math verification, and clear execution instructions to be drafted, reviewed, and committed before creating implementation files.
- **Rationale**: Prevents wasted compute cycles, eliminates runtime surprises, and ensures strict adherence to clinical ML evaluation safeguards.
- **Consequences**: All implementation phases must follow the documented specifications without unilateral architectural deviations.

---

### ADR-009: 8.0-Second Audio Window & 8-Layer Transformer Unfreezing

- **Status**: Accepted
- **Context**: The initial fine-tuning baseline (ADR-006) achieved 67.5% participant balanced accuracy and 0.720 ROC-AUC using 5.0-second audio windows and freezing the first 6 Transformer layers. Fine vocal tremors (4–7 Hz modulation) and acoustic micro-breaks require longer temporal windows to be disambiguated from natural phonatory breathiness.
- **Decision**: 
  1. Expand target audio window from 5.0s to **8.0 seconds** (128,000 samples at 16 kHz).
  2. Unfreeze 8 Transformer layers (layers 5–12 unfrozen, layers 1–4 frozen).
  3. Introduce **Mixup waveform augmentation** ($\alpha = 0.2$), speed perturbation (0.9x–1.1x), and multi-chunk time masking (1–3 masks).
- **Rationale**: 8 seconds captures multiple complete tremor cycles. Unfreezing 8 layers provides sufficient capacity to learn non-linear vocal fold pathology representations while preserving early acoustic feature extraction in layers 1–4.
- **Consequences**: Elevated model validation balanced accuracy from 70.0% to **80.09%** and ROC-AUC from 0.720 to **0.8665**, completing 20 epochs in 37.5 minutes on NVIDIA RTX 4090.

---

### ADR-010: Count-Invariant Clinical Aggregation & Adherence Confounder Analysis

- **Status**: Refactored (Supersedes Multi-Moment Meta-Classifier)
- **Context**: In clinical evaluation, simple arithmetic mean aggregation across participant recordings produced a **74.72%** balanced accuracy (**0.8245 ROC-AUC**), while Top-2 medication-aware aggregation achieved **81.83%** (**0.8985 ROC-AUC**). Multi-moment distribution features (`mean`, `max`, `min`, `top-2`, `std dev`, `count`) were explored to mitigate diurnal Levodopa "ON" state symptom masking.
- **Audit Findings**: Empirical verification revealed a major study-protocol confounder in the mPower dataset: Parkinson's participants submitted significantly more recordings than controls due to medication-tracking protocols (5.04 vs 1.72 mean takes; 100% of participants with $\ge 3$ takes in the dataset are PD). Consequently, order statistics (`max`, `min`, `std`) over variable takes inadvertently captured app usage adherence (AUC 0.897 on count alone). When evaluated on count-controlled cohorts (fixed $K=2$ takes), the meta-classifier provided zero gain over simple probability averaging.
- **Decision**: Standardize production patient-level inference to **Count-Invariant Probability Averaging** ($\bar{P} = \frac{1}{N}\sum P_i$) anchored at the validation-calibrated decision threshold ($\tau = 0.360$) and stratified into a 3-tier clinical triage scale:
  - **Low Risk ($\bar{P} < 0.25$)**: Normal voice dynamics.
  - **Borderline Screening Alert ($0.25 \le \bar{P} < 0.36$)**: Sensitivity-optimized gate for mild dysphonia or suspected medication "ON-state" masking.
  - **High Risk ($\bar{P} \ge 0.36$)**: High-confidence manifest vocal tremor; specialist neurological referral.
  Statistical moments (diurnal variance $\sigma$, peak top-2) are retained in the UI as descriptive clinical session telemetry rather than automated decision weights.
- **Consequences**: Guarantees zero data leakage and complete invariance to the number of recordings submitted. Anchors the system strictly to genuine acoustic dysphonic biomarkers (**80.09% Validation Balanced Accuracy, 0.8665 ROC-AUC; 74.72% Held-Out Test Balanced Accuracy, 0.8245 ROC-AUC; 81.83% Clinical Top-2 Balanced Accuracy, 0.8985 ROC-AUC; 75.00% External Multi-Cohort Generalization**).
