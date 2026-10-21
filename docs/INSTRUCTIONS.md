# Project Architecture Directives & Constraints

> **Status note (2026-10-21):** This file records the project's initial directives. The final shipped model uses **WaveLM Base+, 2 frozen / 10 trainable Transformer layers (layers 1–2 frozen, 3–12 trainable), and 8.0-second audio**. The "frozen lower 6 / trainable upper 6" constraint below describes the earlier Wav2Vec2 plan, not the deployed checkpoint. See `README.md` and `DECISIONS.md` (ADR-009/011/012) for the final configuration.

This document records the architectural constraints, hardware budgets, and evaluation guidelines for the voice-based Parkinson's detection project.

---

## 1. Core User Directives

1. **Be Completely Sure Before Implementing**
   - No experimental or guesswork code changes. Every architectural component, mathematical formulation, parameter-budget calculation, and hyperparameter must be verified against empirical evidence and domain constraints before writing code.
   - Run verification checks (e.g., dummy forward/backward passes, dimension audits, VRAM calculations) before launching full-scale runs.

2. **Docker-First Containerization for Local and Cloud Portability**
   - The user develops on a Linux workstation equipped with an NVIDIA GeForce GTX 1650 (4 GB VRAM) running Python 3.14 on the host. Host Python 3.14 lacks official PyTorch CUDA wheels.
   - The user requires the ability to run locally when desired, but also deploy seamlessly to **cloud GPUs from GridShare** (or RunPod/Lambda) without altering application code or encountering environment mismatches.
   - All deep learning training and evaluation must be encapsulated in Docker containers with volume mounts for `data/`, `artifacts/`, and Hugging Face model cache (`.cache/huggingface`).

3. **Overcome Baseline Generalization Ceiling on Held-Out Test Data**
   - Previous baseline classifiers (linear models, SVMs, and gradient-boosted trees on openSMILE acoustic features) hit a ceiling of **72.0% participant-level accuracy on the held-out test split**, despite achieving 81.2% on validation.
   - The architecture scientifically addresses why static representations failed (phoneme bias, naïve mean pooling, acoustic background overfitting) and implements end-to-end representation learning, learnable temporal attention pooling, SpecAugment, and count-invariant Gated Attention-MIL to maximize genuine acoustic discrimination.

4. **Document Before Starting**
   - Full documentation, architectural specifications, decisions (`DECISIONS.md`), changelog (`CHANGELOG.md`), and step-by-step implementation plans must be committed before implementing codebase files.

---

## 2. Hardware Constraints & Operational Profiles

### Local Profile (NVIDIA GeForce GTX 1650 - 4 GB VRAM)
- Physical VRAM cap: 4,096 MiB.
- Must operate within **< 3.2 GB peak VRAM** to prevent CUDA Out-Of-Memory (OOM) errors.
- Mandatory techniques:
  - PyTorch Automatic Mixed Precision (AMP `fp16`).
  - Frozen 7-layer CNN feature extractor.
  - Initial plan: froze lower 6 Transformer encoder layers (layers 1–6).
  - Shipped checkpoint: layers 1–2 frozen, layers 3–12 trainable (initial plan: upper 6, layers 7–12) with low learning rate ($2 \times 10^{-5}$).
  - PyTorch Gradient Checkpointing enabled on the Transformer backbone (saves ~60% activation memory).
  - Physical batch size = 4, Gradient Accumulation steps = 8 ($\text{Effective Batch Size} = 32$).

### Cloud GPU Profile (GridShare - A100 / RTX 3090 / T4)
- Identical Docker container image and codebase.
- Command-line flags switch to high-throughput cloud settings:
  - Batch size = 16 or 32.
  - Gradient Accumulation steps = 1 or 2 ($\text{Effective Batch Size} = 32$).
  - Training duration drops from ~2.5–3.5 hours locally to ~25–35 minutes on cloud GPU.

---

## 3. Strict Research Integrity Rules

1. **Participant Separation**: Zero leakage across `train.csv`, `validation.csv`, and `test.csv`. Splitting is strictly by `healthCode`.
2. **Single-Use Held-Out Test Set**: The test set is evaluated exactly once after the best model checkpoint and decision threshold are locked on the validation set.
3. **Medical Disclaimer**: Model predictions are research screening artifacts, not clinical diagnostic verdicts.

