# Parkinson's Disease Speech Detection (>85% Target Pipeline)

This repository contains the end-to-end deep learning pipeline for detecting Parkinson's Disease from voice recordings in the mPower dataset, designed to exceed **85% participant-level accuracy** on held-out test data.

The project is packaged for Docker to ensure 100% environment reproducibility across local development (NVIDIA GeForce GTX 1650, 4 GB VRAM) and cloud GPU instances (GridShare / RunPod / Lambda).

---

## Documentation Navigation

- **[Project Directives & Instructions](docs/INSTRUCTIONS.md)**: Mandatory rules, hardware safety caps, and execution protocols.
- **[Architecture Decisions Record (ADR)](docs/DECISIONS.md)** (also at root: [`DECISIONS.md`](DECISIONS.md)): Detailed decisions ADR-001 through ADR-008.
- **[Changelog](docs/CHANGELOG.md)** (also at root: [`CHANGELOG.md`](CHANGELOG.md)): Phase-by-phase version history from audio preprocessing to deep learning fine-tuning.
- **[Implementation Plan & Mathematical Justification](docs/IMPLEMENTATION_PLAN.md)**: Mathematical formulation of temporal attention pooling, SpecAugment, VRAM budgets, and pipeline architecture.

---

## Architecture at a Glance

```
Raw Audio (16 kHz mono WAV)
    ├──► 1. 7-Layer CNN Feature Extractor (Frozen, saves ~1.2 GB VRAM)
    ├──► 2. SpecAugment & Gaussian Noise Injection (Microphone invariance)
    ├──► 3. Transformer Layers 1–6 (Frozen, acoustic primitives)
    ├──► 4. Transformer Layers 7–12 (Trainable, fine-tunes to dysphonia & tremor)
    ├──► 5. Learnable Temporal Attention Pooling (Focuses on phonation micro-breaks)
    ├──► 6. Classification Head (LayerNorm -> Linear -> GELU -> Linear)
    └──► 7. Participant-Level Aggregation (Averages recording probabilities per healthCode)
```

---

## Execution Modes (Docker-First)

### 1. Local Mode (NVIDIA GTX 1650 - 4GB VRAM)
- Uses PyTorch AMP `fp16`, gradient checkpointing, batch size 4, gradient accumulation steps 8.
- Peak VRAM footprint: **~2.8–3.1 GB** (stays safely within physical 4.0 GB VRAM limit).
```bash
docker compose run --rm --gpus all local-train
```

### 2. Cloud GPU Mode (GridShare / RunPod / Lambda)
- Uses batch size 16–32 with high GPU throughput.
- Full 10-epoch training in **~25–35 minutes**.
```bash
docker compose run --rm --gpus all cloud-train
```

### 3. Evaluation Mode (Held-Out Test Set)
- Runs one-time evaluation on `data/metadata/test.csv` using the best checkpoint chosen on validation.
```bash
docker compose run --rm --gpus all evaluate
```

