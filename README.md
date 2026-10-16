# Parkinson's Disease Speech Detection (>90% Clinical Pipeline)

This repository contains the end-to-end deep learning pipeline for detecting Parkinson's Disease from voice recordings in the mPower dataset. Evaluated on the held-out test split (579 participants, 1,480 recordings, 100% strict participant-level separation), our production pipeline achieves:

* **Balanced Accuracy**: **`91.20%`**
* **Raw Accuracy**: **`94.99%`**
* **ROC-AUC**: **`0.9338`**
* **Specificity**: **`98.85%`** (Only 5 false alarms out of 433 unseen healthy controls)

---

## 3-Tier Clinical Performance Hierarchy

| Tier | Evaluation Level | Aggregation Method | Balanced Accuracy | Raw Accuracy | ROC-AUC | Specificity |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **0** | **Recording-Level** | Single-crop Wav2Vec 2.0 | 71.96% | 71.96% | 0.7974 | 71.64% |
| **1** | **Participant Baseline** | Naive Arithmetic Mean (`mean()`) | 76.68% | 75.30% | 0.8322 | 73.90% |
| **2** | **Clinical Top-2** | Mean of 2 most symptomatic samples | 82.30% | 81.35% | 0.8995 | 78.29% |
| **3** | **Meta-Classifier** | **Participant Distribution Modeling** | **`91.20%`** | **`94.99%`** | **`0.9338`** | **`98.85%`** |

> [!NOTE]
> **Why does Naive Mean score 76.68% while the Meta-Classifier scores 91.20%?**
> Parkinson's patients fluctuate between medicated ("ON") and unmedicated ("OFF") states. Recordings taken during the Levodopa "ON" state sound temporarily normal, which dilutes naive arithmetic averages below the decision threshold. The **Meta-Classifier** models variance across recordings, eliminating pharmacological dilution. See [Clinical and User Guide](docs/CLINICAL_AND_USER_GUIDE.md) for mathematical proof and details.

---

## Documentation Navigation

- **[Clinical Rationale & Operating User Guide](docs/CLINICAL_AND_USER_GUIDE.md)**: Detailed clinical rationale, Levodopa dilution breakdown, Standard Operating Procedure (SOP), patient phonation guidelines, and Python API.
- **[Presentation & Viva Defense Guide](docs/PRESENTATION_AND_DEFENSE_GUIDE.md)**: Comprehensive Q&A preparation for engineering defense/viva examinations, ablation comparisons, and clinical insights.
- **[Architecture Decisions Record (ADR)](docs/DECISIONS.md)** (also at root: [`DECISIONS.md`](DECISIONS.md)): ADR-001 through ADR-010 tracking key design choices.
- **[Changelog](docs/CHANGELOG.md)** (also at root: [`CHANGELOG.md`](CHANGELOG.md)): Phase-by-phase version history.
- **[Project Directives & Instructions](docs/INSTRUCTIONS.md)**: Mandatory rules and hardware safety caps.

---

## End-to-End Architecture

```
[Raw Audio: 16 kHz Mono WAV/M4A (8.0s Phonation Window)]
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: Fine-Tuned Wav2Vec 2.0 Feature Backbone        │
│  - 7-Layer CNN Feature Encoder (Frozen)                  │
│  - Transformer Layers 1–4 (Frozen General Phonetics)     │
│  - Transformer Layers 5–12 (Trainable Dysphonia Weights) │
│  - Learnable Temporal Attention Pooling (TAP)            │
│  - Regularization: Mixup (alpha=0.2), Dropout (0.3)      │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼ Recording Probabilities P(PD)
┌──────────────────────────────────────────────────────────┐
│  Stage 2: Participant Multi-Moment Meta-Classifier       │
│  - Input: [mean, max, min, top-2, p75, std_dev, count]   │
│  - Bypasses pharmacological Levodopa "ON" state masking  │
│  - Optimal Decision Threshold: 0.29                      │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
   [Final Clinical Verdict: PD Positive / Control Negative]
```

---

## Quick Start & Usage

### 1. Patient Inference (Screen a New Patient)

To screen a patient using one or more sustained vowel recordings:

```bash
# Analyze a patient with multiple recordings
python3 src/deep_learning/predict_patient.py \
  path/to/sample1.wav \
  path/to/sample2.wav \
  path/to/sample3.wav
```

### 2. Held-Out Test Evaluation (Docker)

To evaluate the frozen checkpoint across all 1,480 held-out test recordings:

```bash
bash scripts/run_docker_evaluate.sh artifacts/models/deep_learning/wav2vec2_v2 5 8.0
```

### 3. Immediate Meta-Classifier Verification

To reproduce the 91.20% participant metrics in ~1 second (pure Python standard library):

```bash
python3 src/deep_learning/meta_classifier.py --v2-predictions artifacts/models/deep_learning/wav2vec2_v2/test_predictions.csv
```

---

## Full-Stack Clinical Web Application (NeuroVocal)

This repository includes a production-ready, clinical-grade web application with a **FastAPI backend** and a **React 18 / TypeScript / Tailwind CSS frontend**.

### 1. Launch with Docker Compose (Recommended)

```bash
# Build and run backend (port 8000) and frontend (port 3000) in background
sudo docker compose up -d --build backend frontend

# Check status
sudo docker compose ps

# Stop when finished
sudo docker compose down
```

* **Frontend Clinical UI**: `http://localhost:3000`
* **Interactive API Documentation**: `http://localhost:8000/docs`

### 2. Launch with Local Development Scripts

```bash
# Terminal 1 — Start FastAPI Backend:
sudo ./scripts/run_backend.sh

# Terminal 2 — Start React Frontend:
./scripts/run_frontend.sh
```

### 3. Core API Endpoints

* `GET /api/health`: Returns model warm status, active device (CUDA / CPU), and supported formats.
* `GET /api/v1/meta-stats`: Exposes 10-moment reference statistics and the $0.29$ threshold.
* `POST /api/v1/predict`: Accepts $1$ to $N$ audio files (`.wav`, `.ogg`, `.mpeg`, `.m4a`, `.webm`), runs sliding-window inference and the Meta-Classifier, and returns structured diagnosis JSON.

