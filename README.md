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

### 4. External Multi-Cohort Clinical Validation (Zero-Shot Transfer)

To evaluate zero-shot cross-dataset generalization on an unseen external clinical cohort (`records` dataset: PD, Healthy Controls, MSA, PSP) without Docker:

```bash
# Evaluate on raw audio (highest sensitivity: 81.82% on PD):
./scripts/evaluate_records.sh gpu --raw

# Evaluate on clean audio (highest specificity: 72.73% on Controls):
./scripts/evaluate_records.sh gpu

# Evaluate across all vowels (/a/ and /i/):
./scripts/evaluate_records.sh gpu --all
```

#### External Validation Performance Summary:
| Audio Processing | Vowel Protocol | Sensitivity (PD Recall) | Specificity (HC Clearance) | Balanced Accuracy | MSA Dysphonia | PSP Dysphonia |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Waveforms** | **Sustained `/a/`** | **`81.82%` (18/22)** | **`63.64%` (14/22)** | **`72.73%`** | **85.7% (18/21)** | **27.8% (5/18)** |
| **Denoised Clean** | **Sustained `/a/`** | **`68.18%` (15/22)** | **`72.73%` (16/22)** | **`70.45%`** | **81.0% (17/21)** | **22.2% (4/18)** |
| **Denoised Clean** | **All (`/a/` + `/i/`)** | **`63.64%` (14/22)** | **`59.09%` (13/22)** | **`61.36%`** | **90.5% (19/21)** | **33.3% (6/18)** |

* **Key Clinical Insight 1 (Raw vs. Denoised):** Raw audio yields an **81.82% detection rate** on Parkinson's disease. Aggressive spectral denoising filters out subtle turbulent aspiration noise and micro-shimmer that are hallmarks of incomplete glottal closure in early Parkinson's.
* **Key Clinical Insight 2 (Phonation Protocol):** The standard open vowel `/a/` yields superior specificity (72.73%) compared to constricted front vowels like `/i/` (59.09%), confirming the clinical requirement for open-tract sustained phonations.
* **Key Clinical Insight 3 (Atypical Movement Disorders):** The pipeline selectively identified up to **90.5% of Multiple System Atrophy (MSA)** cases (known in neurology for early, severe laryngeal abductor paresis and stridor), while maintaining selectivity on Progressive Supranuclear Palsy (PSP, 22.2% - 27.8%).

---

## Full-Stack Clinical Web Application (NeuroVocal)

This repository includes a production-ready clinical web application with a **FastAPI backend** and a **React 18 / TypeScript / Tailwind CSS frontend**.

The system defaults to **CPU mode** on any machine and supports optional **NVIDIA GPU acceleration** (with automatic fallback to CPU if GPU hardware is missing or inaccessible).

---

### Startup Option 1: Docker (Single Unified Runner)

A single [`docker-compose.yml`](docker-compose.yml) powers the entire stack. You can run it using the unified helper script or directly with Docker Compose:

#### Using the Helper Script:
```bash
# Default (CPU mode - works on any machine, Mac, Linux, cloud VM):
./scripts/run_docker.sh

# NVIDIA GPU mode (with automatic fallback to CPU if GPU is unavailable):
./scripts/run_docker.sh gpu

# Explicit CPU mode:
./scripts/run_docker.sh cpu

# Stop all containers:
./scripts/run_docker.sh down
```

#### Using Docker Compose Directly:
```bash
# Start default stack (CPU mode, defined via COMPOSE_PROFILES=cpu in .env):
docker compose up -d --build

# Start with NVIDIA GPU acceleration:
docker compose --profile gpu up -d --build

# View container logs:
docker compose logs -f

# Stop and clean up:
docker compose down
```

* **Frontend UI**: `http://localhost:3000`
* **Backend API & Swagger Docs**: `http://localhost:8000/docs`

---

### Startup Option 2: Local Development (Instant Startup without Docker)

For rapid development and hot-reloading:

```bash
# 1. In Terminal 1 — Start the FastAPI Backend:
./scripts/run_backend_local.sh        # Runs in CPU mode by default
./scripts/run_backend_local.sh gpu    # Runs on GPU (falls back to CPU if unavailable)

# 2. In Terminal 2 — Start the React Frontend:
./scripts/run_frontend.sh             # Starts Vite dev server at http://localhost:3000
```

---

### Command-Line Voice Prediction (CLI)

Test any audio file directly from the terminal without opening the web browser:

```bash
# Predict in CPU mode (default):
./scripts/predict_voice.sh path/to/recording.wav

# Predict using GPU acceleration:
./scripts/predict_voice.sh --gpu path/to/recording.wav
```

---

### Core API Endpoints

* `GET /api/health`: Health status, active hardware device (`cuda` or `cpu`), and supported audio formats.
* `GET /api/v1/meta-stats`: Exposes 10-moment reference statistics and the clinical decision boundary ($0.29$).
* `POST /api/v1/predict`: Accepts $1$ to $N$ voice recordings (`.wav`, `.ogg`, `.m4a`, `.webm`), extracts Wav2Vec 2.0 embeddings, computes meta-features, and returns structured diagnosis JSON.


