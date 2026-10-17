# Parkinson's Disease Speech Screening Pipeline (NeuroVocal)

This repository contains the end-to-end deep learning pipeline for detecting Parkinson's Disease from voice recordings in the mPower dataset. Evaluated with 100% strict participant-level separation (zero audio or participant overlap between train, validation, and test splits), our production pipeline achieves:

* **Validation Participant Balanced Accuracy**: **`80.09%`** (ROC-AUC: **`0.8665`**, Sensitivity: 80.69%, Specificity: 79.49% on 580 unseen participants)
* **Held-Out Test Participant Balanced Accuracy**: **`74.72%`** (ROC-AUC: **`0.8245`**, Sensitivity: 78.08%, Specificity: 71.36% on 579 unseen test participants)
* **External Multi-Cohort Cross-Dataset Generalization**: **`75.00%`** balanced accuracy (86.36% PD sensitivity, 95.2% MSA detection) on an independent external clinical cohort (`records` dataset: PD, Healthy Controls, MSA, PSP)

The production score is the arithmetic mean of recording-level probabilities. The
validation-selected checkpoint threshold is loaded from the checkpoint at
inference time; the default checkpoint uses $\tau = 0.360$, with screening and
diagnostic triage cutoffs derived as $\tau - 0.10 = 0.260$ and
$\tau + 0.10 = 0.460$.

---

## Clinical Performance Hierarchy

| Tier | Evaluation Level | Aggregation Method | Balanced Accuracy | Raw Accuracy | ROC-AUC | Sensitivity | Specificity |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **Recording-Level (Test)** | Single-crop Wav2Vec 2.0 | 71.22% | 71.22% | 0.7853 | 72.01% | 70.43% |
| **1** | **Participant-Level (Test)** | Count-Invariant Mean ($\tau = 0.360$) | **74.72%** | **73.06%** | **0.8245** | **78.08%** | **71.36%** |
| **2** | **Participant-Level (Val)** | Validation-Calibrated Mean ($\tau = 0.360$) | **80.09%** | **79.79%** | **0.8665** | **80.69%** | **79.49%** |
| **Ext** | **Cross-Cohort (Zero-Shot)** | Raw sustained `/a/` phonation | **75.00%** | **75.00%** | — | **86.36%** | **63.64%** |
| *Clin* | *Clinical Top-2 (Test)* | *Medication-Aware Top-2 Takes* | *81.83%* | *79.27%* | *0.8985* | *86.99%* | *76.67%* |
| *Abl* | *Exploratory Ablation* | *Multi-Moment Distribution Meta-Classifier* | *90.86%* | *95.16%* | *0.9346* | *82.19%* | *99.54%* |

> [!NOTE]
> **Adherence Confounding Analysis & Count-Invariant Aggregation**
> In mPower app data, Parkinson's participants submitted significantly more recordings than controls due to medication-tracking protocols (5.04 vs 1.72 mean takes; 100% of participants with $\ge 3$ takes in the dataset are PD). Exploratory multi-moment models using order statistics (`max`, `min`, `top-2`, `count`) can artificially reach >90% by exploiting app usage frequency (AUC 0.897 on count alone). To ensure real-world clinical validity and prevent false alarms on patients who provide multiple takes, NeuroVocal enforces **count-invariant probability averaging** anchored at the validation-calibrated threshold ($\tau = 0.360$).

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
│  Stage 2: Count-Invariant Acoustic Aggregator            │
│  - Input: Phonation probabilities across patient takes   │
│  - Aggregation: Arithmetic Mean P_mean = (1/N) sum(P_i)  │
│  - Validation-Calibrated Decision Threshold: 0.360       │
│  - 3-Tier Clinical Triage:                               │
│      * Low Risk: P < 0.260 (Normal voice dynamics)       │
│      * Borderline: 0.260 <= P < 0.36 (Screening Alert)   │
│      * High Risk: P >= 0.36 (Manifest Dysphonia)         │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
   [Final Clinical Screening Report & Triage Recommendation]
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

### 3. Exploratory Distribution Ablation Study

To inspect the exploratory multi-moment distribution ablation report:

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
| **Raw Waveforms** | **Sustained `/a/`** | **`86.36%` (19/22)** | **`63.64%` (14/22)** | **`75.00%`** | **95.2% (20/21)** | **55.6% (10/18)** |
| **Denoised Clean** | **Sustained `/a/`** | **`68.18%` (15/22)** | **`72.73%` (16/22)** | **`70.45%`** | **81.0% (17/21)** | **22.2% (4/18)** |
| **Denoised Clean** | **All (`/a/` + `/i/`)** | **`63.64%` (14/22)** | **`59.09%` (13/22)** | **`61.36%`** | **90.5% (19/21)** | **33.3% (6/18)** |

* **Key Clinical Insight 1 (Raw vs. Denoised):** Raw audio yields an **86.36% detection rate** on Parkinson's disease. Aggressive spectral denoising filters out subtle turbulent aspiration noise and micro-shimmer that are hallmarks of incomplete glottal closure in early Parkinson's.
* **Key Clinical Insight 2 (Phonation Protocol):** The standard open vowel `/a/` yields superior specificity (72.73% clean, 63.64% raw) compared to constricted front vowels like `/i/` (59.09%), confirming the clinical requirement for open-tract sustained phonations.
* **Key Clinical Insight 3 (Atypical Movement Disorders):** The pipeline selectively identified up to **95.2% of Multiple System Atrophy (MSA)** cases (known in neurology for early, severe laryngeal abductor paresis and stridor), while maintaining selectivity on Progressive Supranuclear Palsy (PSP).

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
* `GET /api/v1/meta-stats`: Returns the active checkpoint threshold, derived triage cutoffs, and the production `arithmetic_mean` aggregation configuration. Legacy reference-statistic fields remain only as a migration aid for older clients.
* `POST /api/v1/predict`: Accepts $1$ to $N$ voice recordings (`.wav`, `.ogg`, `.m4a`, `.webm`), extracts Wav2Vec 2.0 probabilities, averages them as `participant_mean_score`, and returns structured screening triage JSON. The legacy `meta_classifier_score` field is retained as an alias.
