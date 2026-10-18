# NeuroVocal: Voice-Based Parkinson's Detection (Minor Project)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning minor project for detecting Parkinson's Disease from sustained vowel audio recordings (`/aaa/`). Built on a fine-tuned **Microsoft WaveLM Base+** speech foundation model combined with **Gated Attention Multiple Instance Learning (Attention-MIL)** pooling. 

Evaluated with **100% strict participant-level separation** (zero participant overlap between training, validation, and test splits).

---

## Key Results (mPower Held-Out Test Set)

Evaluated on **579 unseen test participants** (146 Parkinson's, 433 Healthy Controls; 1,480 total audio recordings):

| Model Architecture | Pooling Method | Balanced Accuracy | Raw Accuracy | Sensitivity | Specificity | ROC-AUC | Optimal Cutoff ($\tau$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wav2Vec 2.0 Base** | Arithmetic Mean | 74.72% | 73.06% | 78.08% | 71.36% | 0.8245 | 0.360 |
| **WaveLM Base+** | Arithmetic Mean (Baseline) | 79.11% | 79.27% | 78.77% | 79.45% | 0.8585 | 0.310 |
| **WaveLM Base+ (Proposed)** | **Gated Attention-MIL** | **`81.86%`** | **`82.04%`** | **`81.51%`** | **`82.22%`** | **`0.8728`** | **`0.300`** |

* **Held-Out Test Balanced Accuracy**: **`81.86%`** (95% Bootstrap CI: `[78.1%, 85.4%]`, Threshold $\tau = 0.300$)
* **Parkinson's Detection Sensitivity**: **`81.51%`** (119 / 146 PD participants correctly detected)
* **Healthy Control Specificity**: **`82.22%`** (356 / 433 healthy controls correctly cleared)
* **Test ROC-AUC**: **`0.8728`** (95% CI: `[0.845, 0.909]`)
* **Single Model Efficiency**: Fine-tuned WaveLM Base+ (90M parameters, frozen during MIL inference); runs in **<1 second** per recording on an NVIDIA GTX 1650.

---

## Evaluation Figures

### 1. ROC Curve
Shows model discrimination on the 579 test participants. WaveLM + Gated Attention-MIL operates at **81.51% Sensitivity** and **82.22% Specificity** ($\tau = 0.300$, AUC = 0.8728).

![Receiver Operating Characteristic (ROC) Curve](docs/figures/roc_curves.png)

---

### 2. Confusion Matrices (Baseline vs. Proposed Model)
Side-by-side comparison evaluated on the identical held-out test cohort of 579 participants (433 Healthy, 146 Parkinson's):

![Confusion Matrices](docs/figures/confusion_matrices.png)

* **Cleared False Alarms**: Attention-MIL reduced false alarms from 89 down to 77 (clearing 12 healthy individuals falsely flagged by arithmetic averaging).
* **Rescued True Positives**: Correctly identified 119 of 146 Parkinson's patients (up from 115, catching 4 additional cases missed by standard averaging).
* **Balanced Accuracy Gain**: Improved test balanced accuracy from **79.11% to 81.86%** (+2.76% gain, $\tau = 0.300$).

---

### 3. Prediction Score Distribution
Distribution of model probabilities for Healthy Controls vs. Parkinson's participants, showing low-risk ($P < 0.20$), retest ($0.20 \le P < 0.50$), and high-risk ($P \ge 0.50$) confidence zones:

![Prediction Score Distribution](docs/figures/clinical_triage_distribution.png)

---

## How It Works (Pipeline Architecture)

```
[Raw Voice Recording: 16 kHz Mono WAV/M4A]
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Audio Pre-Processing                              │
│  - Convert/resample audio to 16 kHz Mono float32            │
│  - Peak amplitude normalization to [-1.0, 1.0]              │
│  - Silence and minimum duration validation (>= 1.0s)        │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Multi-Crop Sliding Window                         │
│  - Slices 8.0s overlapping crops across sustained phonation │
│  - Avoids initial onset hesitation / vocal fry              │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼ Tensor of Crops (N_crops, 128000)
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Fine-Tuned WaveLM Feature Backbone (90M Params)   │
│  - Microsoft WaveLM Base+ (Pretrained on 94k hrs of speech) │
│  - 7-Layer CNN Feature Encoder (Frozen)                     │
│  - Transformer Layers 1–4 (Frozen General Acoustic Phonetics)│
│  - Transformer Layers 5–12 (Fine-tuned Dysphonia Attention) │
│  - Temporal Attention Pooling (TAP)                         │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼ Embeddings Tensor (N_crops, 768)
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Gated Attention-MIL Head (mil_head.pt)            │
│  - Two-branch gating: a_i = w^T (tanh(V h) * sigmoid(U h))  │
│  - Computes attention weights alpha_i summing to 1.0        │
│  - Dynamically downweights uninformative transient noise    │
│  - Operating Decision Threshold: tau = 0.300                │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Output Verdict & Score                            │
│  - Predicted Class: POSITIVE (Parkinson's) or NEGATIVE      │
│  - Confidence Level & Per-Recording Attention Breakdown     │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset & Split Isolation

* **Dataset**: mPower Mobile Parkinson Study (Sage Bionetworks), containing **9,986 audio recordings across 3,868 participants**.
* **Strict Participant Splitting**:
  - **Train**: 2,707 participants (7,006 recordings)
  - **Validation**: 580 participants (1,510 recordings)
  - **Held-Out Test**: 579 participants (1,480 recordings)
  - **Zero Leakage**: No participant in the test set appears in training or validation.

---

## Quick Start & Usage

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/np-19/parkinsons-detector.git
cd parkinsons-detector

# Create and activate Python virtual environment
python3 -m venv data/.venv
source data/.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Predict a Patient's Voice (CLI)

Pass one or more voice recordings (`.wav`, `.m4a`, `.mp3`) to get a prediction:

```bash
python3 src/deep_learning/predict_patient.py \
  data/processed_audio/test/PD/100009.wav \
  data/processed_audio/test/PD/100136.wav
```

**Example Output:**
```text
>>> Loading single speech model from: artifacts/models/deep_learning/wavelm/best_model.pt (on cuda)...
>>> [ATTENTION-MIL] Gated Attention Aggregation Loaded: artifacts/models/deep_learning/wavelm/mil_head.pt
    - Calibrated Operating Threshold: 0.300

=================================================================
      PARKINSON'S ACOUSTIC SCREENING REPORT
=================================================================
  Final Diagnosis:      POSITIVE (Acoustic Dysphonia Detected)
  Clinical Triage:      HIGH_RISK (Screening: <0.200 | Diagnostic: ≥0.400)
  Confidence Level:     High Confidence Positive
  Aggregation Method:   gated_attention_mil
  Participant Score:    0.9695 (Decision Threshold: 0.300)
  Recordings Analyzed:  2
  Recordings Failed:    0
-----------------------------------------------------------------
  Recording Details:
    [1] 100009.wav (10.05s): P(PD) = 0.8913 [Attn: 0.381] -> Dysarthric / Tremor
    [2] 100136.wav (10.05s): P(PD) = 0.9692 [Attn: 0.619] -> Dysarthric / Tremor
=================================================================
```

For JSON output:
```bash
python3 src/deep_learning/predict_patient.py sample.wav --json
```

### 3. Run Web App Backend & Frontend

Start the FastAPI backend:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs are available at `http://localhost:8000/docs`.

Start the React dashboard:
```bash
cd frontend && npm install && npm run dev
```

### 4. Run Unit Tests

```bash
python3 -m pytest tests/
```

---

## Project Documentation

* **[Viva & Defense Guide](docs/PRESENTATION_AND_DEFENSE_GUIDE.md)**: Frequently asked viva/evaluator questions with simple, direct answers.
* **[User Guide](docs/CLINICAL_AND_USER_GUIDE.md)**: Operating steps, recording guidelines, and Python API usage.
* **[Architecture Decisions Record (ADR)](docs/DECISIONS.md)**: Log of technical design decisions (ADR-001 through ADR-013).
* **[Changelog](docs/CHANGELOG.md)**: Project version history.

---

## References

1. **Ilse, M., Tomczak, J., & Welling, M.** (2018). *Attention-based Deep Multiple Instance Learning*. ICML.
2. **Chen, S., et al.** (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*. IEEE JSTSP.
3. **Trister, A. D., et al.** (2016). *The mPower study, Parkinson disease mobile data collected using ResearchKit*. Nature Scientific Data.
