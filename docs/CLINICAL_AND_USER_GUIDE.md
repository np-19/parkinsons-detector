# User & System Operation Guide (NeuroVocal)

> **Student Minor Project Notice**: This software was developed as an undergraduate engineering minor project for acoustic voice feature analysis. It is an educational and exploratory tool, not a certified clinical diagnostic device.

---

## 1. Executive Summary & Evaluation Performance

NeuroVocal detects early acoustic signs of Parkinson's Disease from short audio recordings of sustained phonation (saying *"aaaaah"* steadily at conversational volume). 

The system was evaluated on the held-out test split of the **mPower Mobile Parkinson Study** (579 participants, 1,480 audio recordings) under strict participant-level separation (zero data leakage between train and test sets):

| Tier | Architecture | Aggregation Method | Balanced Accuracy | Raw Accuracy | ROC-AUC | Sensitivity | Specificity | Description |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Wav2Vec 2.0 Base** | Arithmetic Mean ($\tau = 0.360$) | **74.72%** | 73.06% | 0.8245 | 78.08% | 71.36% | Baseline speech representation model. |
| **1** | **WaveLM Base+** | Arithmetic Mean ($\tau = 0.310$) | **79.11%** | 79.27% | **0.8585** | 78.77% | 79.45% | Masked speech denoising foundation backbone. |
| **2** | **WaveLM Base+** | **Attention-MIL ($\tau = 0.300$)** | **`82.54%`** | **`82.04%`** | **`0.8791`** | **`83.56%`** | **`81.52%`** | **Production Pipeline**: Gated attention bag pooling with zero test leakage. |

---

## 2. Voice Analysis Rationale & Triage Zones

### A. How Parkinson's Affects the Voice
In Parkinson's Disease, motor control degradation affects the vocal cords and larynx muscles:
- **Vocal Tremor (4–7 Hz)**: Involuntary modulation in pitch and amplitude.
- **Acoustic Micro-Breaks**: Brief interruptions or unsteady phonation while trying to hold a pitch.
- **Hypophonia & Breathiness**: Reduced vocal intensity and air leakage through the glottis.

Asking individuals to sustain an open vowel (such as *"aaaaah"*) for 8 to 10 seconds isolates vocal cord stability without confounding factors from language, accent, grammar, or vocabulary.

### B. The 3-Tier Triage Strategy
Because human voice fluctuates naturally and fatigue or background noise can occur, NeuroVocal classifies scores into three runtime triage tiers:

![Confidence Score Distribution](figures/clinical_triage_distribution.png)

1. **Low Risk ($P < 0.20$)**: Normal voice stability. In the held-out test predictions, 351/579 participants (60.6%) fell here, including **329/433 controls (76.0%)**.
2. **Borderline / Retest ($0.20 \le P < 0.30$)**: Indeterminate or subtle acoustic variation (26/579, 4.5%). The system flags these cases for a quiet re-test.
3. **High Risk ($P \ge 0.30$)**: Clear acoustic instability and phonatory tremor. This band contained **122/146 PD participants (83.6%)**; a medical follow-up is recommended.

Across the **553/579 participants (95.5%)** in a decisive band ($P < 0.20$ or $P \ge 0.30$), the post-hoc balanced accuracy is **82.58%**. The separate `0.40` value is the *high-confidence marker*: it changes the confidence wording (`High Confidence Positive`) but is not the point where high risk begins.

> These band statistics are descriptive counts from `artifacts/models/deep_learning/wavelm/test_predictions.csv`; they are not a separately validated clinical triage accuracy.

### C. Count-Invariance: Preventing Dataset Adherence Bias
In real-world mobile app datasets like mPower, participants with Parkinson's submitted significantly more voice samples on average than healthy volunteers (**5.04 takes vs 1.72 takes** in the held-out test split).

If a model uses distribution heuristics like `max()`, `min()`, or recording count, it artificially learns how frequently the app was used rather than genuine acoustic features. NeuroVocal enforces strict **Count-Invariance** via Gated Attention-MIL ($\sum \alpha_i = 1.0$), ensuring identical mathematical treatment whether a user submits 1 take or 5 takes.

---

## 3. Audio Recording Guidelines

To get accurate and consistent results, follow these simple recording steps:

### Step 1: Environment & Microphone Setup
1. **Quiet Room**: Record in a quiet indoor room away from loud fans, television, or open windows.
2. **Microphone Distance**: Hold the phone or microphone **15 to 20 cm (6 to 8 inches)** from the mouth.
3. **Microphone Angle**: Tilt the microphone slightly at a **45-degree angle** rather than blowing directly into it to avoid wind noise.

### Step 2: Voice Delivery
1. Take a deep breath.
2. Say *"Aaah"* continuously at your natural conversational pitch and loudness.
3. Keep the sound as steady and even as possible until running out of breath.
4. **Duration**: Sustain the vowel for **at least 8 seconds**. Audio under 1.0s or pure silence is automatically rejected.

### Step 3: Multiple Takes (Recommended)
Providing **2 to 3 takes** recorded a few moments apart helps the Attention-MIL pooling head filter out transient throat clearing or ambient noises.

---

## 4. Running Predictions (CLI & Python API)

### A. Command-Line Interface (CLI)

Run inference on one or more WAV recordings using [`src/deep_learning/predict_patient.py`](../src/deep_learning/predict_patient.py):

```bash
# Evaluate a participant with multiple recordings using WaveLM + Attention-MIL
python3 src/deep_learning/predict_patient.py \
  data/processed_audio/test/PD/100009.wav \
  data/processed_audio/test/PD/100136.wav
```

**Terminal Output:**
```text
>>> Loading single speech model from: artifacts/models/deep_learning/wavelm/best_model.pt (on cuda)...
>>> [ATTENTION-MIL] Gated Attention Aggregation Loaded: artifacts/models/deep_learning/wavelm/mil_head.pt
    - Calibrated Operating Threshold: 0.300
    - Calibrated Recording Threshold: 0.310

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
  Triage Recommendation: Acoustic dysphonia and vocal stability degradation detected above decision threshold. Specialist neurological evaluation advised.
-----------------------------------------------------------------
  Statistical Distribution:
    - Mean Probability: 0.9302
    - Top-2 Probability:0.9302
    - Max Probability:  0.9692
    - Std Deviation:    0.0390
-----------------------------------------------------------------
  Recording Details:
    [1] 100009.wav (10.05s): P(PD) = 0.8913 [Attn: 0.381] -> Dysarthric / Tremor
    [2] 100136.wav (10.05s): P(PD) = 0.9692 [Attn: 0.619] -> Dysarthric / Tremor
=================================================================
```

> These values are the actual output for these two held-out test takes (recorded on 2026-10-21). Exact scores depend on the input audio. `HIGH_RISK` starts at the participant threshold (0.300); the `Diagnostic: ≥0.400` text is the confidence marker.

For JSON output:
```bash
python3 src/deep_learning/predict_patient.py sample1.wav sample2.wav --json
```

### B. Python API Integration

```python
from pathlib import Path
from src.deep_learning.predict_patient import PatientPredictor

# Initialize the production pipeline
predictor = PatientPredictor(
    model_path="artifacts/models/deep_learning/wavelm/best_model.pt",
    mil_model_path="artifacts/models/deep_learning/wavelm/mil_head.pt",
    use_attention_mil=True,
    device="cuda",  # or "cpu"
)

# Run prediction
audio_files = ["data/processed_audio/test/PD/100009.wav", "data/processed_audio/test/PD/100136.wav"]
result = predictor.predict_patient(audio_files)

print("Diagnosis:          ", result["participant_verdict"])
print("Triage Level:       ", result["triage_level"])
print("Score:              ", result["participant_mean_score"])
print("Aggregation Method: ", result["aggregation_method"])
for rec in result["recordings_detail"]:
    print(f"  - {Path(rec['file']).name}: P={rec['probability_PD']}, Attention={rec['attention_weight']}")
```

---

## 5. Web API Reference (FastAPI Backend)

Start the local backend server:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### POST `/api/v1/predict`
Accepts one or more audio files (`.wav`, `.mp3`, `.mpeg`, `.ogg`, `.m4a`, `.aac`, `.flac`, `.webm`) via multipart form data (`files`).

**Sample JSON Response** (abridged actual response for the two takes above):
```json
{
  "diagnosis": "POSITIVE (Acoustic Dysphonia Detected)",
  "is_pd": true,
  "triage_level": "HIGH_RISK",
  "triage_recommendation": "Acoustic dysphonia and vocal stability degradation detected above decision threshold. Specialist neurological evaluation advised.",
  "confidence_level": "High Confidence Positive",
  "participant_mean_score": 0.9695,
  "meta_classifier_score": 0.9695,
  "threshold": 0.30000000000000004,
  "screening_cutoff": 0.20000000000000004,
  "diagnostic_cutoff": 0.4,
  "recordings_analyzed": 2,
  "distribution": {
    "mean_probability": 0.9302,
    "top2_probability": 0.9302,
    "max_probability": 0.9692,
    "min_probability": 0.8913,
    "std_deviation": 0.039,
    "recordings_analyzed": 2
  },
  "recordings": [
    {
      "filename": "100009.wav",
      "duration_seconds": 10.05,
      "probability_pd": 0.8913,
      "recording_status": "Dysarthric / Tremor",
      "attention_weight": 0.3812
    },
    {
      "filename": "100136.wav",
      "duration_seconds": 10.05,
      "probability_pd": 0.9692,
      "recording_status": "Dysarthric / Tremor",
      "attention_weight": 0.6188
    }
  ],
  "failed_recordings": [],
  "clinical_warnings": [],
  "timestamp": "2026-10-21T09:00:26.468186+00:00",
  "aggregation_method": "gated_attention_mil",
  "attention_weights": [0.0035, 0.1956, 0.1821, 0.1844, 0.218, 0.2164]
}
```

> `meta_classifier_score` is a deprecated compatibility alias for `participant_mean_score`; new clients should read `participant_mean_score`. The `diagnostic_cutoff` is a confidence marker, not the high-risk decision boundary (which is `threshold = 0.300`).

### GET `/api/v1/meta-stats`
Returns system decision metadata and operating thresholds:
```json
{
  "threshold": 0.30,
  "screening_cutoff": 0.20,
  "diagnostic_cutoff": 0.40,
  "aggregation": "gated_attention_mil",
  "score_field": "participant_mean_score",
  "model_version": "WaveLM + Gated Attention-MIL (mPower)"
}
```
