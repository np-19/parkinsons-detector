# Clinical Rationale, Operating Procedure & User Guide

> **Medical Disclaimer**: This software is designed strictly as a digital acoustic screening biomarker tool and clinical decision support system. It is not an automated diagnostic device and must be evaluated alongside comprehensive clinical neurological assessments (e.g., MDS-UPDRS Part III).

---

## 1. Executive Summary & Clinical Performance

Our Parkinson's Disease detection pipeline operates on raw voice waveforms (sustained phonation of the vowel `/a/`). Evaluated on the held-out test split of the **mPower Mobile Parkinson Study** (579 participants, 1,480 recordings, 100% strict participant separation with zero data leakage):

| Tier | Evaluation Level | Method / Strategy | Balanced Accuracy | Raw Accuracy | ROC-AUC | Clinical Role |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **0** | **Recording-Level** | Single-crop Wav2Vec 2.0 Acoustic Model | **71.96%** | 71.96% | 0.7974 | Baseline perceptual acoustic feature extraction. |
| **1** | **Participant Baseline** | Naive Arithmetic Mean across all recordings | **76.68%** | 75.30% | 0.8322 | Standard ML aggregation; vulnerable to pharmacological dilution. |
| **2** | **Clinical Top-2** | Mean of the 2 most symptomatic recordings | **82.30%** | 81.35% | **0.8995** | Overcomes transient medication masking without complex models. |
| **3** | **Meta-Classifier** | Multi-Moment Distribution Modeling (`[mean, max, min, top-2, p75, std, n]`) | **`91.20%`** | **`94.99%`** | **`0.9338`** | **Production Pipeline**: Captures medication variance. **98.85% specificity** (only 5 false alarms out of 433 controls). |

---

## 2. Fine Clinical Details: Why Naive Mean Fails & Why Meta-Classifier Succeeds

### A. The Pharmacological Dilemma: Levodopa Masking (The "ON" / "OFF" State)
Parkinson's Disease is characterized by basal ganglia dopamine depletion, causing laryngeal rigidity, hypophonia (reduced volume), and phonatory micro-tremor (4–7 Hz).

Patients manage symptoms using dopaminergic replacement therapy (principally **Levodopa / Carbidopa**):
- **"ON" State (Peak Medication, ~1–3 hours post-dose)**: Vocal tremor and rigidity temporarily diminish; phonation temporarily resembles healthy speech.
- **"OFF" State (Pre-Dose / End-of-Dose Wearing Off)**: True motor dysarthria and vocal instabilities emerge prominently.

### B. The Mathematical Dilution Problem
When a patient submits multiple recordings over several days, some recordings inevitably occur during the "ON" state. 

Consider a true Parkinson's patient who submits 4 recordings:
- Recording 1 ("ON" state): $P(\text{PD}) = 0.15$
- Recording 2 ("ON" state): $P(\text{PD}) = 0.18$
- Recording 3 ("ON" state): $P(\text{PD}) = 0.22$
- Recording 4 ("OFF" state, unmedicated tremor): $P(\text{PD}) = 0.85$

Under standard arithmetic averaging (`mean()`):
$$\text{Score}_{\text{patient}} = \frac{0.15 + 0.18 + 0.22 + 0.85}{4} = \frac{1.40}{4} = 0.35$$

At the optimal validation decision threshold ($\tau = 0.360$), **this patient is classified as 0 (Healthy Control)** — a False Negative! The three medicated recordings diluted the true pathological signal.

### C. The Solution: Statistical Distribution Modeling
Healthy individuals **never** produce dysarthric vocal tremor or laryngeal rigidity. Their probability distribution across time is tightly bounded at the bottom ($\text{low mean}, \text{low max}, \text{near-zero std dev}$).

Parkinson's patients, by contrast, exhibit **high temporal volatility ($\sigma$)** and **elevated peak scores ($\text{max}, \text{top-2}$)** due to the medication cycle.

The **Participant Meta-Classifier** evaluates the complete feature vector:
$$\mathbf{x} = [\mu, \max, \min, \text{top-2 mean}, Q_{75}, \sigma, \ln(N+1)]$$

This statistical boundary achieves **91.20% Balanced Accuracy** and **98.85% Specificity** (only 5 false alarms out of 433 healthy controls).

---

## 3. Architecture Ablation: Model 1 vs Model 2

| Property | Model 1 (`wav2vec2_finetuned`) | Model 2 (`wav2vec2_v2`) | Clinical Rationale for Model 2 |
| :--- | :--- | :--- | :--- |
| **Audio Window** | 5.0 seconds (80,000 samples) | **8.0 seconds (128,000 samples)** | Captures 30–50 complete vocal tremor cycles (4–7 Hz). |
| **Frozen Layers** | 6 Transformer layers frozen | **4 layers frozen (8 layers unfrozen)** | Allows deep self-attention layers to specialize in dysphonia. |
| **Pooling Method** | Temporal Attention Pooling | **Temporal Attention Pooling** | Dynamically focuses on acoustic tremor bursts. |
| **Regularization** | Dropout 0.1 | **Mixup ($\alpha=0.2$), Dropout 0.3, Label Smoothing 0.1** | Prevents overfitting to acoustic background noise. |
| **Test Balanced Acc** | 67.50% | **76.68% (Raw) / 91.20% (Meta-Classifier)** | 23.7% absolute accuracy gain. |
| **Role in Project** | Baseline Ablation Reference | **Production Screening Engine** | Model 1 serves as the scientific comparison. |

> [!NOTE]
> **Do we need Model 1 for Model 2 to run?**
> **No.** Model 2 was trained independently from Hugging Face pretrained weights (`facebook/wav2vec2-base`). Model 2 does not require Model 1. Model 1 is retained solely for scientific documentation and defense presentation.

---

## 4. Standard Operating Procedure (SOP): How to Best Use This App

To achieve maximum clinical diagnostic accuracy (>91%) in real-world deployment, follow this Standard Operating Procedure:

### Step 1: Acoustic Environment & Hardware Placement
1. **Quiet Environment**: Test in an enclosed room with ambient noise $<45\text{ dBA}$ (avoid air conditioners, open windows, or background speech).
2. **Device Distance**: Hold the smartphone or microphone **15 to 20 cm (6 to 8 inches)** from the mouth.
3. **Microphone Angle**: Position the microphone at a **45-degree angle** to the airflow (not directly in front of the mouth) to eliminate aerodynamic plosive wind noise.

### Step 2: Patient Phonation Instructions
1. Instruct the patient:
   > *"Take a deep breath, and say 'Aaah' as steadily and continuously as you can at your natural speaking pitch and volume until you run out of breath."*
2. **Duration**: The recording must be sustained for **at least 8 to 10 seconds**. (The model analyzes an 8.0-second window; recordings shorter than 8s will be automatically zero-padded).
3. **Pitch & Loudness**: Avoid singing, pitch glides, or shouting. The phonation should be steady.

### Step 3: The Multi-Session Sampling Protocol (The Golden Rule)
- **Do not rely on a single recording.** Error analysis proved that **62.5% of false negatives occurred when only 1 recording was submitted**, due to Levodopa medication masking.
- **Protocol**: Capture **2 to 3 recordings across 24–48 hours**:
  - **Sample 1 (Morning "OFF" state)**: First thing in the morning, **before** taking the morning dose of Levodopa.
  - **Sample 2 (Peak "ON" state)**: 1.5 to 2 hours after medication.
  - **Sample 3 (Evening Wearing-Off)**: Before the evening dose.
- When $\ge 2$ recordings are analyzed, the system's sensitivity jumps to **`91.6%`**.

---

## 5. Running Patient Inference (CLI & Python API)

### A. Command-Line Interface (CLI)

Run inference on one or more audio files for a patient using [`src/deep_learning/predict_patient.py`](file:///home/neeleshparihar/parkinsons-detector/src/deep_learning/predict_patient.py):

```bash
# Evaluate a patient with multiple recordings
python3 src/deep_learning/predict_patient.py \
  data/processed_audio/patient_sample1.wav \
  data/processed_audio/patient_sample2.wav \
  data/processed_audio/patient_sample3.wav
```

**Example Terminal Output:**
```text
=================================================================
      PARKINSON'S ACOUSTIC SCREENING REPORT
=================================================================
  Final Diagnosis:      POSITIVE (Parkinsonian Dysphonia Detected)
  Confidence Level:     High Confidence Positive
  Meta-Classifier Score:0.8741 (Threshold: 0.29)
  Recordings Analyzed:  3
-----------------------------------------------------------------
  Statistical Distribution:
    - Mean Probability: 0.5412
    - Top-2 Probability:0.7380
    - Max Probability:  0.8845
    - Std Deviation:    0.2891
-----------------------------------------------------------------
  Recording Details:
    [1] patient_sample1.wav (8.2s): P(PD) = 0.8845 -> Dysarthric / Tremor
    [2] patient_sample2.wav (9.1s): P(PD) = 0.1915 -> Normal Phonation
    [3] patient_sample3.wav (8.0s): P(PD) = 0.5916 -> Dysarthric / Tremor
-----------------------------------------------------------------
  Clinical Observations / Warnings:
    * High variance between recordings detected (std dev > 0.25). 
      This is strongly indicative of cyclical ON/OFF Levodopa medication fluctuations.
=================================================================
```

For JSON output (for integration into web apps, mobile apps, or electronic health records):
```bash
python3 src/deep_learning/predict_patient.py patient_sample1.wav --json
```

### B. Python API Integration

```python
from src.deep_learning.predict_patient import PatientPredictor

# Initialize the pipeline (loads Model 2 + Meta-Classifier)
predictor = PatientPredictor(
    model_path="artifacts/models/deep_learning/wav2vec2_v2/best_model.pt",
    device="cuda",  # or "cpu"
)

# Predict for a patient
audio_files = ["sample_morning.wav", "sample_evening.wav"]
result = predictor.predict_patient(audio_files)

print("Diagnosis:", result["participant_verdict"])
print("Meta-Score:", result["meta_classifier_score"])
print("Warnings:", result["clinical_warnings"])
```

---

## 6. How to Run Full Held-Out Test Evaluation

To reproduce the complete benchmark on all 1,480 held-out test recordings:

```bash
# Using Docker (pure environment reproducibility)
bash scripts/run_docker_evaluate.sh artifacts/models/deep_learning/wav2vec2_v2 5 8.0

# Or using the local Python environment
python3 src/deep_learning/meta_classifier.py --v2-predictions artifacts/models/deep_learning/wav2vec2_v2/test_predictions.csv
```

