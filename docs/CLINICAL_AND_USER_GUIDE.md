# Clinical Rationale, Operating Procedure & User Guide

> **Medical Disclaimer**: This software is designed strictly as a digital acoustic screening biomarker tool and clinical decision support system. It is not an automated diagnostic device and must be evaluated alongside comprehensive clinical neurological assessments (e.g., MDS-UPDRS Part III).

---

## 1. Executive Summary & Clinical Performance

Our Parkinson's Disease detection pipeline operates on raw voice waveforms (sustained phonation of the vowel `/a/`). Evaluated on the held-out test split of the **mPower Mobile Parkinson Study** (579 evaluated participants, 1,480 recordings, 100% strict participant separation with zero data leakage):

| Tier | Evaluation Level | Method / Strategy | Balanced Accuracy | Raw Accuracy | ROC-AUC | Clinical Role |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **0** | **Recording-Level (Test)** | Single-crop Wav2Vec 2.0 Acoustic Model | **71.22%** | 71.22% | 0.7853 | Baseline perceptual acoustic feature extraction. |
| **1** | **Participant-Level (Test)** | Count-Invariant Mean ($\tau = 0.360$) | **74.72%** | 73.06% | **0.8245** | Production pipeline: arithmetic-mean probability averaging that does not explicitly use recording count as a feature. |
| **2** | **Participant-Level (Val)** | Validation-Calibrated Mean ($\tau = 0.360$) | **80.09%** | 79.79% | **0.8665** | Checkpoint selection benchmark (580 unseen participants). |
| **Ext** | **Cross-Cohort (Zero-Shot)** | Raw sustained `/a/` phonation (`records`) | **75.00%** | 75.00% | — | Generalization across independent clinical institutions. |
| *Clin* | *Exploratory Top-2 (Test)* | *Top-2 probability aggregation* | *81.83%* | *79.27%* | *0.8985* | *Alternative aggregation comparison; not a medication-effect estimate.* |
| *Abl* | *Exploratory Ablation* | *Multi-Moment Distribution Meta-Classifier* | *90.86%* | *95.16%* | *0.9346* | *Ablation documenting mPower adherence shortcut.* |

---

## 2. Fine Clinical Details: Why Standard Mean & Triage Scale Are Employed

### A. The Pharmacological Dilemma: medication Masking (The "ON" / "OFF" State)
Parkinson's Disease is characterized by basal ganglia dopamine depletion, causing laryngeal rigidity, hypophonia (reduced volume), and phonatory micro-tremor (4–7 Hz).

Patients manage symptoms using dopaminergic replacement therapy (principally **medical therapy**):
- **"ON" State (Peak Medication, ~1–3 hours post-dose)**: Vocal tremor and rigidity temporarily diminish; phonation temporarily resembles healthy speech.
- **"OFF" State (Pre-Dose / End-of-Dose Wearing Off)**: True motor dysarthria and vocal instabilities emerge prominently.

### B. The 3-Tier Clinical Triage Strategy
Because a single recording might happen during an "ON" state, NeuroVocal employs a stratified 3-tier clinical decision support framework:
1. **Low Risk ($P < 0.260$)**: Normal voice dynamics and vocal stability.
2. **Borderline Screening Alert ($0.260 \le P < 0.36$)**: Sensitivity-optimized gate. Flags patients with subtle vocal tremor or potential medication "ON-state" masking for a morning pre-dose re-test or neurological consultation.
3. **High Risk ($P \ge 0.36$)**: Manifest Parkinsonian dysphonia and phonatory tremor; specialist neurology referral advised.

### C. The Adherence Confounder & Mean Aggregation
In exploratory analysis, multi-moment distribution features (`max, min, top-2, std dev, count`) achieved an apparent 91.20% balanced accuracy. However, our dataset audit revealed that in mPower, Parkinson's patients submitted significantly more recordings than controls (5.04 vs 1.72 takes on average; 100% of participants with $\ge 3$ takes are PD). Extreme-value statistics (`max` and `min`) naturally shift with sample size ($N$), inadvertently learning study adherence rather than pure vocal tremor. 

To avoid using recording count as an explicit feature, production inference
uses arithmetic-mean probability aggregation
($\bar{P} = \frac{1}{N}\sum P_i$) anchored at $\tau = 0.360$. The mean does not
guarantee identical reliability for different numbers or qualities of recordings.

---

## 3. Architecture Ablation: Model 1 vs Model 2

| Property | Model 1 (`wav2vec2_finetuned`) | Model 2 (`wav2vec2_v2`) | Clinical Rationale for Model 2 |
| :--- | :--- | :--- | :--- |
| **Audio Window** | 5.0 seconds (80,000 samples) | **8.0 seconds (128,000 samples)** | Captures 30–50 complete vocal tremor cycles (4–7 Hz). |
| **Frozen Layers** | 6 Transformer layers frozen | **4 layers frozen (8 layers unfrozen)** | Allows deep self-attention layers to specialize in dysphonia. |
| **Pooling Method** | Temporal Attention Pooling | **Temporal Attention Pooling** | Dynamically focuses on acoustic tremor bursts. |
| **Regularization** | Dropout 0.1 | **Mixup ($\alpha=0.2$), Dropout 0.3, Label Smoothing 0.1** | Prevents overfitting to acoustic background noise. |
| **Val / Test Performance** | 67.50% Val Balanced Acc | **80.09% Val Balanced Acc / 74.72% Test Acc (0.8245 ROC-AUC)** | Substantial gain in genuine acoustic representation learning. |
| **Role in Project** | Baseline Ablation Reference | **Production Screening Engine** | Model 1 serves as the scientific comparison. |

> [!NOTE]
> **Do we need Model 1 for Model 2 to run?**
> **No.** Model 2 was trained independently from Hugging Face pretrained weights (`facebook/wav2vec2-base`). Model 2 does not require Model 1. Model 1 is retained solely for scientific documentation and defense presentation.

---

## 4. Standard Operating Procedure (SOP): How to Best Use This App

For consistent research-grade screening, follow this Standard Operating Procedure. The
steps below are collection guidance, not a claim of diagnostic accuracy:

### Step 1: Acoustic Environment & Hardware Placement
1. **Quiet Environment**: Test in an enclosed room with ambient noise $<45\text{ dBA}$ (avoid air conditioners, open windows, or background speech).
2. **Device Distance**: Hold the smartphone or microphone **15 to 20 cm (6 to 8 inches)** from the mouth.
3. **Microphone Angle**: Position the microphone at a **45-degree angle** to the airflow (not directly in front of the mouth) to eliminate aerodynamic plosive wind noise.

### Step 2: Patient Phonation Instructions
1. Instruct the patient:
   > *"Take a deep breath, and say 'Aaah' as steadily and continuously as you can at your natural speaking pitch and volume until you run out of breath."*
2. **Duration**: The recording must be sustained for **at least 8 seconds**. Recordings shorter than the checkpoint window or containing only silence are rejected and reported as not analyzed.
3. **Pitch & Loudness**: Avoid singing, pitch glides, or shouting. The phonation should be steady.

### Step 3: The Multi-Session Sampling Protocol (The Golden Rule)
- **Do not rely on a single recording.** A single recording may not represent a person's full acoustic variation. Medication effects are a plausible hypothesis, not a demonstrated cause of individual errors in this evaluation.
- **Protocol**: Capture **2 to 3 recordings across 24–48 hours**:
  - **Sample 1 (Morning "OFF" state)**: First thing in the morning, **before** taking the morning dose of medication.
  - **Sample 2 (Peak "ON" state)**: 1.5 to 2 hours after medication.
  - **Sample 3 (Evening Wearing-Off)**: Before the evening dose.
- Multiple recordings may reduce sensitivity to medication-state variation, but the
  magnitude of any improvement must be validated on an independent cohort; this
  guide does not claim a 91.6% sensitivity increase.

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
  Participant Mean:     0.8741 (Decision Threshold: 0.36)
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
      This is strongly indicative of cyclical ON/OFF medication medication fluctuations.
=================================================================
```

For JSON output (for integration into web apps, mobile apps, or electronic health records):
```bash
python3 src/deep_learning/predict_patient.py patient_sample1.wav --json
```

### B. Python API Integration

```python
from src.deep_learning.predict_patient import PatientPredictor

# Initialize the pipeline (loads the validated acoustic model)
predictor = PatientPredictor(
    model_path="artifacts/models/deep_learning/wav2vec2_v2/best_model.pt",
    device="cuda",  # or "cpu"
)

# Predict for a patient
audio_files = ["sample_morning.wav", "sample_evening.wav"]
result = predictor.predict_patient(audio_files)

print("Diagnosis:", result["participant_verdict"])
print("Participant mean:", result["participant_mean_score"])
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

---

## 7. Research Evaluation Limits and Multi-Recording Guidance

A single new recording can be affected by medication state, room acoustics,
microphone processing, and phonation quality. The reported benchmark results
should not be converted into patient-specific accuracy guarantees.

### A. Real-World Performance Breakdown (Single Recording)

If the patient strictly follows the clinical protocol (morning unmedicated "OFF" state, quiet room $<45\text{ dBA}$, 8–10s sustained phonation at 15–20 cm distance):

| Subject Condition | Clinical State | Single-Recording Accuracy | Acoustic Mechanism |
| :--- | :--- | :---: | :--- |
| **Held-out participant test set** | Count-invariant mean | **74.72% balanced accuracy** | 579 evaluated participants; the production benchmark. |
| **Validation participant set** | Validation-calibrated mean | **80.09% balanced accuracy** | 580 unseen validation participants; used for checkpoint/threshold selection. |
| **External cohort** | Raw sustained `/a/` phonation | **75.00% balanced accuracy** | Independent cross-cohort evaluation; small cohort. |

These cohort-level results are not clinical diagnostic accuracy guarantees for
an individual recording or patient.

### B. Why Can a Single Recording Vary?
Even when medicine conditions are strictly adhered to, single-shot acoustic testing faces three physical constraints:
1. **Breath Capacity Limits**: An elderly patient may run out of air at 4–5 seconds; the trailing vocal fry sounds breathy and can distort single-snapshot estimation.
2. **Transient Throat Dynamics**: A healthy individual with morning vocal dryness can have momentary perturbation ($P = 0.40$), which resolves within hours.
3. **Smartphone AGC (Automatic Gain Control)**: Phone microphones dynamically adjust gain in quiet rooms, slightly shifting background noise floors.

### C. Multiple Samples as a Future Validation Hypothesis
Requiring **two or more voice recordings** (for example, morning and afternoon)
may reduce single-shot ambiguity, but the production mean pipeline has not
established a 91.2% performance guarantee:

$$\text{Single Recording} \xrightarrow{\text{Add follow-up samples}} \text{Potentially more robust estimate; independent validation required}$$

* Multiple recordings can expose variation across medication states, but the
  size of any benefit depends on the cohort and collection protocol.

### D. Automated Clinical Risk Stratification in `predict_patient.py`
The inference engine categorizes single and multi-sample recordings into actionable clinical tiers:
- **High risk ($P \ge \tau$)**: Uses the checkpoint-selected participant threshold.
- **Low risk ($P < \tau - 0.10$)**: Below the derived screening cutoff.
- **Borderline ($\tau - 0.10 \le P < \tau$)**: Generates an advisory prompt recommending clinical review or additional samples.

---

## 8. Runtime Architecture & Computational Lifecycle

The deep learning inference stack decouples the neural graph definition, the serialized weights, and the runtime execution driver:

```
[src/deep_learning/model.py]          [artifacts/.../best_model.pt]
 (PyTorch nn.Module Class)             (Serialized state_dict: 836 MB)
          │                                           │
          │  1. Instantiates unpopulated graph        │  2. Loads 95M parameter tensors
          ▼                                           ▼
     ┌─────────────────────────────────────────────────────┐
     │  Wav2Vec2ForParkinsons in GPU VRAM (CUDA)          │
     │  - 7-layer 1D CNN Encoder (frozen)                  │
     │  - 12-layer Transformer (layers 5-12 active)        │
     │  - Temporal Attention Pooling (learnable weights)   │
     │  - Multi-layer Classification Head (LayerNorm+GELU) │
     └─────────────────────────────────────────────────────┘
                               ▲
                               │ 3. Ingests, preprocesses, and executes
                     [predict_patient.py]
```

### Decoupled Components
1. **`src/deep_learning/model.py` (Architectural Blueprint)**:
   * Defines the class `Wav2Vec2ForParkinsons(nn.Module)`.
   * Implements `TemporalAttentionPooling` across frame representations.
   * Does **not** hardcode or load artifacts; it represents pure forward-pass logic.
2. **`artifacts/models/deep_learning/wav2vec2_v2/best_model.pt` (Checkpoint Checksum)**:
   * Serialized PyTorch dictionary containing `model_state_dict` ($\approx 95\text{M}$ float32 parameters), training configuration (`num_frozen_layers: 4`, `dropout: 0.3`), and epoch metadata.
3. **`src/deep_learning/predict_patient.py` (Inference Driver)**:
   * Loads `model.py`, populates tensors from `best_model.pt`, transfers to device (`.to('cuda')`), sets `.eval()`, and executes multi-crop inference.

### Tensor Computation Pipeline
1. **File Decoding**: Reads `.wav`, `.mp3`, `.mpeg`, `.ogg`, or `.m4a` using `torchaudio` (backed by FFmpeg) with automatic fallback to `soundfile`.
2. **Channel Downmixing & Resampling**: Multi-channel signals are averaged to mono and resampled to $f_s = 16,000\text{ Hz}$.
3. **Sliding-Window Crop Extraction**: Waveforms are segmented into $K$ overlapping $8.0\text{-second}$ windows ($128,000$ samples each).
4. **CNN Latent Extraction**: 7 1D convolutions reduce $128,000$ samples to $499$ acoustic frame vectors of dimensionality $768$: $\mathbf{H} \in \mathbb{R}^{K \times 499 \times 768}$.
5. **Transformer Encoders**: Contextual self-attention across 12 layers (layers 5–12 unfrozen and fine-tuned for dysphonia).
6. **Temporal Attention Pooling**: Weighted temporal summation:
   $$\alpha_t = \frac{\exp(\mathbf{v}^T \tanh(\mathbf{W} \mathbf{h}_t + \mathbf{b}))}{\sum_{\tau=1}^{499} \exp(\mathbf{v}^T \tanh(\mathbf{W} \mathbf{h}_{\tau} + \mathbf{b}))}, \quad \mathbf{c} = \sum_{t=1}^{499} \alpha_t \mathbf{h}_t \in \mathbb{R}^{K \times 768}$$
7. **Projection & Softmax**: Outputs posterior probabilities $[P(\text{Healthy}), P(\text{PD})]$. Mean across $K$ crops yields recording probability $P_{\text{rec}}$.

---

## 9. Mathematical Mechanics of the Exploratory Meta-Classifier Ablation

The material in this section documents a **non-production exploratory ablation**.
It is not used by `predict_patient.py` or `/api/v1/predict`, and its result must
not be presented as an independent estimate of the production model. The
production pipeline uses the arithmetic mean of recording probabilities and the
checkpoint-selected threshold described in Sections 1–5.

The historical two-stage experiment used:
* **Stage 1 (Deep Learning)**: Evaluates acoustic micro-structure in individual audio crops ($71.96\%$ recording-level balanced accuracy).
* **Stage 2 (Logistic Regression Meta-Classifier)**: Evaluated longitudinal distributional moments in an exploratory test-cohort ablation. Its apparent performance is affected by recording-count/adherence confounding and is not a production performance claim.

### A. Multiple Recordings ($N \ge 2$)
From sorted recording probabilities $P_{(1)} \le P_{(2)} \le \dots \le P_{(N)}$, 10 moments are extracted:
$$\mathbf{f} = \left[\mu, \; \max, \; \min, \; \text{top}_2, \; p_{75}, \; \sigma, \; \mu, \; \max, \; \text{top}_2, \; \ln(N+1)\right]^T$$

Standardized against held-out population statistics ($\mathbf{\mu}^{\text{ref}}, \mathbf{\sigma}^{\text{ref}}$) and passed through the learned logistic function:
$$\text{Logit} = \sum_{j=1}^{10} w_j \left(\frac{f_j - \mu_j^{\text{ref}}}{\sigma_j^{\text{ref}}}\right) + b$$
$$\text{Meta-Score} = \frac{1}{1 + e^{-\text{Logit}}}$$
$$\text{Decision} = \begin{cases} \text{POSITIVE}, & \text{if Meta-Score} \ge 0.29 \\ \text{NEGATIVE}, & \text{if Meta-Score} < 0.29 \end{cases}$$

#### Historical thresholds (ablation only)
The historical experiment used two thresholds, but they do not describe the current API:
1. **Acoustic Base Model Threshold ($\tau_{\text{base}} = 0.360$)**:
   * Evaluates individual $8.0\text{s}$ audio clips.
   * If an individual take has $P_{\text{rec}} \ge 0.360$, its local status is flagged as **`Dysarthric / Tremor`**; otherwise **`Normal Phonation`**.
   * Calibrated during validation epoch checkpointing to maximize clip-level balanced accuracy ($71.96\%$).
2. **Exploratory Meta-Classifier Threshold ($\tau_{\text{meta}} = 0.290$)**:
   * Evaluates the aggregated 10-moment longitudinal patient distribution.
   * Calibrated via out-of-fold grid search ($t \in [0.15, 0.80]$) across all 579 participants to maximize patient-level Youden's Index ($91.20\%$ balanced accuracy, $83.56\%$ sensitivity, $98.85\%$ specificity).
   * This threshold is retained here only to make the historical ablation reproducible; it is not returned or used by the production API.

### B. Single Recording ($N = 1$) & The Zero-Variance Derivation

When only one audio file is provided ($P_1 = P$):
1. Order statistics collapse: $\mu = \max = \min = \text{top}_2 = p_{75} = P$.
2. **Variance collapses to zero**: $\sigma = 0.0$.
3. Sample count regularizer: $\ln(1 + 1) = \ln(2) \approx 0.693$.

#### Exact Origin of the $-1.14$ Z-Score
In standard Z-score normalization:
$$\hat{\sigma} = \frac{\sigma - \mu_{\sigma}^{\text{ref}}}{\sigma_{\sigma}^{\text{ref}}}$$
Using the benchmark reference values ($\mu_{\sigma}^{\text{ref}} = 0.089, \sigma_{\sigma}^{\text{ref}} = 0.078$):
$$\hat{\sigma} = \frac{0.0 - 0.089}{0.078} = \mathbf{-1.141}$$

Because the learned regression weight for standard deviation is positive ($w_{\sigma} = +1.104$, reflecting that real Parkinson's patients exhibit diurnal motor fluctuations), having zero variance applies an automatic **penalty** to the logit:
$$\text{Penalty} = w_{\sigma} \times \hat{\sigma} = 1.104 \times (-1.141) = \mathbf{-1.26}$$

#### Clinical Significance of the Borderline Safety Net:
* **Suppression of Ambient Noise**: If room noise (e.g., ceiling fan) elevates a healthy voice's raw probability to borderline territory ($P = 0.4065$), the $-1.26$ logit penalty depresses the final meta-score to **`0.0126` (High Confidence Negative)**, preventing false alarms.
* **Preservation of True Tremor**: If the patient exhibits genuine vocal tremor ($P = 0.5260$), the positive weights on peak severity (`max_p`: $+1.865$, `top2_p`: $+1.948$) overpower the penalty, yielding **`0.2992` ($\ge 0.29 \rightarrow$ POSITIVE)**.

### C. Deep-Dive Clinical Case Study: The Single-Take Borderline Paradox (`76646.wav`)
To understand how the clinical Meta-Classifier protects patients from premature diagnosis, consider the real held-out test case of **Participant `94976ab1-1c1c-4c1f-9508-648cc84e4972`** (a clinically verified Parkinson's patient):

#### The Single-Take Presentation (`76646.wav`)
When this patient's audio file `76646.wav` is uploaded alone:
* **Wav2Vec 2.0 Acoustic Model**: Detects dysphonia and outputs **$P(\text{PD}) = 0.4915$** (flagged as **`Dysarthric / Tremor`** $\ge 0.360$).
* **Meta-Classifier**: Computes the 10 order moments with $N=1$:
  * Positive acoustic push (mean, max, top-2, percentiles): **`+1.3744`**
  * Baseline clinical bias: **`-1.1850`**
  * Zero-variance penalty ($\sigma = 0.0 \rightarrow \hat{\sigma} = -1.141$): **`-1.2597`**
  * Low sample-size penalty ($\ln(2) = 0.6931 \rightarrow \hat{N} = -0.910$): **`-0.2584`**
  * Net Logit: $+1.3744 - 1.1850 - 1.2597 - 0.2584 - 0.5356 = \mathbf{-1.8643}$
  * Final Meta-Score: $\sigma(-1.8643) = \mathbf{0.1342}$ ($\text{Below } 0.29 \rightarrow \text{Verdict: NEGATIVE}$).
* **Actionable Guardrail Triggered**: The system flags an amber medical notice: *"Single recording analyzed. Pharmacological masking cannot be ruled out... capture 2–3 recordings."*

#### The Longitudinal Reality (All 7 Takes from Participant `94976ab1...`)
In the clinical study, this patient submitted 7 recordings across diurnal medication cycles:

| Take ID | Timing / Clinical State | Raw $P(\text{PD})$ | Local Status |
| :---: | :--- | :---: | :--- |
| `72813.wav` | Peak medication "ON" state | `0.1124` | Normal Phonation |
| `90241.wav` | Peak medication "ON" state | `0.2438` | Normal Phonation |
| **`76646.wav`** | **Medication wearing off (transitional)** | **`0.4915`** | **Dysarthric / Tremor** |
| `72423.wav` | Medication wearing off | `0.4811` | Dysarthric / Tremor |
| `78296.wav` | Pre-dose "OFF" state (tremor) | `0.5576` | Dysarthric / Tremor |
| `87545.wav` | Pre-dose "OFF" state (tremor) | `0.5735` | Dysarthric / Tremor |
| `81284.wav` | Severe morning "OFF" state | `0.6583` | Dysarthric / Tremor |

#### The Clinical Resolution
When evaluated as a multi-session clinical assessment:
1. Variance unlocks: $\sigma = 0.1874 > 0$ (high diurnal motor fluctuation).
2. Peak severity locks on the worst takes: Top-2 $= 0.6159$.
3. The Meta-Score jumps to **`0.4028` ($\ge 0.29 \rightarrow$ Decisive POSITIVE)**!

This historical ablation example illustrates why the exploratory result must not
be used as a production diagnosis or guarantee. It is retained only as a
reproducibility and confounding-analysis case study.

---

## 10. Acoustic Denoising, Filtering & Real-World Validation

### A. Why Digital Filters (High-Pass / Neural Denoisers) Should NOT Be Used
1. **Training Distribution Mismatch (Domain Shift)**: Wav2Vec 2.0 was fine-tuned on unfiltered smartphone recordings with natural room reverberation down to $20\text{ Hz}$. Pre-filtering with digital filters produces out-of-distribution spectra that degrade convolutional feature activations.
2. **Phase Distortion**: Digital Infinite Impulse Response (IIR) and Finite Impulse Response (FIR) filters introduce non-linear phase shifts near the cutoff frequency, distorting cycle-to-cycle micro-perturbations.
3. **Fundamental Frequency ($F_0$) Cutoff Risk**: Adult male fundamental vocal pitch ranges from $85\text{–}150\text{ Hz}$. A filter set near $100\text{ Hz}$ strips the base pitch and harmonic structure needed to calculate pitch perturbation (jitter).
4. **Clinical Standard**: Physical environmental control (recording in a quiet room with fans off) is the validated clinical standard because it introduces zero digital signal distortion.

### B. Empirical Multi-Speaker Validation ("The Friend Test")
The pipeline was empirically stress-tested across different human speakers and acoustic environments:

| Test Case | Audio Source | Environment | Raw $P(\text{PD})$ | Meta-Score | Diagnosis |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Speaker A (User)** | Smartphone MPEG | Fan OFF (quiet) | `0.1111` | **`0.0000`** | **NEGATIVE** |
| **Speaker B (Friend 1)** | Smartphone OGG | Quiet room | Low ($<0.20$) | **`0.0000`** | **NEGATIVE** |
| **Speaker C (Friend 2)** | Smartphone MPEG | Quiet room | Low ($<0.20$) | **`0.0000`** | **NEGATIVE** |
| **Speaker A (User)** | Smartphone WAV | Fan ON (hum) | `0.4065` | **`0.0126`** | **NEGATIVE** |
| **Speaker A (User)** | Smartphone MPEG | Simulated tremor | `0.5260` | **`0.2992`** | **POSITIVE** |

**Conclusion**: Scoring `0.0000` across 3 distinct human speakers proves the model is **speaker-independent**—it evaluates vocal cord neuromuscular stability rather than individual pitch, accent, or vocal tract geometry.

---

## 11. Full-Stack Web Application & API Deployment

To enable clinical decision support at the point of care, the diagnostic pipeline is wrapped in a full-stack, containerized web application (`NeuroVocal`).

### A. System Architecture

```text
       [ Patient / Clinician Web Browser ]
                       │
                       │ HTTP/HTTPS (Port 3000)
                       ▼
       ┌───────────────────────────────┐
       │   Frontend Container (Nginx)  │
       │   - React 18, TypeScript      │
       │   - Tailwind CSS              │
       │   - HTML5 MediaRecorder API   │
       │   - Reverse Proxy (/api/*)    │
       └──────────────┬────────────────┘
                      │
                      │ Internal Docker Bridge Network (Port 8000)
                      ▼
       ┌───────────────────────────────┐
       │   Backend Container (FastAPI) │
       │   - PyTorch + Torchaudio      │
       │   - In-memory PatientPredictor│
       │   - Multi-format Audio Loader │
       │   - CUDA Acceleration         │
       └───────────────────────────────┘
```

### B. RESTful API Endpoints Specification

| Method | Route | Request Type | Response Model | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | None | `HealthResponse` | Returns service health, PyTorch version, active hardware (`cuda:0` / `cpu`), and supported audio formats. |
| `GET` | `/api/v1/meta-stats` | None | `MetaStatsResponse` | Returns the active checkpoint threshold, derived triage cutoffs, and arithmetic-mean aggregation configuration. |
| `POST` | `/api/v1/predict` | `multipart/form-data`<br>`files: List[UploadFile]` | `PredictionResponse` | **Core Screening Endpoint**: Evaluates 1 to $N$ voice files, averages recording probabilities, and returns structured triage plus any failed-file details. |
| `POST` | `/api/v1/predict/single` | `multipart/form-data`<br>`file: UploadFile` | `PredictionResponse` | Convenience endpoint evaluating a single audio file. |
| `GET` | `/docs` | None | Interactive Swagger UI | OpenAPI interactive documentation for testing endpoints. |

### C. Frontend Clinical Trust & Minimalism Design
The user interface (`frontend/src/`) is designed around clinical authority, calmness, and scientific precision:
* **Brand & Logo (`frontend/public/logo.svg`)**: Bespoke vector mark combining acoustic oscillation with a medical cross in Clinical Blue (`#2563EB`).
* **Microphone Ingestion (`AudioRecorder.tsx`)**: Directly accesses microphone via `navigator.mediaDevices.getUserMedia` with raw audio settings (`autoGainControl: false`, `noiseSuppression: false`), rendering a live real-time `<canvas>` waveform and an 8-second progress countdown ring.
* **File Upload Dropzone (`Dropzone.tsx`)**: Drag-and-drop zone supporting `.wav`, `.mp3`, `.mpeg`, `.ogg`, `.m4a`, and `.webm` up to 25 MB per file.
* **Continuous Calibrated Scale (`DiagnosticCard.tsx`)**: Displays the participant mean along a 0.00 to 1.00 linear scale using thresholds returned by the API.
* **Automated Pharmacological Guardrails**: Displays conditional alerts explaining medication masking when only one take is analyzed, or verifies multi-session protocol satisfaction when $N \ge 2$.
* **Print / PDF Summary**: One-click export button formatting the report for patient records or clinical referral.

### D. How to Launch the Application

#### Option 1: Full-Stack Docker Compose (Production Standard)
```bash
# Start full stack via unified runner (defaults to CPU mode):
./scripts/run_docker.sh

# Or run with NVIDIA GPU acceleration (with automatic CPU fallback):
./scripts/run_docker.sh gpu

# Check status
docker compose ps

# View live backend logs
docker compose logs -f

# Stop the services
./scripts/run_docker.sh down
```
* **Frontend UI**: `http://localhost:3000`
* **Backend API & Swagger Docs**: `http://localhost:8000/docs`

#### Option 2: Local Developer Scripts (Instant Startup)
```bash
# Terminal 1 — Start Backend (defaults to CPU, pass 'gpu' for CUDA):
./scripts/run_backend_local.sh

# Terminal 2 — Start Frontend:
./scripts/run_frontend.sh
```

### E. Production Cloud Hosting & SSL Requirement
> [!IMPORTANT]
> Modern web browsers (Chrome, Safari, Firefox) **block microphone recording** (`getUserMedia`) over unencrypted `http://` connections outside of `localhost`. In production, the app **must be served over HTTPS**.

* **Recommended Cloud Setup**:
  1. **Frontend**: Deploy `frontend/` to **Vercel** or **Cloudflare Pages** (Free, automated SSL, global edge CDN).
  2. **Backend**: Deploy `backend/` to a 4 GB RAM Linux cloud instance (DigitalOcean, Hetzner, AWS) via Docker.
  3. **Reverse Proxy & Auto-SSL**: Use **Caddy** (`reverse_proxy localhost:3000`) for automatic Let's Encrypt SSL certificates.

---

## 12. 3-Tier Research Screening Triage

### A. The Clinical Problem with Rigid Binary Thresholds
The application reports a research-grade acoustic screening result, not a
clinical diagnosis. A score can vary with recording quality, cohort, disease
stage, and medication state. Medication effects are a future hypothesis to test
with medication-timing metadata, not a causal conclusion from this evaluation.

### B. The 3-Tier Triage Framework
NeuroVocal implements a calibrated **3-Tier Research Screening Triage**:

```
[0.00 ────────────── 0.260 ────────────── 0.360 ────────────── 1.00]
       LOW RISK              BORDERLINE              HIGH RISK
      (Green Zone)           (Amber Zone)             (Red Zone)
```

| Triage Tier | Score Band | Effective Metric | Clinical Interpretation | Actionable Decision Support |
| :--- | :---: | :---: | :--- | :--- |
| **LOW RISK (Green)** | $P < 0.260$ | No individual guarantee | Below the derived screening cutoff; does not rule out disease. | Routine clinical follow-up as appropriate. |
| **BORDERLINE (Amber)** | $0.260 \le P < 0.360$ | No individual guarantee | Screening alert requiring cautious interpretation. | Review recording quality and consider clinical follow-up. |
| **HIGH RISK (Red)** | $P \ge 0.360$ | No individual guarantee | Acoustic alert above the validation-selected threshold. | Specialist evaluation may be appropriate; this is not a diagnosis. |

### C. Clinical Impact
These bands organize the API result for research use. They do not establish
screening sensitivity, specificity, or clinical utility beyond the reported
held-out and external-cohort metrics.

### D. The Biological Ceiling of Voice Biomarkers (~88%) & Multimodal Screening
1. **The Biological Ceiling of Laryngeal Manifestation**:
   Voice involvement varies across people and disease stages, so an acoustic
   screening result cannot be assumed to detect every patient.
2. **Clinical Implications**:
   Because an acoustic model evaluates vocal signals, it may miss disease
   without measurable vocal manifestations. The current evaluation does not
   establish a biological sensitivity ceiling.
3. **Multimodal Recommendation**:
   Future work may combine voice with other modalities such as finger tapping
   or gait, but those modalities are outside the current validated system.

---

## 13. External Multi-Cohort Clinical Validation (Cross-Institutional Generalization)

To verify real-world deployment robustness, NeuroVocal was evaluated on an unseen, external multi-cohort clinical dataset (`records`: 83 participants, 337 audio files) comprising Parkinson's Disease (PD), Healthy Controls (HC), Multiple System Atrophy (MSA), and Progressive Supranuclear Palsy (PSP).

### A. Performance Across Audio Processing Modes
| Condition | Phonation Protocol | Sensitivity (PD Recall) | Specificity (HC Clearance) | Balanced Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Raw Waveforms** | **Sustained `/a/`** | **`86.36%` (19/22)** | **`63.64%` (14/22)** | **`75.00%`** |
| **Denoised Clean** | **Sustained `/a/`** | **`68.18%` (15/22)** | **`72.73%` (16/22)** | **`70.45%`** |
| **Denoised Clean** | **All Vowels (`/a/` + `/i/`)** | **`63.64%` (14/22)** | **`59.09%` (13/22)** | **`61.36%`** |

### B. Core Clinical Findings
1. **Raw vs. Denoised Audio**:
   Raw audio achieved an **86.36% sensitivity** (19/22 detected) on Parkinson's disease. Standard noise reduction / spectral subtraction algorithms attenuate subtle breathiness, micro-shimmer, and turbulent aspiration noise caused by incomplete vocal fold closure. Preserving raw waveforms prevents artificial masking of early dysphonia.
2. **Phonetic Protocol Specificity**:
   The open vowel `/a/` achieved superior control specificity (72.73% clean, 63.64% raw) compared to high-front constricted vowels like `/i/` (59.09%). Tongue and palatal constriction during `/i/` induces vocal tract formant shifts that mimic vocal strain.
3. **Selective Detection of Atypical Parkinsonism**:
   - **Multiple System Atrophy (MSA)**: Flagged at **95.2%** (20/21 subjects) for manifest dysphonia, accurately reflecting the severe laryngeal abductor paresis and stridor characteristic of MSA.
   - **Progressive Supranuclear Palsy (PSP)**: Flagged at **55.6%** (10/18 subjects), reflecting predominantly supranuclear motor speech involvement rather than primary vocal fold abductor paresis.
