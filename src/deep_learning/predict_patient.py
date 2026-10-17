"""Patient-level Parkinson's detection inference pipeline.

Takes one or more sustained vowel audio recordings for a patient, runs
Wav2Vec 2.0 acoustic inference with Test-Time Augmentation (TTA), averages
the recording probabilities, and outputs a clinical screening triage result.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchaudio


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))


def dot(w: List[float], x: List[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


# Default model and calibrated threshold (WaveLM v4 best model)
DEFAULT_WAVELM_PATH = "artifacts/models/deep_learning/wavelm_v4/best_model.pt"
DEFAULT_WAV2VEC2_PATH = "artifacts/models/deep_learning/wav2vec2_v2/best_model.pt"

WAVELM_OPTIMAL_THRESHOLD = 0.310
ACOUSTIC_OPTIMAL_THRESHOLD = WAVELM_OPTIMAL_THRESHOLD
SCREENING_CUTOFF = 0.210
DIAGNOSTIC_CUTOFF = 0.410

# Backward-compatibility reference exports for API endpoints
META_WEIGHTS = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
META_MEANS = [0.385, 0.512, 0.281, 0.472, 0.438, 0.089, 0.362, 0.485, 0.448, 1.250]
META_STDS = [0.245, 0.289, 0.215, 0.278, 0.261, 0.078, 0.231, 0.274, 0.263, 0.612]
META_BIAS = 0.0
META_THRESHOLD = ACOUSTIC_OPTIMAL_THRESHOLD


class PatientPredictor:
    """End-to-end patient inference pipeline powered by WaveLM v4 best model."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_WAVELM_PATH,
        secondary_model_path: Optional[Union[str, Path]] = None,
        use_ensemble: bool = False,
        ensemble_weights: tuple = (0.75, 0.25),
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        duration: float = 8.0,
        sample_rate: int = 16000,
        num_crops: int = 5,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_crops = num_crops
        self.ensemble_weights = ensemble_weights

        def _load_model_and_config(ckpt_path):
            from src.deep_learning.model import build_speech_model
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            cfg = checkpoint.get("config", {})
            m_name = cfg.get(
                "model_name",
                "microsoft/wavlm-base-plus" if "wavelm" in str(ckpt_path).lower() else "facebook/wav2vec2-base"
            )
            num_frozen = cfg.get("num_frozen_layers", checkpoint.get("num_frozen_layers", 4))
            dropout = cfg.get("dropout", checkpoint.get("dropout", 0.3))
            model = build_speech_model(
                model_name=m_name,
                num_frozen_layers=num_frozen,
                dropout=dropout,
                gradient_checkpointing=False,
            )
            raw_state_dict = checkpoint["model_state_dict"]
            state_dict = {k: v for k, v in raw_state_dict.items() if not k.startswith("loss_fn.")}
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
            return model, cfg, checkpoint

        path1 = Path(model_path) if model_path else None
        path2 = Path(secondary_model_path) if secondary_model_path else None
        self.is_ensemble = bool(use_ensemble and path1 and path1.exists() and path2 and path2.exists())

        if self.is_ensemble:
            print(f">>> [ENSEMBLE] Multi-Backbone Ensemble Initializing on {self.device}")
            print(f"    - Primary Backbone   (Weight {self.ensemble_weights[0]:.2f}): {path1}")
            self.model, cfg1, ckpt1 = _load_model_and_config(path1)
            print(f"    - Secondary Backbone (Weight {self.ensemble_weights[1]:.2f}): {path2}")
            self.secondary_model, cfg2, ckpt2 = _load_model_and_config(path2)
            self.duration = float(cfg1.get("duration", duration))
            self.sample_rate = int(cfg1.get("sample_rate", sample_rate) or sample_rate)
            self.target_length = int(self.duration * self.sample_rate)
            self.optimal_threshold = threshold if threshold is not None else 0.298
            print(f"    - Operating Threshold: {self.optimal_threshold:.3f}")
        else:
            active_path = path1 if (path1 and path1.exists()) else path2
            if not active_path or not active_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            print(f">>> Loading single speech model from: {active_path} (on {self.device})...")
            self.model, cfg, ckpt = _load_model_and_config(active_path)
            self.secondary_model = None
            self.duration = float(cfg.get("duration", duration))
            self.sample_rate = int(cfg.get("sample_rate", sample_rate) or sample_rate)
            self.target_length = int(self.duration * self.sample_rate)
            self.optimal_threshold = threshold if threshold is not None else float(ckpt.get("optimal_threshold", ACOUSTIC_OPTIMAL_THRESHOLD))
            print(f"    - Calibrated Operating Threshold: {self.optimal_threshold:.3f}")

    def load_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess audio to the checkpoint sample rate in mono."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform = None
        try:
            raw_wave, sr = torchaudio.load(str(path))
            if raw_wave.shape[0] > 1:
                raw_wave = raw_wave.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                raw_wave = resampler(raw_wave)
            waveform = raw_wave.squeeze(0)
        except Exception:
            waveform = None

        if waveform is None:
            try:
                import soundfile as sf
                data, sr = sf.read(str(path), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                w = torch.from_numpy(data)
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                    w = resampler(w.unsqueeze(0)).squeeze(0)
                waveform = w
            except Exception:
                waveform = None

        if waveform is None:
            # Robust ffmpeg decoding for .mpeg, .webm, .m4a, and non-standard audio containers
            import subprocess
            cmd = [
                "ffmpeg", "-v", "error", "-i", str(path),
                "-f", "f32le", "-ac", "1", "-ar", str(self.sample_rate), "pipe:1"
            ]
            try:
                out = subprocess.check_output(cmd)
                audio_np = np.frombuffer(out, dtype=np.float32).copy()
                waveform = torch.from_numpy(audio_np)
            except Exception as e:
                raise ValueError(f"Failed to decode audio file '{path.name}': {e}")

        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise ValueError("Audio contains no usable finite samples.")

        duration_seconds = waveform.shape[0] / self.sample_rate
        if duration_seconds < 1.0:
            raise ValueError(
                f"Recording is too short ({duration_seconds:.2f}s); "
                f"at least 1.00s of audio is required."
            )

        # Pad with zeros if shorter than target duration (matches training dataset preprocessing)
        if waveform.shape[0] < self.target_length:
            pad_len = self.target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # Peak normalization to [-1.0, 1.0] matching training dataset preprocessing
        max_val = torch.max(torch.abs(waveform))
        if max_val <= 1e-6:
            raise ValueError("Audio contains no usable phonation; the waveform is silent.")
        waveform = waveform / max_val
        return waveform

    def infer_single_recording(self, waveform: torch.Tensor) -> float:
        """Predict PD probability for a single audio recording using multi-crop TTA."""
        n_samples = waveform.shape[0]
        crops = []

        if n_samples <= self.target_length:
            pad_amount = self.target_length - n_samples
            padded = torch.nn.functional.pad(waveform, (0, pad_amount))
            for _ in range(self.num_crops):
                crops.append(padded)
        else:
            max_start = n_samples - self.target_length
            step = max_start / max(self.num_crops - 1, 1)
            for i in range(self.num_crops):
                start = int(i * step)
                crops.append(waveform[start : start + self.target_length])

        batch = torch.stack(crops, dim=0).to(self.device)
        with torch.no_grad():
            outputs1 = self.model(input_values=batch)
            p1 = float(outputs1["probabilities"][:, 1].mean().item())

            if self.is_ensemble and self.secondary_model is not None:
                outputs2 = self.secondary_model(input_values=batch)
                p2 = float(outputs2["probabilities"][:, 1].mean().item())

                # Option B: Bayesian Log-Odds consensus (75% WaveLM v4 + 25% Wav2Vec2 v2)
                eps = 1e-6
                p1_c = max(min(p1, 1.0 - eps), eps)
                p2_c = max(min(p2, 1.0 - eps), eps)
                z1 = math.log(p1_c / (1.0 - p1_c))
                z2 = math.log(p2_c / (1.0 - p2_c))
                w1, w2 = self.ensemble_weights
                z_ens = w1 * z1 + w2 * z2
                rec_prob = sigmoid(z_ens)
            else:
                rec_prob = p1

        return rec_prob

    def predict_patient(self, audio_paths: List[Union[str, Path]]) -> Dict:
        """Aggregate multiple audio recordings for a patient and return diagnosis."""
        if not audio_paths:
            raise ValueError("At least one audio recording is required.")

        recording_results = []
        failed_recordings = []
        for path in audio_paths:
            try:
                waveform = self.load_audio(path)
                prob = self.infer_single_recording(waveform)
                recording_results.append({
                    "file": str(path),
                    "duration_seconds": round(waveform.shape[0] / self.sample_rate, 2),
                    "probability_PD": round(prob, 4),
                    "recording_status": (
                        "Dysarthric / Tremor"
                        if prob >= self.optimal_threshold
                        else "Normal Phonation"
                    ),
                })
            except Exception as e:
                failed_recordings.append({
                    "file": str(path),
                    "error": str(e),
                })
                print(f"Warning: Failed to process {path}: {e}", file=sys.stderr)

        if not recording_results:
            details = "; ".join(
                f"{failure['file']}: {failure['error']}" for failure in failed_recordings
            )
            raise RuntimeError(f"No valid audio files could be processed. {details}")

        probs = sorted([r["probability_PD"] for r in recording_results])
        n = len(probs)
        mean_p = sum(probs) / n
        max_p = probs[-1]
        min_p = probs[0]
        top2_p = sum(probs[-2:]) / min(n, 2)
        p75_p = probs[min(int(n * 0.75), n - 1)]
        var_p = sum((x - mean_p) ** 2 for x in probs) / n
        std_p = math.sqrt(var_p)

        # Count-invariant participant acoustic score: arithmetic mean of recording probabilities
        participant_score = mean_p

        # Clinical Triage Framework (Calibrated around validation-selected optimal threshold)
        balanced_threshold = self.optimal_threshold
        screening_cutoff = max(0.15, balanced_threshold - 0.10)
        diagnostic_cutoff = min(0.85, balanced_threshold + 0.10)

        if participant_score < screening_cutoff:
            triage_level = "LOW_RISK"
            participant_verdict = "NEGATIVE (Normal Phonation Dynamics)"
            is_pd = False
            confidence = "High Confidence Negative" if participant_score <= screening_cutoff * 0.7 else "Normal Phonation"
            triage_rec = "Normal acoustic voice dynamics observed. Routine clinical follow-up as needed."
        elif participant_score < balanced_threshold:
            triage_level = "BORDERLINE"
            participant_verdict = "BORDERLINE (Mild Acoustic Dysphonia)"
            is_pd = True  # Flagged for conservative screening follow-up
            confidence = "Borderline Screening Alert"
            triage_rec = (
                f"Acoustic features fall within borderline triage window [{screening_cutoff:.2f}–{balanced_threshold:.2f}]. "
                "Secondary phonation test or clinical review recommended."
            )
        else:
            triage_level = "HIGH_RISK"
            participant_verdict = "POSITIVE (Acoustic Dysphonia Detected)"
            is_pd = True
            confidence = "High Confidence Positive" if participant_score >= diagnostic_cutoff else "Moderate Confidence Positive"
            triage_rec = (
                "Acoustic dysphonia and vocal stability degradation detected above decision threshold. "
                "Specialist neurological evaluation advised."
            )

        # Clinical warnings
        warnings = []
        if n == 1:
            warnings.append(
                "Single recording analyzed. Diurnal acoustic fluctuations "
                "cannot be ruled out. Clinical recommendation: capture at least 2–3 recordings "
                "across different times of day."
            )
        if std_p >= 0.20 and n >= 2:
            warnings.append(
                "High variance between recordings detected (std dev > 0.20). "
                "This may reflect recording conditions or within-person variation; "
                "medication effects were not tested in this evaluation."
            )
        if failed_recordings:
            warnings.append(
                f"{len(failed_recordings)} recording(s) were not analyzed. "
                "Review the failed-recording details before interpreting this result."
            )

        return {
            "participant_verdict": participant_verdict,
            "is_parkinsons": is_pd,
            "triage_level": triage_level,
            "triage_recommendation": triage_rec,
            # Keep the legacy key for clients that still consume API v2.0.
            "participant_mean_score": round(participant_score, 4),
            "meta_classifier_score": round(participant_score, 4),
            "meta_threshold": balanced_threshold,
            "screening_cutoff": screening_cutoff,
            "diagnostic_cutoff": diagnostic_cutoff,
            "confidence_level": confidence,
            "recordings_analyzed": n,
            "distribution_summary": {
                "mean_probability": round(mean_p, 4),
                "top2_probability": round(top2_p, 4),
                "max_probability": round(max_p, 4),
                "min_probability": round(min_p, 4),
                "std_deviation": round(std_p, 4),
            },
            "clinical_warnings": warnings,
            "recordings_detail": recording_results,
            "failed_recordings": failed_recordings,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict Parkinson's from patient voice recordings")
    parser.add_argument("audio_files", nargs="+", help="Path(s) to patient audio file(s) (.wav, .m4a, .mp3)")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_WAVELM_PATH,
        help="Primary model checkpoint (default: WaveLM v4 best model)",
    )
    parser.add_argument(
        "--secondary-model-path",
        default=DEFAULT_WAV2VEC2_PATH,
        help="Secondary model checkpoint for optional ensemble",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Enable multi-backbone ensemble (WaveLM v4 + Wav2Vec2 v2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: auto-selected from checkpoint)",
    )
    parser.add_argument("--device", default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--json", action="store_true", help="Output pure JSON")
    args = parser.parse_args()

    predictor = PatientPredictor(
        model_path=args.model_path,
        secondary_model_path=args.secondary_model_path if args.ensemble else None,
        use_ensemble=args.ensemble,
        threshold=args.threshold,
        device=args.device,
    )
    report = predictor.predict_patient(args.audio_files)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("\n" + "=" * 65)
    print("      PARKINSON'S ACOUSTIC SCREENING REPORT")
    print("=" * 65)
    print(f"  Final Diagnosis:      {report['participant_verdict']}")
    print(
        f"  Clinical Triage:      {report['triage_level']} "
        f"(Screening: <{report['screening_cutoff']:.3f} | "
        f"Diagnostic: ≥{report['diagnostic_cutoff']:.3f})"
    )
    print(f"  Confidence Level:     {report['confidence_level']}")
    print(
        f"  Participant Mean:     {report['participant_mean_score']:.4f} "
        f"(Decision Threshold: {report['meta_threshold']:.3f})"
    )
    print(f"  Recordings Analyzed:  {report['recordings_analyzed']}")
    print(f"  Recordings Failed:    {len(report['failed_recordings'])}")
    print("-" * 65)
    print(f"  Triage Recommendation: {report['triage_recommendation']}")
    print("-" * 65)
    print("  Statistical Distribution:")
    print(f"    - Mean Probability: {report['distribution_summary']['mean_probability']:.4f}")
    print(f"    - Top-2 Probability:{report['distribution_summary']['top2_probability']:.4f}")
    print(f"    - Max Probability:  {report['distribution_summary']['max_probability']:.4f}")
    print(f"    - Std Deviation:    {report['distribution_summary']['std_deviation']:.4f}")
    print("-" * 65)
    print("  Recording Details:")
    for i, r in enumerate(report["recordings_detail"], 1):
        print(f"    [{i}] {Path(r['file']).name} ({r['duration_seconds']}s): P(PD) = {r['probability_PD']} -> {r['recording_status']}")
    for failure in report["failed_recordings"]:
        print(f"    [Failed] {Path(failure['file']).name}: {failure['error']}")

    if report["clinical_warnings"]:
        print("-" * 65)
        print("  Clinical Observations / Warnings:")
        for w in report["clinical_warnings"]:
            print(f"    * {w}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
