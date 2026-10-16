"""Patient-Level Parkinson's Detection Inference Pipeline.

Takes one or more sustained vowel audio recordings for a patient, runs
Wav2Vec 2.0 acoustic inference with Test-Time Augmentation (TTA), extracts
the multi-recording statistical distribution, and outputs a clinical
diagnosis using the Participant-Level Meta-Classifier.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))


def dot(w: List[float], x: List[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


# Meta-classifier weights learned across 5-fold cross validation
META_WEIGHTS = [1.242, 1.865, -0.412, 1.948, 0.835, 1.104, 0.621, 0.891, 0.952, 0.284]
META_MEANS = [0.385, 0.512, 0.281, 0.472, 0.438, 0.089, 0.362, 0.485, 0.448, 1.250]
META_STDS = [0.245, 0.289, 0.215, 0.278, 0.261, 0.078, 0.231, 0.274, 0.263, 0.612]
META_BIAS = -1.185
META_THRESHOLD = 0.29


class PatientPredictor:
    """End-to-end patient inference pipeline."""

    def __init__(
        self,
        model_path: Union[str, Path] = "artifacts/models/deep_learning/wav2vec2_v2/best_model.pt",
        device: Optional[str] = None,
        duration: float = 8.0,
        sample_rate: int = 16000,
        num_crops: int = 5,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.duration = duration
        self.sample_rate = sample_rate
        self.target_length = int(duration * sample_rate)
        self.num_crops = num_crops

        print(f"Loading Wav2Vec2 Parkinson's model from: {model_path} (on {self.device})...")
        from src.deep_learning.model import Wav2Vec2ForParkinsons

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        num_frozen = config.get("num_frozen_layers", checkpoint.get("num_frozen_layers", 4))
        dropout = config.get("dropout", checkpoint.get("dropout", 0.3))
        self.model = Wav2Vec2ForParkinsons(
            num_frozen_layers=num_frozen,
            dropout=dropout,
            gradient_checkpointing=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def load_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess audio to 16 kHz mono waveform."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)
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
            outputs = self.model(input_values=batch)
            probs = outputs["probabilities"][:, 1]
            rec_prob = float(probs.mean().item())

        return rec_prob

    def predict_patient(self, audio_paths: List[Union[str, Path]]) -> Dict:
        """Aggregate multiple audio recordings for a patient and return diagnosis."""
        if not audio_paths:
            raise ValueError("At least one audio recording is required.")

        recording_results = []
        for path in audio_paths:
            try:
                waveform = self.load_audio(path)
                prob = self.infer_single_recording(waveform)
                recording_results.append({
                    "file": str(path),
                    "duration_seconds": round(waveform.shape[0] / self.sample_rate, 2),
                    "probability_PD": round(prob, 4),
                    "recording_status": "Dysarthric / Tremor" if prob >= 0.360 else "Normal Phonation",
                })
            except Exception as e:
                print(f"Warning: Failed to process {path}: {e}", file=sys.stderr)

        if not recording_results:
            raise RuntimeError("No valid audio files could be processed.")

        probs = sorted([r["probability_PD"] for r in recording_results])
        n = len(probs)
        mean_p = sum(probs) / n
        max_p = probs[-1]
        min_p = probs[0]
        top2_p = sum(probs[-2:]) / min(n, 2)
        p75_p = probs[min(int(n * 0.75), n - 1)]
        var_p = sum((x - mean_p) ** 2 for x in probs) / n
        std_p = math.sqrt(var_p)

        # Multi-moment meta-features
        feats = [mean_p, max_p, min_p, top2_p, p75_p, std_p, mean_p, max_p, top2_p, math.log(n + 1)]
        norm_x = [(feats[j] - META_MEANS[j]) / META_STDS[j] for j in range(len(feats))]
        meta_score = sigmoid(dot(META_WEIGHTS, norm_x) + META_BIAS)

        is_pd = meta_score >= META_THRESHOLD
        if meta_score >= 0.70:
            confidence = "High Confidence Positive"
        elif meta_score >= META_THRESHOLD:
            confidence = "Moderate Confidence Positive"
        elif meta_score <= 0.15:
            confidence = "High Confidence Negative"
        else:
            confidence = "Moderate Confidence Negative"

        # Clinical warnings
        warnings = []
        if n == 1:
            warnings.append(
                "Single recording analyzed. Pharmacological masking (Levodopa 'ON' state) "
                "cannot be ruled out. Clinical recommendation: capture at least 2–3 recordings "
                "24 hours apart (morning pre-medication and evening)."
            )
        if std_p >= 0.25 and n >= 2:
            warnings.append(
                "High variance between recordings detected (std dev > 0.25). This is strongly "
                "indicative of cyclical ON/OFF Levodopa medication fluctuations."
            )

        return {
            "participant_verdict": "POSITIVE (Parkinsonian Dysphonia Detected)" if is_pd else "NEGATIVE (Healthy Phonation)",
            "is_parkinsons": is_pd,
            "meta_classifier_score": round(meta_score, 4),
            "meta_threshold": META_THRESHOLD,
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
        }


def main():
    parser = argparse.ArgumentParser(description="Predict Parkinson's from patient voice recordings")
    parser.add_argument("audio_files", nargs="+", help="Path(s) to patient audio file(s) (.wav, .m4a, .mp3)")
    parser.add_argument("--model-path", default="artifacts/models/deep_learning/wav2vec2_v2/best_model.pt")
    parser.add_argument("--device", default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--json", action="store_true", help="Output pure JSON")
    args = parser.parse_args()

    predictor = PatientPredictor(model_path=args.model_path, device=args.device)
    report = predictor.predict_patient(args.audio_files)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("\n" + "=" * 65)
    print("      PARKINSON'S ACOUSTIC SCREENING REPORT")
    print("=" * 65)
    print(f"  Final Diagnosis:      {report['participant_verdict']}")
    print(f"  Confidence Level:     {report['confidence_level']}")
    print(f"  Meta-Classifier Score:{report['meta_classifier_score']:.4f} (Threshold: {report['meta_threshold']})")
    print(f"  Recordings Analyzed:  {report['recordings_analyzed']}")
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

    if report["clinical_warnings"]:
        print("-" * 65)
        print("  Clinical Observations / Warnings:")
        for w in report["clinical_warnings"]:
            print(f"    * {w}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

