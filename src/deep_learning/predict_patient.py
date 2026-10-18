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


# Default model and calibrated threshold (WaveLM champion model + Attention-MIL)
DEFAULT_WAVELM_PATH = "artifacts/models/deep_learning/wavelm/best_model.pt"
DEFAULT_WAV2VEC2_PATH = "artifacts/models/deep_learning/wav2vec2/best_model.pt"
DEFAULT_MIL_PATH = "artifacts/models/deep_learning/wavelm/mil_head.pt"

WAVELM_OPTIMAL_THRESHOLD = 0.310
MIL_OPTIMAL_THRESHOLD = 0.300
ACOUSTIC_OPTIMAL_THRESHOLD = MIL_OPTIMAL_THRESHOLD
SCREENING_CUTOFF = 0.200
DIAGNOSTIC_CUTOFF = 0.500

# Backward-compatibility reference exports for API endpoints
META_WEIGHTS = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
META_MEANS = [0.385, 0.512, 0.281, 0.472, 0.438, 0.089, 0.362, 0.485, 0.448, 1.250]
META_STDS = [0.245, 0.289, 0.215, 0.278, 0.261, 0.078, 0.231, 0.274, 0.263, 0.612]
META_BIAS = 0.0
META_THRESHOLD = ACOUSTIC_OPTIMAL_THRESHOLD


class PatientPredictor:
    """End-to-end patient inference pipeline powered by WaveLM + Gated Attention-MIL."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_WAVELM_PATH,
        mil_model_path: Optional[Union[str, Path]] = DEFAULT_MIL_PATH,
        use_attention_mil: bool = True,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        duration: float = 8.0,
        sample_rate: int = 16000,
        num_crops: int = 3,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_crops = num_crops
        self.use_attention_mil = use_attention_mil
        self.mil_model_path = Path(mil_model_path) if mil_model_path else None
        self.mil_model = None

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

        active_path = Path(model_path) if model_path else Path(DEFAULT_WAVELM_PATH)
        if not active_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        print(f">>> Loading single speech model from: {active_path} (on {self.device})...")
        self.model, cfg, ckpt = _load_model_and_config(active_path)
        self.duration = float(cfg.get("duration", duration))
        self.sample_rate = int(cfg.get("sample_rate", sample_rate) or sample_rate)
        self.target_length = int(self.duration * self.sample_rate)
        self.optimal_threshold = threshold if threshold is not None else float(ckpt.get("optimal_threshold", WAVELM_OPTIMAL_THRESHOLD))
        print(f"    - Backbone Operating Threshold: {self.optimal_threshold:.3f}")

        # Initialize Gated Attention-MIL aggregation head
        if self.use_attention_mil and self.mil_model_path and self.mil_model_path.exists():
            try:
                from src.mil.model import GatedAttentionMIL
                mil_ckpt = torch.load(self.mil_model_path, map_location=self.device, weights_only=False)
                mil_cfg = mil_ckpt.get("config", {})
                self.mil_model = GatedAttentionMIL(
                    in_dim=mil_cfg.get("in_dim", 768),
                    attention_dim=mil_cfg.get("attention_dim", 128),
                    hidden_dim=mil_cfg.get("hidden_dim", 128),
                    dropout=mil_cfg.get("dropout", 0.25),
                    use_residual_mean=mil_cfg.get("use_residual_mean", False),
                )
                self.mil_model.load_state_dict(mil_ckpt["model_state_dict"])
                self.mil_model.to(self.device).eval()
                if threshold is None:
                    self.optimal_threshold = float(mil_ckpt.get("optimal_threshold", MIL_OPTIMAL_THRESHOLD))
                print(f">>> [ATTENTION-MIL] Gated Attention Aggregation Loaded: {self.mil_model_path}")
                print(f"    - Calibrated Operating Threshold: {self.optimal_threshold:.3f}")
            except Exception as e:
                print(f"    [Warning] Failed to load Attention-MIL head ({e}); falling back to arithmetic mean.")
                self.mil_model = None

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

    def infer_single_recording(self, waveform: torch.Tensor, return_embeddings: bool = False):
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
            rec_prob = float(outputs["probabilities"][:, 1].mean().item())
            embs = outputs["embeddings"].cpu()

        if return_embeddings:
            return rec_prob, embs
        return rec_prob

    def predict_patient(self, audio_paths: List[Union[str, Path]]) -> Dict:
        """Aggregate multiple audio recordings for a patient and return diagnosis."""
        if not audio_paths:
            raise ValueError("At least one audio recording is required.")

        recording_results = []
        failed_recordings = []
        all_embs = []
        for path in audio_paths:
            try:
                waveform = self.load_audio(path)
                prob, embs = self.infer_single_recording(waveform, return_embeddings=True)
                all_embs.append(embs)
                recording_results.append({
                    "file": str(path),
                    "duration_seconds": round(waveform.shape[0] / self.sample_rate, 2),
                    "probability_PD": round(prob, 4),
                    "recording_status": (
                        "Dysarthric / Tremor"
                        if prob >= self.optimal_threshold
                        else "Normal Phonation"
                    ),
                    "attention_weight": round(1.0 / len(audio_paths), 4),
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

        # Gated Attention-MIL participant aggregation
        aggregation_method = "arithmetic_mean"
        crop_weights_list = None
        if n == 1:
            # Single recording session: participant score is directly the recording's calibrated score
            participant_score = recording_results[0]["probability_PD"]
            recording_results[0]["attention_weight"] = 1.0
            if self.use_attention_mil and self.mil_model is not None:
                aggregation_method = "gated_attention_mil"
        elif self.use_attention_mil and self.mil_model is not None and all_embs:
            bag_tensor = torch.cat(all_embs, dim=0).to(self.device)
            with torch.no_grad():
                mil_out = self.mil_model(bag_tensor)
                mil_prob = float(mil_out["probability"].item())
                crop_weights = mil_out["attention_weights"].cpu().numpy()
                crop_weights_list = [round(float(w), 4) for w in crop_weights]

            aggregation_method = "gated_attention_mil"
            crops_per_rec = self.num_crops
            total_w = 0.0
            for idx, r in enumerate(recording_results):
                r_start = idx * crops_per_rec
                r_end = r_start + crops_per_rec
                if r_end <= len(crop_weights):
                    rec_weight = float(np.sum(crop_weights[r_start:r_end]))
                else:
                    rec_weight = 1.0 / len(recording_results)
                r["attention_weight"] = rec_weight
                total_w += rec_weight

            # Normalize weights to sum to 1.0
            if total_w > 1e-6:
                for r in recording_results:
                    r["attention_weight"] = round(r["attention_weight"] / total_w, 4)

            # Attention-weighted instance probability
            weighted_p = sum(r["attention_weight"] * r["probability_PD"] for r in recording_results)
            participant_score = 0.5 * weighted_p + 0.5 * mil_prob

            # Clinical Consensus Guardrail:
            # If ALL recordings are unanimously below the operating threshold (Normal Phonation),
            # the participant cannot be declared Positive.
            if all(r["probability_PD"] < self.optimal_threshold for r in recording_results):
                participant_score = min(participant_score, max_p)
        else:
            participant_score = mean_p
            for r in recording_results:
                r["attention_weight"] = round(1.0 / len(recording_results), 4)

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
            "aggregation_method": aggregation_method,
            "attention_weights": crop_weights_list,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict Parkinson's from patient voice recordings")
    parser.add_argument("audio_files", nargs="+", help="Path(s) to patient audio file(s) (.wav, .m4a, .mp3)")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_WAVELM_PATH,
        help="Model checkpoint path (default: WaveLM champion model)",
    )
    parser.add_argument(
        "--mil-path",
        default=DEFAULT_MIL_PATH,
        help="Gated Attention-MIL head path (default: WaveLM mil_head.pt)",
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
        mil_model_path=args.mil_path,
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
    print(f"  Aggregation Method:   {report.get('aggregation_method', 'arithmetic_mean')}")
    print(
        f"  Participant Score:    {report['participant_mean_score']:.4f} "
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
        att_str = f" [Attn: {r['attention_weight']:.3f}]" if r.get('attention_weight') is not None else ""
        print(f"    [{i}] {Path(r['file']).name} ({r['duration_seconds']}s): P(PD) = {r['probability_PD']}{att_str} -> {r['recording_status']}")
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
