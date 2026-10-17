"""Inference service wrapping the PatientPredictor engine."""
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from backend.config import (
    DEVICE,
    ENSEMBLE_THRESHOLD,
    MODEL_PATH,
    SECONDARY_MODEL_PATH,
    USE_ENSEMBLE,
)
from backend.schemas import (
    MetaStatsResponse,
    FailedRecording,
    PredictionResponse,
    RecordingResult,
    StatisticalDistribution,
)
from src.deep_learning.predict_patient import (
    ACOUSTIC_OPTIMAL_THRESHOLD,
    META_BIAS,
    META_MEANS,
    META_STDS,
    META_WEIGHTS,
    PatientPredictor,
)

logger = logging.getLogger("backend.predictor_service")


class PredictorService:
    """Thread-safe singleton managing model lifecycle and audio inference."""

    _instance: Optional["PredictorService"] = None

    def __init__(self):
        self.predictor: Optional[PatientPredictor] = None
        self.model_path = MODEL_PATH
        self.secondary_model_path = SECONDARY_MODEL_PATH
        self.use_ensemble = USE_ENSEMBLE
        self.threshold = ENSEMBLE_THRESHOLD
        self.device = DEVICE

    @classmethod
    def get_instance(cls) -> "PredictorService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """Load model weights into memory and GPU VRAM."""
        if self.predictor is not None:
            logger.info("Predictor is already initialized.")
            return

        logger.info(
            "Initializing PatientPredictor (WaveLM best model, ensemble=%s, device=%s)...",
            self.use_ensemble,
            self.device,
        )
        self.predictor = PatientPredictor(
            model_path=self.model_path,
            secondary_model_path=self.secondary_model_path if self.use_ensemble else None,
            use_ensemble=self.use_ensemble,
            threshold=self.threshold,
            device=self.device,
            duration=8.0,
            sample_rate=16000,
            num_crops=5,
        )
        logger.info("PatientPredictor initialized successfully on %s.", self.predictor.device)

    def is_loaded(self) -> bool:
        return self.predictor is not None

    def get_device_name(self) -> str:
        if self.predictor is None:
            return "not_loaded"
        return str(self.predictor.device)

    def predict_audio_files(self, files: List[Tuple[str, bytes]]) -> PredictionResponse:
        """Run patient prediction on a list of (filename, file_bytes) tuples."""
        if self.predictor is None:
            raise RuntimeError("Inference engine is not initialized. Please call initialize() first.")

        if not files:
            raise ValueError("At least one audio file is required for assessment.")

        # Create isolated temporary directory for this request
        temp_dir = tempfile.mkdtemp(prefix="parkinsons_inference_")
        temp_paths: List[Path] = []
        original_filenames: Dict[str, str] = {}

        try:
            for idx, (filename, content) in enumerate(files):
                ext = Path(filename).suffix.lower() or ".wav"
                safe_name = f"take_{idx}{ext}"
                temp_path = Path(temp_dir) / safe_name
                with open(temp_path, "wb") as f:
                    f.write(content)
                temp_paths.append(temp_path)
                original_filenames[str(temp_path)] = filename

            # Execute model forward passes & meta-classifier
            raw_result = self.predictor.predict_patient(temp_paths)

            # Map raw output back to original filenames
            recordings: List[RecordingResult] = []
            for rec in raw_result["recordings_detail"]:
                orig_name = original_filenames.get(rec["file"], Path(rec["file"]).name)
                recordings.append(
                    RecordingResult(
                        filename=orig_name,
                        duration_seconds=rec["duration_seconds"],
                        probability_pd=rec["probability_PD"],
                        recording_status=rec["recording_status"],
                    )
                )

            dist_summary = raw_result["distribution_summary"]
            distribution = StatisticalDistribution(
                mean_probability=dist_summary["mean_probability"],
                top2_probability=dist_summary["top2_probability"],
                max_probability=dist_summary["max_probability"],
                min_probability=dist_summary["min_probability"],
                std_deviation=dist_summary["std_deviation"],
                recordings_analyzed=raw_result["recordings_analyzed"],
            )

            score = raw_result["meta_classifier_score"]
            screening_cutoff = raw_result.get("screening_cutoff", 0.260)
            diagnostic_cutoff = raw_result.get("diagnostic_cutoff", 0.460)
            balanced_cutoff = raw_result.get("meta_threshold", 0.360)

            triage_level = raw_result.get("triage_level", "LOW_RISK")
            diagnosis = raw_result.get("participant_verdict", "NEGATIVE (Normal Acoustic Biomarkers)")
            is_pd = raw_result.get("is_parkinsons", False)
            confidence_level = raw_result.get("confidence_level", "Normal Voice Dynamics")
            triage_recommendation = raw_result.get("triage_recommendation", "")
            failed_recordings = [
                FailedRecording(
                    filename=original_filenames.get(item["file"], Path(item["file"]).name),
                    error=item["error"],
                )
                for item in raw_result.get("failed_recordings", [])
            ]

            return PredictionResponse(
                diagnosis=diagnosis,
                is_pd=is_pd,
                triage_level=triage_level,
                triage_recommendation=triage_recommendation,
                confidence_level=confidence_level,
                participant_mean_score=score,
                meta_classifier_score=score,
                threshold=balanced_cutoff,
                screening_cutoff=screening_cutoff,
                diagnostic_cutoff=diagnostic_cutoff,
                recordings_analyzed=raw_result["recordings_analyzed"],
                distribution=distribution,
                recordings=recordings,
                failed_recordings=failed_recordings,
                clinical_warnings=raw_result["clinical_warnings"],
            )

        finally:
            # Clean up temporary directory and all files
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def get_meta_stats() -> MetaStatsResponse:
        """Return the production participant-score aggregation configuration."""
        predictor = PredictorService.get_instance().predictor
        threshold = (
            predictor.optimal_threshold
            if predictor is not None
            else ACOUSTIC_OPTIMAL_THRESHOLD
        )
        screening_cutoff = max(0.15, threshold - 0.10)
        diagnostic_cutoff = min(0.85, threshold + 0.10)
        model_ver = (
            "Option B Ensemble (WaveLM v4 [75%] + Wav2Vec2 v2 [25%])"
            if (predictor and getattr(predictor, "is_ensemble", False))
            else "WaveLM v4 (mPower)"
        )
        return MetaStatsResponse(
            threshold=round(threshold, 3),
            screening_cutoff=round(screening_cutoff, 3),
            diagnostic_cutoff=round(diagnostic_cutoff, 3),
            aggregation="arithmetic_mean",
            score_field="participant_mean_score",
            model_version=model_ver,
            weights=META_WEIGHTS,
            reference_means=META_MEANS,
            reference_stds=META_STDS,
            bias=META_BIAS,
        )
