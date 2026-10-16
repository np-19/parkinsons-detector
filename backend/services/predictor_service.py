"""Inference service wrapping the PatientPredictor engine."""
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from backend.config import DEVICE, MODEL_PATH
from backend.schemas import (
    MetaStatsResponse,
    PredictionResponse,
    RecordingResult,
    StatisticalDistribution,
)
from src.deep_learning.predict_patient import (
    META_BIAS,
    META_MEANS,
    META_STDS,
    META_THRESHOLD,
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

        logger.info("Initializing PatientPredictor from %s (device: %s)...", self.model_path, self.device)
        self.predictor = PatientPredictor(
            model_path=self.model_path,
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

            diagnosis = (
                "POSITIVE (Parkinsonian Dysphonia Detected)"
                if raw_result["is_parkinsons"]
                else "NEGATIVE (Normal Phonation Dynamics)"
            )

            return PredictionResponse(
                diagnosis=diagnosis,
                is_pd=raw_result["is_parkinsons"],
                confidence_level=raw_result["confidence_level"],
                meta_classifier_score=raw_result["meta_classifier_score"],
                threshold=raw_result["meta_threshold"],
                recordings_analyzed=raw_result["recordings_analyzed"],
                distribution=distribution,
                recordings=recordings,
                clinical_warnings=raw_result["clinical_warnings"],
            )

        finally:
            # Clean up temporary directory and all files
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def get_meta_stats() -> MetaStatsResponse:
        """Return reference statistical parameters used by the clinical Meta-Classifier."""
        return MetaStatsResponse(
            threshold=META_THRESHOLD,
            weights=META_WEIGHTS,
            reference_means=META_MEANS,
            reference_stds=META_STDS,
            bias=META_BIAS,
        )

