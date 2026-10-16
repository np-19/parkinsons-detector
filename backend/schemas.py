"""Pydantic data models for request and response serialization."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class RecordingResult(BaseModel):
    """Evaluation result for an individual audio recording session."""
    filename: str = Field(..., description="Original name of the uploaded audio file")
    duration_seconds: float = Field(..., description="Duration of the audio recording in seconds")
    probability_pd: float = Field(..., ge=0.0, le=1.0, description="Raw deep learning acoustic probability of Parkinson's")
    recording_status: str = Field(..., description="Local status: 'Normal Phonation' or 'Dysarthric / Tremor'")


class StatisticalDistribution(BaseModel):
    """Longitudinal statistical moments across multiple recording takes."""
    mean_probability: float = Field(..., description="Arithmetic mean of raw probabilities")
    top2_probability: float = Field(..., description="Average of the two highest severity recordings")
    max_probability: float = Field(..., description="Peak recording probability observed")
    min_probability: float = Field(..., description="Lowest recording probability observed")
    std_deviation: float = Field(..., description="Standard deviation (diurnal motor fluctuation)")
    recordings_analyzed: int = Field(..., description="Total number of valid recordings evaluated")


class PredictionResponse(BaseModel):
    """Comprehensive patient-level clinical acoustic screening report."""
    diagnosis: str = Field(..., description="'NEGATIVE (Normal Phonation Dynamics)' or 'POSITIVE (Parkinsonian Dysphonia Detected)'")
    is_pd: bool = Field(..., description="Boolean flag: True if positive, False if negative")
    confidence_level: str = Field(..., description="Stratified clinical confidence tier")
    meta_classifier_score: float = Field(..., ge=0.0, le=1.0, description="Standardized multi-moment Meta-Classifier score")
    threshold: float = Field(default=0.29, description="Clinical diagnostic decision boundary threshold")
    recordings_analyzed: int = Field(..., description="Count of recordings analyzed")
    distribution: StatisticalDistribution = Field(..., description="Statistical distribution metrics")
    recordings: List[RecordingResult] = Field(..., description="Breakdown of each recording evaluated")
    clinical_warnings: List[str] = Field(default_factory=list, description="Clinical advisories and pharmacological masking warnings")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class MetaStatsResponse(BaseModel):
    """Reference statistics used by the Meta-Classifier."""
    threshold: float = 0.29
    weights: List[float]
    reference_means: List[float]
    reference_stds: List[float]
    bias: float


class HealthResponse(BaseModel):
    """Healthcheck and runtime environment status."""
    status: str = "healthy"
    model_loaded: bool
    model_version: str = "Wav2Vec2_v2 (mPower)"
    device: str
    torch_version: str
    supported_formats: List[str]
