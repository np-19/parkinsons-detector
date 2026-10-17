"""Pydantic data models for request and response serialization."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class RecordingResult(BaseModel):
    """Evaluation result for an individual audio recording session."""
    filename: str = Field(..., description="Original name of the uploaded audio file")
    duration_seconds: float = Field(..., description="Duration of the audio recording in seconds")
    probability_pd: float = Field(..., ge=0.0, le=1.0, description="Raw deep learning acoustic probability of Parkinson's")
    recording_status: str = Field(
        ...,
        description="Local status using the checkpoint-selected acoustic threshold",
    )


class FailedRecording(BaseModel):
    """Audio file that could not be evaluated."""
    filename: str
    error: str


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
    diagnosis: str = Field(..., description="'NEGATIVE (Normal Phonation Dynamics)', 'BORDERLINE (Mild Dysphonia)', or 'POSITIVE (Parkinsonian Dysphonia)'")
    is_pd: bool = Field(..., description="Boolean flag: True if positive/borderline screening alert, False if negative")
    triage_level: str = Field(default="LOW_RISK", description="'LOW_RISK', 'BORDERLINE', or 'HIGH_RISK'")
    triage_recommendation: str = Field(default="", description="Actionable clinical next steps and follow-up guidance")
    confidence_level: str = Field(..., description="Stratified clinical confidence tier")
    participant_mean_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Arithmetic mean of recording-level Parkinson's probabilities",
    )
    meta_classifier_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Deprecated compatibility alias for participant_mean_score",
    )
    threshold: float = Field(default=0.36, description="Balanced diagnostic cutoff threshold")
    screening_cutoff: float = Field(default=0.26, description="Screening triage cutoff derived from the checkpoint threshold")
    diagnostic_cutoff: float = Field(default=0.46, description="Diagnostic triage cutoff derived from the checkpoint threshold")
    recordings_analyzed: int = Field(..., description="Count of recordings analyzed")
    distribution: StatisticalDistribution = Field(..., description="Statistical distribution metrics")
    recordings: List[RecordingResult] = Field(..., description="Breakdown of each recording evaluated")
    failed_recordings: List[FailedRecording] = Field(
        default_factory=list,
        description="Uploaded recordings rejected before model inference",
    )
    clinical_warnings: List[str] = Field(default_factory=list, description="Clinical advisories and pharmacological masking warnings")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class MetaStatsResponse(BaseModel):
    """Threshold and aggregation configuration used by patient inference."""
    threshold: float = 0.36
    screening_cutoff: float = 0.26
    diagnostic_cutoff: float = 0.46
    aggregation: str = "arithmetic_mean"
    score_field: str = "participant_mean_score"
    model_version: str = "Wav2Vec2_v2 (mPower)"
    # Deprecated compatibility fields retained for existing API consumers.
    weights: List[float] = Field(default_factory=list)
    reference_means: List[float] = Field(default_factory=list)
    reference_stds: List[float] = Field(default_factory=list)
    bias: float = 0.0


class HealthResponse(BaseModel):
    """Healthcheck and runtime environment status."""
    status: str = "healthy"
    model_loaded: bool
    model_version: str = "Wav2Vec2_v2 (mPower)"
    device: str
    torch_version: str
    supported_formats: List[str]
