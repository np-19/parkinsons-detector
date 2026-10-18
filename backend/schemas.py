"""Pydantic data models for request and response serialization."""
from datetime import datetime, timezone
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
    attention_weight: Optional[float] = Field(
        default=None,
        description="Relative Gated Attention-MIL importance weight for this recording",
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
        description=(
            "Final participant-level Parkinson's score produced by the production pipeline: "
            "a 50:50 blend of attention-weighted recording probabilities and the Gated "
            "Attention-MIL model probability, bounded by the clinical consensus guardrail."
        ),
    )
    meta_classifier_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Deprecated compatibility alias returning the exact value of participant_mean_score "
            "(the 50:50 Attention-MIL production blended score)."
        ),
    )
    threshold: float = Field(default=0.30, description="Balanced diagnostic cutoff threshold")
    screening_cutoff: float = Field(default=0.20, description="Screening triage cutoff derived from the checkpoint threshold")
    diagnostic_cutoff: float = Field(default=0.40, description="Diagnostic triage cutoff derived from the checkpoint threshold")
    recordings_analyzed: int = Field(..., description="Count of recordings analyzed")
    distribution: StatisticalDistribution = Field(..., description="Statistical distribution metrics")
    recordings: List[RecordingResult] = Field(..., description="Breakdown of each recording evaluated")
    failed_recordings: List[FailedRecording] = Field(
        default_factory=list,
        description="Uploaded recordings rejected before model inference",
    )
    clinical_warnings: List[str] = Field(default_factory=list, description="Clinical advisories and pharmacological masking warnings")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    aggregation_method: str = Field(default="gated_attention_mil", description="Participant-level aggregation method")
    attention_weights: Optional[List[float]] = Field(default=None, description="Per-crop attention weights")


class MetaStatsResponse(BaseModel):
    """Threshold and aggregation configuration used by patient inference."""
    threshold: float = 0.30
    screening_cutoff: float = 0.20
    diagnostic_cutoff: float = 0.50
    aggregation: str = "gated_attention_mil"
    score_field: str = "participant_mean_score"
    model_version: str = "WaveLM + Gated Attention-MIL (mPower)"
    # Deprecated compatibility fields retained for existing API consumers.
    weights: List[float] = Field(default_factory=list)
    reference_means: List[float] = Field(default_factory=list)
    reference_stds: List[float] = Field(default_factory=list)
    bias: float = 0.0


class HealthResponse(BaseModel):
    """Healthcheck and runtime environment status."""
    status: str = "healthy"
    model_loaded: bool
    model_version: str = "WaveLM + Gated Attention-MIL (mPower)"
    device: str
    torch_version: str
    supported_formats: List[str]
