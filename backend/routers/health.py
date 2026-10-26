"""Health and system status router."""
import torch
from fastapi import APIRouter

from backend.config import ALLOWED_EXTENSIONS
from backend.schemas import HealthResponse, MetaStatsResponse
from backend.services.predictor_service import PredictorService

router = APIRouter(tags=["System & Health"])


@router.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return backend health, PyTorch version, and GPU/CPU device status."""
    service = PredictorService.get_instance()
    is_mil = bool(service.predictor and getattr(service.predictor, "mil_model", None) is not None)
    model_ver = (
        "WavLM + Gated Attention-MIL (mPower)"
        if is_mil
        else "WavLM (mPower)"
    )
    return HealthResponse(
        status="healthy",
        model_loaded=service.is_loaded(),
        model_version=model_ver,
        device=service.get_device_name(),
        torch_version=torch.__version__,
        supported_formats=sorted(list(ALLOWED_EXTENSIONS)),
    )


@router.get("/api/v1/meta-stats", response_model=MetaStatsResponse)
def get_meta_stats() -> MetaStatsResponse:
    """Return clinical Meta-Classifier weights, reference means, stds, and cutoff threshold."""
    return PredictorService.get_meta_stats()

