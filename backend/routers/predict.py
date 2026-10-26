"""Prediction endpoints for single and multi-session audio evaluation."""
import asyncio
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES
from backend.schemas import PredictionResponse
from backend.services.predictor_service import PredictorService

logger = logging.getLogger("backend.routers.predict")
router = APIRouter(prefix="/api/v1", tags=["Inference & Diagnostics"])


def _validate_file(file: UploadFile) -> str:
    """Validate audio file extension and return normalized filename."""
    filename = file.filename or "recording.wav"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio format '{ext}' for file '{filename}'. "
                f"Supported formats: {sorted(list(ALLOWED_EXTENSIONS))}"
            ),
        )
    return filename


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Evaluate One or More Patient Voice Recordings",
    description=(
        "Upload one or more audio files (sustained vowel 'aaah'). Evaluates each file using the fine-tuned "
        "Microsoft WavLM Base+ acoustic model with 3-crop multi-crop embedding extraction and Gated "
        "Attention-MIL aggregation with 50:50 blend and clinical consensus guardrail to produce the "
        "final participant-level screening triage."
    ),
)
async def predict_recordings(
    files: List[UploadFile] = File(
        ...,
        description="One or more sustained phonation audio files (.wav, .m4a, .ogg, .mpeg, .webm)",
    ),
) -> PredictionResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one audio file must be uploaded for assessment.",
        )

    file_tuples = []
    for f in files:
        filename = _validate_file(f)
        content = await f.read()

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{filename}' is empty (0 bytes).",
            )

        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{filename}' exceeds maximum allowed size of {max_mb} MB.",
            )

        file_tuples.append((filename, content))

    service = PredictorService.get_instance()
    try:
        # Offload tensor computation and model forward passes to thread pool
        result = await asyncio.to_thread(service.predict_audio_files, file_tuples)
        return result
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Acoustic inference engine error: {str(e)}",
        )


@router.post(
    "/predict/single",
    response_model=PredictionResponse,
    summary="Evaluate a Single Voice Recording",
    description="Convenience endpoint to upload a single audio file for acoustic evaluation.",
)
async def predict_single_recording(
    file: UploadFile = File(
        ...,
        description="Single sustained phonation audio file (.wav, .m4a, .ogg, .mpeg, .webm)",
    ),
) -> PredictionResponse:
    return await predict_recordings(files=[file])
