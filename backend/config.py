"""Backend configuration and environment settings."""
import os
from pathlib import Path
from typing import List

# Base workspace directory (points to root of repository)
BASE_DIR = Path(__file__).resolve().parent.parent

# Model weights path (WaveLM v4 best model)
DEFAULT_WAVELM_PATH = str(BASE_DIR / "artifacts/models/deep_learning/wavelm_v4/best_model.pt")
DEFAULT_WAV2VEC2_PATH = str(BASE_DIR / "artifacts/models/deep_learning/wav2vec2_v2/best_model.pt")

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_WAVELM_PATH)
SECONDARY_MODEL_PATH = os.getenv("SECONDARY_MODEL_PATH", None)
USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "false").lower() in ("true", "1", "yes")
ENSEMBLE_THRESHOLD = float(os.getenv("ENSEMBLE_THRESHOLD", "0.310"))

# Inference hardware configuration
DEVICE = os.getenv("DEVICE", None)  # None = auto-detect CUDA if available, else CPU

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Allowed origins for CORS (React / Next.js frontend)
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*",
]

# Supported audio formats
ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".m4a",
    ".aac",
    ".flac",
    ".webm",
}

# Maximum upload size per file (in bytes) - 25 MB
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024

