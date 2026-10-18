"""Backend configuration and environment settings."""
import os
from pathlib import Path
from typing import List

# Base workspace directory (points to root of repository)
BASE_DIR = Path(__file__).resolve().parent.parent

# Model weights path (WaveLM champion model + Gated Attention-MIL)
DEFAULT_WAVELM_PATH = str(BASE_DIR / "artifacts/models/deep_learning/wavelm/best_model.pt")
DEFAULT_MIL_PATH = str(BASE_DIR / "artifacts/models/deep_learning/wavelm/mil_head.pt")

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_WAVELM_PATH)
MIL_PATH = os.getenv("MIL_PATH", DEFAULT_MIL_PATH)
USE_ATTENTION_MIL = os.getenv("USE_ATTENTION_MIL", "true").lower() in ("true", "1", "yes")
MIL_THRESHOLD = float(os.getenv("MIL_THRESHOLD", "0.300"))

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

