#!/usr/bin/env bash
# ==============================================================================
# Launch NeuroVocal FastAPI Backend Locally on Host (NO DOCKER)
# Default mode is CPU. Supports GPU with automatic fallback to CPU:
#   ./scripts/run_backend_local.sh        # Runs in default CPU mode
#   ./scripts/run_backend_local.sh cpu    # Runs in CPU mode
#   ./scripts/run_backend_local.sh gpu    # Runs on GPU (falls back to CPU if unavailable)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Setup Python paths including data/.venv packages
VENV_PACKAGES="${REPO_ROOT}/data/.venv/lib/python3.14/site-packages"
export PYTHONPATH="${REPO_ROOT}:${VENV_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Parse command line argument (default is CPU)
MODE="${1:-cpu}"
MODE_LOWER="$(echo "${MODE}" | tr '[:upper:]' '[:lower:]' | sed 's/--//')"

case "${MODE_LOWER}" in
  gpu|cuda)
    # Check if GPU is actually available in PyTorch
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        export DEVICE="cuda"
        GPU_NAME="$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'NVIDIA GPU')"
        HARDWARE_TAG="${GPU_NAME} (GPU mode)"
    else
        echo "[WARNING] GPU mode requested, but CUDA is not available. Falling back to CPU mode."
        export DEVICE="cpu"
        HARDWARE_TAG="CPU Mode (Fallback from GPU)"
    fi
    ;;
  cpu|"")
    export DEVICE="cpu"
    HARDWARE_TAG="CPU Mode (Default)"
    ;;
  *)
    echo "Usage: $0 [cpu|gpu]"
    echo "  $0        # Runs in default CPU mode"
    echo "  $0 cpu    # Explicitly run on CPU"
    echo "  $0 gpu    # Run on GPU (with fallback to CPU if unavailable)"
    exit 1
    ;;
esac

echo "======================================================================"
echo " Starting NeuroVocal Clinical Decision Support Backend (LOCAL)"
echo " Hardware:    ${HARDWARE_TAG}"
echo " Device Flag: DEVICE=${DEVICE}"
echo " API Docs:    http://localhost:${PORT}/docs"
echo " Healthcheck: http://localhost:${PORT}/api/health"
echo " Predict API: http://localhost:${PORT}/api/v1/predict"
echo "======================================================================"

exec python3 -m uvicorn backend.main:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --reload
