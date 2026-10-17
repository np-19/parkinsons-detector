#!/usr/bin/env bash
# ==============================================================================
# Unified Docker Compose Runner for NeuroVocal System
# Default mode is CPU. Supports GPU with automatic fallback to CPU:
#   ./scripts/run_docker.sh        # Runs in default CPU mode
#   ./scripts/run_docker.sh cpu    # Runs in CPU mode
#   ./scripts/run_docker.sh gpu    # Runs on GPU (falls back to CPU if unavailable)
#   ./scripts/run_docker.sh down   # Stops all containers
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Detect Docker binary (with sudo if non-root and not in docker group)
DOCKER_COMPOSE="docker compose"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_COMPOSE="sudo docker compose"
    fi
fi

# Parse argument (default is CPU)
ARG="${1:-cpu}"
ARG_LOWER="$(echo "${ARG}" | tr '[:upper:]' '[:lower:]' | sed 's/--//')"

if [ "${ARG_LOWER}" = "down" ] || [ "${ARG_LOWER}" = "stop" ]; then
    echo ">>> Stopping all NeuroVocal Docker containers..."
    ${DOCKER_COMPOSE} --profile gpu --profile cpu down
    exit 0
fi

case "${ARG_LOWER}" in
  gpu|cuda)
    # Check if NVIDIA GPU and driver are accessible
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        PROFILE="gpu"
        MODE_DESC="NVIDIA GPU (CUDA mode)"
    else
        echo "[WARNING] GPU mode requested, but NVIDIA GPU / CUDA driver is not accessible. Falling back to CPU mode."
        PROFILE="cpu"
        MODE_DESC="CPU Mode (Fallback from GPU)"
    fi
    ;;
  cpu|"")
    PROFILE="cpu"
    MODE_DESC="CPU Mode (Default)"
    ;;
  *)
    echo "Usage: $0 [cpu|gpu|down]"
    echo "  $0        # Runs in default CPU mode"
    echo "  $0 cpu    # Explicitly run in CPU mode"
    echo "  $0 gpu    # Run on GPU (with fallback to CPU if unavailable)"
    echo "  $0 down   # Stop all services"
    exit 1
    ;;
esac

echo "======================================================================"
echo " Starting NeuroVocal Full Stack via Docker Compose"
echo " Hardware: ${MODE_DESC}"
echo " Profile:  --profile ${PROFILE}"
echo " Frontend: http://localhost:3000"
echo " Backend:  http://localhost:8000"
echo " API Docs: http://localhost:8000/docs"
echo "======================================================================"

mkdir -p "${HOME}/.cache/huggingface"

${DOCKER_COMPOSE} --profile "${PROFILE}" up -d --build

echo ""
echo ">>> Services started successfully!"
echo "    View logs with: ${DOCKER_COMPOSE} logs -f"
echo "    Stop with:      $0 down"
