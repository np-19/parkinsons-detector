#!/usr/bin/env bash
# ==============================================================================
# Launch NeuroVocal FastAPI Backend Service on CPU (Zero GPU Required)
# ==============================================================================
set -euo pipefail

DOCKER_BIN="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_BIN="sudo docker"
    fi
fi

echo "======================================================================"
echo " Starting NeuroVocal Backend on CPU Mode (No GPU Required)"
echo " API Docs:    http://localhost:8000/docs"
echo " Healthcheck: http://localhost:8000/api/health"
echo " Predict API: http://localhost:8000/api/v1/predict"
echo " Using:       ${DOCKER_BIN}"
echo "======================================================================"

mkdir -p "${HOME}/.cache/huggingface"

echo ">>> Building backend docker image 'parkinsons-detector-backend:latest'..."
${DOCKER_BIN} build -f backend/Dockerfile -t parkinsons-detector-backend:latest .

echo ">>> Starting CPU backend container on port 8000..."
${DOCKER_BIN} run --rm -it \
  --name parkinsons-backend \
  -p 8000:8000 \
  --shm-size=4g \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e DEVICE=cpu \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$(pwd)/artifacts:/workspace/artifacts:ro,z" \
  -v "$(pwd)/src:/workspace/src:ro,z" \
  -v "$(pwd)/backend:/workspace/backend:z" \
  -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface:z" \
  parkinsons-detector-backend:latest
