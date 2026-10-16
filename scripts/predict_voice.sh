#!/usr/bin/env bash
# ==============================================================================
# Run Parkinson's Disease Inference on one or more audio files via Docker
# ==============================================================================
set -euo pipefail

DOCKER_BIN="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_BIN="sudo docker"
    fi
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_audio1.wav> [path_to_audio2.wav ...]"
    exit 1
fi

${DOCKER_BIN} run --gpus all --rm \
  --shm-size=4g \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$(pwd):/workspace:z" \
  -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface:z" \
  parkinsons-detector:latest \
  python src/deep_learning/predict_patient.py "$@"

