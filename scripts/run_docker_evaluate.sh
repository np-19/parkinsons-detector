#!/usr/bin/env bash
# ==============================================================================
# Run one-time evaluation on held-out test split via pure Docker
# ==============================================================================
set -euo pipefail

DOCKER_BIN="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_BIN="sudo docker"
    fi
fi

MODEL_DIR="${1:-artifacts/models/deep_learning/wav2vec2_v2}"
NUM_CROPS="${2:-5}"
DURATION="${3:-8.0}"

echo "======================================================================"
echo " Evaluating Frozen Best Checkpoint on Held-Out Test Set"
echo " Model Directory: ${MODEL_DIR}"
echo " TTA Crops: ${NUM_CROPS} | Duration: ${DURATION}s"
echo " Using: ${DOCKER_BIN}"
echo "======================================================================"

${DOCKER_BIN} run --gpus all --rm \
  --shm-size=4g \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$(pwd)/data:/workspace/data:ro,z" \
  -v "$(pwd)/artifacts:/workspace/artifacts:z" \
  -v "$(pwd)/src:/workspace/src:z" \
  -v "$(pwd)/scripts:/workspace/scripts:z" \
  -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface:z" \
  parkinsons-detector:latest \
  python -m src.deep_learning.evaluate \
    --model-dir "${MODEL_DIR}" \
    --test-metadata data/metadata/test.csv \
    --audio-root data/processed_audio \
    --duration "${DURATION}" \
    --num-crops "${NUM_CROPS}" \
    --batch-size 8
