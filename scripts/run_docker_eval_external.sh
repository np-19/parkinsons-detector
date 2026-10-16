#!/usr/bin/env bash
# ==============================================================================
# Run evaluation on external voice dataset (test_samples/Voice_Dataset) via Docker
# ==============================================================================
set -euo pipefail

DOCKER_BIN="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_BIN="sudo docker"
    fi
fi

DATASET_DIR="${1:-test_samples/Voice_Dataset}"
MODEL_PATH="${2:-artifacts/models/deep_learning/wav2vec2_v2/best_model.pt}"
NUM_CROPS="${3:-5}"
DURATION="${4:-3.5}"

echo "======================================================================"
echo " Evaluating Model on External Voice Dataset"
echo " Dataset:   ${DATASET_DIR}"
echo " Model:     ${MODEL_PATH}"
echo " TTA Crops: ${NUM_CROPS} | Duration: ${DURATION}s"
echo " Using:     ${DOCKER_BIN}"
echo "======================================================================"

${DOCKER_BIN} run --gpus all --rm \
  --shm-size=4g \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$(pwd)/test_samples:/workspace/test_samples:ro,z" \
  -v "$(pwd)/artifacts:/workspace/artifacts:z" \
  -v "$(pwd)/src:/workspace/src:z" \
  -v "$(pwd)/scripts:/workspace/scripts:z" \
  -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface:z" \
  parkinsons-detector:latest \
  python scripts/evaluate_external_dataset.py \
    --dataset-dir "${DATASET_DIR}" \
    --model-path "${MODEL_PATH}" \
    --num-crops "${NUM_CROPS}" \
    --duration "${DURATION}"

