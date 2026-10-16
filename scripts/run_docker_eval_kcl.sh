#!/usr/bin/env bash
# ==============================================================================
# Run evaluation on King's College London (KCL) Speech Dataset via Docker
# ==============================================================================
set -euo pipefail

DOCKER_BIN="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_BIN="sudo docker"
    fi
fi

DATASET_DIR="${1:-test_samples/26-29_09_2017_KCL}"
MODEL_PATH="${2:-artifacts/models/deep_learning/wav2vec2_v2/best_model.pt}"
CROPS="${3:-10}"
DURATION="${4:-8.0}"

echo "======================================================================"
echo " Evaluating Model on King's College London (KCL) Dataset"
echo " Dataset:   ${DATASET_DIR}"
echo " Model:     ${MODEL_PATH}"
echo " Crops:     ${CROPS} windows per recording | Window: ${DURATION}s"
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
  python scripts/evaluate_kcl_dataset.py \
    --dataset-dir "${DATASET_DIR}" \
    --model-path "${MODEL_PATH}" \
    --crops "${CROPS}" \
    --duration "${DURATION}"

