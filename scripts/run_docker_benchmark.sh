#!/usr/bin/env bash
# ==============================================================================
# Quick 100-sample benchmark on local GTX 1650 to estimate full training time
# Uses pure 'docker run' with SELinux ':z' flags and explicit PYTHONPATH
# ==============================================================================
set -euo pipefail

# Check if docker needs sudo
DOCKER_BIN="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_BIN="sudo docker"
    fi
fi

echo "======================================================================"
echo " Starting 100-Sample Benchmark on GTX 1650"
echo " (Trains 1 epoch on 100 audio files to measure exact speed & VRAM)"
echo " Using: ${DOCKER_BIN}"
echo "======================================================================"

# Ensure directories exist
mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "$(pwd)/artifacts/models/deep_learning/benchmark_test"

echo ">>> Verifying Docker image 'parkinsons-detector:latest'..."
${DOCKER_BIN} build -t parkinsons-detector:latest .

echo ">>> Cleaning any stale bytecode cache..."
${DOCKER_BIN} run --rm -v "$(pwd)/src:/workspace/src:z" parkinsons-detector:latest rm -rf /workspace/src/deep_learning/__pycache__ /workspace/src/__pycache__ 2>/dev/null || true

echo ">>> Running 100-sample benchmark on GTX 1650..."
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
  python -m src.deep_learning.train \
    --data-root data/processed_audio \
    --metadata-root data/metadata \
    --output-root artifacts/models/deep_learning/benchmark_test \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 1 \
    --max-train-samples 100 \
    --max-val-samples 30 \
    --learning-rate 2e-5 \
    --head-learning-rate 1e-4 \
    --fp16 \
    --gradient-checkpointing \
    --num-frozen-layers 6 \
    --num-workers 2
