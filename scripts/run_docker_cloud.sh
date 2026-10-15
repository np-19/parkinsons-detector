#!/usr/bin/env bash
# ==============================================================================
# Launch high-throughput cloud fine-tuning on GridShare / RunPod via pure Docker
# ==============================================================================
set -euo pipefail

DOCKER_BIN="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_BIN="sudo docker"
    fi
fi

echo "======================================================================"
echo " Starting High-Throughput Cloud Training (10 Epochs)"
echo " Profile: batch size 16, accumulation 2, fp16"
echo " Using: ${DOCKER_BIN}"
echo "======================================================================"

mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "$(pwd)/artifacts/models/deep_learning/wav2vec2_finetuned"

echo ">>> Verifying Docker image 'parkinsons-detector:latest'..."
${DOCKER_BIN} build -t parkinsons-detector:latest .

echo ">>> Starting training inside Docker..."
${DOCKER_BIN} run --gpus all --rm \
  --shm-size=8g \
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
    --output-root artifacts/models/deep_learning/wav2vec2_finetuned \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --epochs 10 \
    --learning-rate 2e-5 \
    --head-learning-rate 1e-4 \
    --fp16 \
    --num-frozen-layers 6 \
    --num-workers 4 \
    --patience 4
