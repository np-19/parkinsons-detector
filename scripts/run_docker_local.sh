#!/usr/bin/env bash
# ==============================================================================
# Launch local fine-tuning on NVIDIA GTX 1650 (4GB VRAM) via pure Docker
# ==============================================================================
set -euo pipefail

DOCKER_BIN="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER_BIN="sudo docker"
    fi
fi

echo "======================================================================"
echo " Starting Full Local Training on GTX 1650 (10 Epochs)"
echo " Profile: fp16 + gradient checkpointing (batch 4, accumulation 8)"
echo " Using: ${DOCKER_BIN}"
echo "======================================================================"

mkdir -p "${HOME}/.cache/huggingface"
mkdir -p "$(pwd)/artifacts/models/deep_learning/wav2vec2_finetuned"

echo ">>> Verifying Docker image 'parkinsons-detector:latest'..."
${DOCKER_BIN} build -t parkinsons-detector:latest .

echo ">>> Starting training inside Docker..."
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
    --output-root artifacts/models/deep_learning/wav2vec2_finetuned \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --epochs 10 \
    --learning-rate 2e-5 \
    --head-learning-rate 1e-4 \
    --fp16 \
    --gradient-checkpointing \
    --num-frozen-layers 6 \
    --num-workers 2 \
    --patience 4
