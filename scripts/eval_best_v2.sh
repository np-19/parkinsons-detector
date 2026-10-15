#!/usr/bin/env bash
# ==============================================================================
# Script to immediately evaluate the saved Epoch 6 model with 5-crop TTA & ensemble
# Runs in ~2-3 minutes!
# ==============================================================================
set -euo pipefail

cd /workspace/parkinsons-detector

echo "=== 1. Evaluating Epoch 6 Best Checkpoint with 5-Crop TTA ==="
python -m src.deep_learning.evaluate \
  --model-dir artifacts/models/deep_learning/wav2vec2_v2 \
  --test-metadata data/metadata/test.csv \
  --audio-root data/processed_audio \
  --duration 8.0 \
  --num-crops 5 \
  --batch-size 16

echo "=== 2. Running Ensemble Fusion (v1 + v2) ==="
if [ -f "artifacts/models/deep_learning/wav2vec2_finetuned/test_predictions.csv" ]; then
  python -m src.deep_learning.ensemble \
    --prediction-files \
      artifacts/models/deep_learning/wav2vec2_finetuned/test_predictions.csv \
      artifacts/models/deep_learning/wav2vec2_v2/test_predictions.csv \
    --output-file artifacts/models/deep_learning/ensemble_v1_v2_predictions.csv
fi

echo "======================================================================"
echo " EVALUATION & ENSEMBLE COMPLETE!"
echo "======================================================================"

