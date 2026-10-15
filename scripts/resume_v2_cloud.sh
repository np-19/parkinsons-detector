#!/usr/bin/env bash
# ==============================================================================
# Script to resume training from the best checkpoint (Epoch 6) on GridShare
# ==============================================================================
set -euo pipefail

cd /workspace/parkinsons-detector

echo "=== Resuming Wav2Vec2 v2 Training from Best Checkpoint ==="
python -m src.deep_learning.train \
  --data-root data/processed_audio \
  --metadata-root data/metadata \
  --output-root artifacts/models/deep_learning/wav2vec2_v2 \
  --resume-from artifacts/models/deep_learning/wav2vec2_v2/best_model.pt \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --epochs 25 \
  --patience 6 \
  --learning-rate 3e-5 \
  --head-learning-rate 1.5e-4 \
  --warmup-ratio 0.1 \
  --duration 8.0 \
  --num-frozen-layers 4 \
  --dropout 0.3 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.2 \
  --fp16 \
  --num-workers 4

echo "=== Evaluating Resumed Model with 5-Crop Test-Time Augmentation ==="
python -m src.deep_learning.evaluate \
  --model-dir artifacts/models/deep_learning/wav2vec2_v2 \
  --test-metadata data/metadata/test.csv \
  --audio-root data/processed_audio \
  --duration 8.0 \
  --num-crops 5 \
  --batch-size 16

echo "=== Running Ensemble Fusion (v1 + v2) ==="
if [ -f "artifacts/models/deep_learning/wav2vec2_finetuned/test_predictions.csv" ]; then
  python -m src.deep_learning.ensemble \
    --prediction-files \
      artifacts/models/deep_learning/wav2vec2_finetuned/test_predictions.csv \
      artifacts/models/deep_learning/wav2vec2_v2/test_predictions.csv \
    --output-file artifacts/models/deep_learning/ensemble_v1_v2_predictions.csv
fi

echo "======================================================================"
echo " RESUMED TRAINING & EVALUATION COMPLETE!"
echo "======================================================================"

