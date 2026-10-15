#!/usr/bin/env bash
# ==============================================================================
# Ultra-Fast Training Script for NVIDIA RTX 4090 (24 GB VRAM)
# Runs in ~35-45 minutes total!
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== 1. Checking / Installing Dependencies ==="
python3 -c "import soundfile" 2>/dev/null || pip install soundfile "transformers==4.46.3"
echo '__all__: list[str] = []' > /usr/local/lib/python3.11/dist-packages/torch/_dynamo/polyfills/sys.py 2>/dev/null || true

echo "=== 2. Starting RTX 4090 Training (Batch Size 16, 8s Audio) ==="
python3 -m src.deep_learning.train \
  --data-root data/processed_audio \
  --metadata-root data/metadata \
  --output-root artifacts/models/deep_learning/wav2vec2_v2 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --epochs 20 \
  --patience 7 \
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

echo "=== 3. Evaluating with 5-Crop Test-Time Augmentation ==="
python3 -m src.deep_learning.evaluate \
  --model-dir artifacts/models/deep_learning/wav2vec2_v2 \
  --test-metadata data/metadata/test.csv \
  --audio-root data/processed_audio \
  --duration 8.0 \
  --num-crops 5 \
  --batch-size 16

echo "=== 4. Running Ensemble Fusion ==="
if [ -f "artifacts/models/deep_learning/wav2vec2_finetuned/test_predictions.csv" ]; then
  python3 -m src.deep_learning.ensemble \
    --prediction-files \
      artifacts/models/deep_learning/wav2vec2_finetuned/test_predictions.csv \
      artifacts/models/deep_learning/wav2vec2_v2/test_predictions.csv \
    --output-file artifacts/models/deep_learning/ensemble_v1_v2_predictions.csv
fi

echo "======================================================================"
echo " RTX 4090 PIPELINE COMPLETE!"
echo "======================================================================"

