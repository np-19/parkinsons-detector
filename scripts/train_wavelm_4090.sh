#!/usr/bin/env bash
# ==============================================================================
# NeuroVocal: WaveLM Training & Evaluation Runner for NVIDIA RTX 4090 (24 GB)
# Designed for AIC Cloud / RunPod / Lambda / Vast.ai
# Expected training time: ~30-35 minutes for 20 epochs
# ==============================================================================
set -euo pipefail

# Ensure working directory is repo root
cd "$(dirname "$0")/.."

echo "======================================================================"
echo "          NEUROVOCAL: WAVELM TRAINING ON NVIDIA RTX 4090              "
echo "======================================================================"

# --- 1. Hardware Inspection ---
echo ">>> Checking GPU hardware..."
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "[WARNING] nvidia-smi not found. Proceeding with PyTorch CUDA checks."
fi

# --- 2. Dependency Verification ---
echo -e "\n>>> Verifying Python dependencies..."
python3 -c "
import torch
print(f'PyTorch Version  : {torch.__version__}')
print(f'CUDA Available   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device       : {torch.cuda.get_device_name(0)}')
    print(f'VRAM Total       : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
"

# Install missing audio/deep learning dependencies if required
python3 -c "import soundfile, transformers, accelerate" 2>/dev/null || {
  echo "Installing required packages (transformers, accelerate, soundfile)..."
  pip install --no-cache-dir soundfile accelerate "transformers>=4.46.0"
}

# --- 3. Directory Setup ---
OUTPUT_DIR="artifacts/models/deep_learning/wavelm_v1"
mkdir -p "$OUTPUT_DIR"

# --- 4. Launch WaveLM Fine-Tuning ---
echo -e "\n======================================================================"
echo " STAGE 1: Fine-Tuning WaveLM (microsoft/wavelm-base-plus) on mPower"
echo " Configuration:"
echo "   - Backbone        : microsoft/wavelm-base-plus (Pretrained on 60,000h speech)"
echo "   - Window Duration : 8.0 seconds (128,000 samples @ 16 kHz)"
echo "   - Batch Size      : 16 (Gradient Acc: 2 -> Effective Batch Size: 32)"
echo "   - Trainable Layers: Transformer Layers 5-12 (Layers 1-4 frozen, CNN frozen)"
echo "   - Regularization  : Mixup (alpha=0.2), Dropout (0.3), Label Smoothing (0.1)"
echo "   - Class Imbalance : Pos-Weight 2.5 (compensates for 1:3 PD-to-control ratio)"
echo "   - Acceleration    : PyTorch AMP FP16 + Cosine Warmup Schedule"
echo "======================================================================"

python3 -m src.deep_learning.train \
  --data-root data/processed_audio \
  --metadata-root data/metadata \
  --output-root "$OUTPUT_DIR" \
  --model-name microsoft/wavelm-base-plus \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --epochs 20 \
  --patience 7 \
  --learning-rate 2.5e-5 \
  --head-learning-rate 1.5e-4 \
  --warmup-ratio 0.1 \
  --duration 8.0 \
  --num-frozen-layers 4 \
  --dropout 0.3 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.2 \
  --pos-weight 2.5 \
  --fp16 \
  --num-workers 4

# --- 5. Evaluate on Held-Out Test Split ---
echo -e "\n======================================================================"
echo " STAGE 2: Evaluating WaveLM on Held-Out Test Split (579 Participants)"
echo "======================================================================"

python3 -m src.deep_learning.evaluate \
  --model-dir "$OUTPUT_DIR" \
  --test-metadata data/metadata/test.csv \
  --audio-root data/processed_audio \
  --duration 8.0 \
  --batch-size 16 \
  --device cuda \
  --fp16

# --- 6. Zero-Shot Cross-Dataset Validation on Clinical 'records' Cohort ---
echo -e "\n======================================================================"
echo " STAGE 3: Zero-Shot Multi-Cohort Clinical Validation ('records' Dataset)"
echo "======================================================================"

if [ -f "test_samples/records/dataset.csv" ]; then
  python3 scripts/evaluate_records_dataset.py \
    --model-path "$OUTPUT_DIR/best_model.pt" \
    --dataset-csv test_samples/records/dataset.csv \
    --audio-dir test_samples/records \
    --output-csv artifacts/wavelm_records_evaluation_results.csv \
    --device cuda
else
  echo "Clinical records dataset not found at test_samples/records/dataset.csv. Skipping Stage 3."
fi

# --- 7. Multi-Backbone Ensemble Fusion (WaveLM + Wav2Vec2) ---
WAV2VEC2_PRED="artifacts/models/deep_learning/wav2vec2_v2/test_predictions.csv"
WAVELM_PRED="$OUTPUT_DIR/test_predictions.csv"

if [ -f "$WAV2VEC2_PRED" ] && [ -f "$WAVELM_PRED" ]; then
  echo -e "\n======================================================================"
  echo " STAGE 4: Multi-Backbone Ensemble Fusion (WaveLM 60% + Wav2Vec2 40%)"
  echo "======================================================================"
  python3 -m src.deep_learning.ensemble \
    --prediction-files "$WAVELM_PRED" "$WAV2VEC2_PRED" \
    --weights 0.6 0.4 \
    --output-file artifacts/models/deep_learning/ensemble_wavelm_wav2vec2_predictions.csv
fi

echo -e "\n======================================================================"
echo " [SUCCESS] WAVELM 4090 CLOUD PIPELINE COMPLETE!"
echo " Checkpoint saved to: $OUTPUT_DIR/best_model.pt"
echo " Test Report saved to: $OUTPUT_DIR/test_evaluation_report.json"
echo "======================================================================"
