#!/usr/bin/env bash
# ==============================================================================
# NeuroVocal: WavLM Training & Evaluation Runner for Cloud GPUs (24 GB VRAM)
# Fully compatible with:
#   • NVIDIA GeForce RTX 3090 (24 GB) - (~50-60 min runtime)
#   • NVIDIA GeForce RTX 4090 (24 GB) - (~30-35 min runtime)
#   • Any modern Ampere / Ada Lovelace / Hopper 24GB+ GPU (A5000, A6000, A100)
# ==============================================================================
set -euo pipefail

# Ensure working directory is repo root
cd "$(dirname "$0")/.."

echo "======================================================================"
echo "         NEUROVOCAL: WAVELM TRAINING ON CLOUD GPU (24 GB)             "
echo "======================================================================"

# --- 1. Hardware Inspection & Auto-Detection ---
echo ">>> Checking GPU hardware..."
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
  GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
  DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
  echo "Detected GPU : $GPU_NAME"
  echo "VRAM Total   : $GPU_VRAM"
  echo "Driver Ver   : $DRIVER_VER"
else
  echo "[WARNING] nvidia-smi not in PATH. Proceeding with PyTorch CUDA checks."
fi

# --- 2. System Audio Dependencies (libsndfile & ffmpeg) ---
if command -v apt-get &>/dev/null; then
  if ! dpkg -s libsndfile1 &>/dev/null || ! command -v ffmpeg &>/dev/null; then
    echo "Installing system audio libraries (libsndfile1, ffmpeg)..."
    apt-get update -qq && apt-get install -y -qq libsndfile1 ffmpeg || true
  fi
fi

# --- 3. Python Environment & Package Verification ---
echo -e "\n>>> Verifying Python & PyTorch dependencies..."
python3 -c "
import sys
import torch
print(f'Python Version   : {sys.version.split()[0]}')
print(f'PyTorch Version  : {torch.__version__}')
print(f'CUDA Available   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Active Device    : {torch.cuda.get_device_name(0)}')
    print(f'Allocatable VRAM : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
else:
    print('[FATAL] CUDA is not available. Please verify NVIDIA drivers / PyTorch build.')
    sys.exit(1)
"

# Ensure transformers is pinned to 4.46.3 (compatible with PyTorch < 2.5 without false backend errors)
python3 -c "import soundfile, transformers, accelerate, sklearn, pandas; assert transformers.is_torch_available()" 2>/dev/null || {
  echo "Installing/fixing required Python packages (transformers==4.46.3)..."
  pip install --no-cache-dir "transformers==4.46.3" soundfile accelerate scikit-learn pandas tqdm
}

# --- 4. Dataset Directory Verification ---
if [ ! -d "data/processed_audio" ] || [ $(ls -1 data/processed_audio 2>/dev/null | wc -l) -lt 500 ]; then
  echo -e "\n[WARNING] 'data/processed_audio' is missing or contains fewer than 500 audio files!"
  echo "Please upload your 'processed_audio.tar.gz' to 'data/' and extract it before starting:"
  echo "  cd data && tar -xzf processed_audio.tar.gz"
  echo "Abort check: do you want to continue anyway? (Ctrl+C to abort now, continuing in 5s...)"
  sleep 5
fi

# --- 5. Directory Setup ---
OUTPUT_DIR="${1:-artifacts/models/deep_learning/wavelm}"
mkdir -p "$OUTPUT_DIR"

# --- 6. Launch WavLM Fine-Tuning ---
echo -e "\n======================================================================"
echo " STAGE 1: Fine-Tuning WavLM (microsoft/wavlm-base-plus) on mPower"
echo " Configuration:"
echo "   - Backbone        : microsoft/wavlm-base-plus (Pretrained on 94,000h diverse speech)"
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
  --model-name microsoft/wavlm-base-plus \
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
  --num-workers 2

# --- 7. Evaluate on Held-Out Test Split ---
echo -e "\n======================================================================"
echo " STAGE 2: Evaluating WavLM on Held-Out Test Split (579 Participants)"
echo "======================================================================"

python3 -m src.deep_learning.evaluate \
  --model-dir "$OUTPUT_DIR" \
  --test-metadata data/metadata/test.csv \
  --audio-root data/processed_audio \
  --duration 8.0 \
  --batch-size 16 \
  --num-workers 0 \
  --device cuda \
  --fp16


echo -e "\n======================================================================"
echo " [SUCCESS] WAVELM CLOUD TRAINING PIPELINE COMPLETE!"
echo " Checkpoint saved to: $OUTPUT_DIR/best_model.pt"
echo " Test Report saved to: $OUTPUT_DIR/test_evaluation_report.json"
echo "======================================================================"

