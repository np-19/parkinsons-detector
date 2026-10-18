#!/usr/bin/env bash
# ==============================================================================
# Run Evaluation on Held-Out Test Split Locally (Zero Docker, Local CUDA GPU)
# Uses the project virtual environment (data/.venv)
#
# Usage:
#   ./scripts/run_evaluate_local.sh                                          # Defaults to WaveLM v4 best model
#   ./scripts/run_evaluate_local.sh artifacts/models/deep_learning/wavelm_v4 # Explicit model directory
#   ./scripts/run_evaluate_local.sh <model_dir> <num_crops> <duration>
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Determine python executable
if [ -x "${REPO_ROOT}/data/.venv/bin/python3" ]; then
    PYTHON_EXEC="${REPO_ROOT}/data/.venv/bin/python3"
else
    PYTHON_EXEC="python3"
fi

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: $0 [model_dir] [num_crops] [duration] [batch_size]"
    echo "  $0                                          # Default: WaveLM v4 best model on local GPU"
    echo "  $0 artifacts/models/deep_learning/wavelm_v4 # Explicit model directory"
    echo "  $0 artifacts/models/deep_learning/wav2vec2_v2"
    exit 0
fi

# Parse optional --fast flag or positional args
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "${arg}" = "--fast" ] || [ "${arg}" = "fast" ]; then
        EXTRA_ARGS+=("--eval-bags")
    fi
done

MODEL_DIR="${1:-artifacts/models/deep_learning/wavelm_v4}"
# If first arg was fast flag, reset MODEL_DIR to default
if [ "${MODEL_DIR}" = "--fast" ] || [ "${MODEL_DIR}" = "fast" ]; then
    MODEL_DIR="artifacts/models/deep_learning/wavelm_v4"
fi
NUM_CROPS="${2:-5}"
DURATION="${3:-8.0}"
BATCH_SIZE="${4:-4}"

# Hardware detection
if "${PYTHON_EXEC}" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo ">>> [HARDWARE] CUDA GPU detected: $("${PYTHON_EXEC}" -c "import torch; print(torch.cuda.get_device_name(0))")"
else
    DEVICE="cpu"
    echo "[WARNING] CUDA GPU not detected. Falling back to CPU."
fi

echo "======================================================================"
echo " Evaluating Frozen Best Checkpoint on Held-Out Test Set (No Docker)"
echo " Model Directory : ${MODEL_DIR}"
echo " TTA Crops       : ${NUM_CROPS} | Duration: ${DURATION}s | Batch Size: ${BATCH_SIZE}"
echo " Inference Device: ${DEVICE}"
echo " Python Binary   : ${PYTHON_EXEC}"
echo "======================================================================"

VENV_PACKAGES="${REPO_ROOT}/data/.venv/lib/python3.14/site-packages"
export PYTHONPATH="${REPO_ROOT}:${VENV_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

exec "${PYTHON_EXEC}" -m src.deep_learning.evaluate \
  --model-dir "${MODEL_DIR}" \
  --test-metadata "data/metadata/test.csv" \
  --audio-root "data/processed_audio" \
  --duration "${DURATION}" \
  --num-crops "${NUM_CROPS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
