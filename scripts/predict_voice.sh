#!/usr/bin/env bash
# ==============================================================================
# Run Parkinson's Voice Inference on Audio Files
# Default mode is CPU. Supports GPU with automatic fallback:
#   ./scripts/predict_voice.sh audio.wav               # Default CPU inference
#   ./scripts/predict_voice.sh audio.wav --gpu         # GPU inference (falls back to CPU)
#   ./scripts/predict_voice.sh audio.wav --cpu         # CPU inference
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_audio1.wav> [path_to_audio2.wav ...] [--gpu|--cpu]"
    exit 1
fi

# Extract options
DEVICE="cpu"
AUDIO_FILES=()

for arg in "$@"; do
    case "${arg}" in
        --gpu|gpu)
            if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
                DEVICE="cuda"
            else
                echo "[WARNING] GPU requested, but CUDA is not available. Falling back to CPU."
                DEVICE="cpu"
            fi
            ;;
        --cpu|cpu)
            DEVICE="cpu"
            ;;
        *)
            AUDIO_FILES+=("${arg}")
            ;;
    esac
done

if [ "${#AUDIO_FILES[@]}" -eq 0 ]; then
    echo "Error: No audio files provided."
    echo "Usage: $0 <path_to_audio.wav> [--gpu|--cpu]"
    exit 1
fi

VENV_PACKAGES="${REPO_ROOT}/data/.venv/lib/python3.14/site-packages"
export PYTHONPATH="${REPO_ROOT}:${VENV_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

exec python3 src/deep_learning/predict_patient.py "${AUDIO_FILES[@]}" --device "${DEVICE}"
