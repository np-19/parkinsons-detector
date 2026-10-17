#!/usr/bin/env bash
# ==============================================================================
# Run Local Clinical Evaluation on 'records' Dataset (Zero Docker)
# Default is CPU mode. Supports optional GPU with automatic fallback:
#   ./scripts/evaluate_records.sh           # Default CPU mode on sustained vowel /a/
#   ./scripts/evaluate_records.sh gpu       # GPU mode (falls back to CPU if unavailable)
#   ./scripts/evaluate_records.sh cpu       # Explicit CPU mode
#   ./scripts/evaluate_records.sh --all     # Evaluate all vowels (/a/ and /i/)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

DEVICE="cpu"
VOWELS="a"
AUDIO_TYPE="clean"

EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        gpu|--gpu|cuda)
            DEVICE="gpu"
            shift
            ;;
        cpu|--cpu)
            DEVICE="cpu"
            shift
            ;;
        --all|all)
            VOWELS="all"
            shift
            ;;
        --raw|raw)
            AUDIO_TYPE="raw"
            shift
            ;;
        --clean|clean)
            AUDIO_TYPE="clean"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [cpu|gpu] [--all] [--raw|--clean] [--limit N]"
            echo "  $0                # Default: CPU mode on sustained vowel /a/"
            echo "  $0 gpu            # Accelerate with NVIDIA GPU (CUDA)"
            echo "  $0 cpu            # Explicit CPU mode"
            echo "  $0 --all          # Test both vowels /a/ and /i/"
            echo "  $0 --raw          # Test uncleaned raw recordings"
            echo "  $0 --limit N      # Evaluate only first N subjects"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

VENV_PACKAGES="${REPO_ROOT}/data/.venv/lib/python3.14/site-packages"
export PYTHONPATH="${REPO_ROOT}:${VENV_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

exec python3 "${SCRIPT_DIR}/evaluate_records_dataset.py" \
  --device "${DEVICE}" \
  --vowels "${VOWELS}" \
  --audio-type "${AUDIO_TYPE}" \
  --csv "test_samples/records/dataset.csv" \
  --audio-dir "test_samples/records" \
  --output-csv "artifacts/records_dataset_evaluation_results.csv" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

