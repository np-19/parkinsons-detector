#!/usr/bin/env bash
# ==============================================================================
# Full Clean Pipeline: Extract Train/Val Bags -> Train MIL -> One-Shot Test
# ==============================================================================
set -euo pipefail

PYTHON_BIN="data/.venv/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo " 1. Waiting for Validation Bags Extraction to Complete..."
echo "======================================================================"
while [ ! -f "artifacts/mil/bags_val.pt" ]; do
    echo "Waiting for bags_val.pt (checking every 15s)..."
    sleep 15
done
echo ">>> Validation bags ready: artifacts/mil/bags_val.pt"

echo "======================================================================"
echo " 2. Extracting Training Bags (data/metadata/train.csv)..."
echo "======================================================================"
if [ ! -f "artifacts/mil/bags_train.pt" ]; then
    ${PYTHON_BIN} -m src.mil.extractor \
        --metadata data/metadata/train.csv \
        --output-path artifacts/mil/bags_train.pt \
        --batch-size 2
    echo ">>> Training bags ready: artifacts/mil/bags_train.pt"
else
    echo ">>> Training bags already exist at artifacts/mil/bags_train.pt"
fi

echo "======================================================================"
echo " 3. Training Gated Attention-MIL on Train Bags & Validating on Val..."
echo "======================================================================"
${PYTHON_BIN} scripts/train_mil_clean.py \
    --train-bags artifacts/mil/bags_train.pt \
    --val-bags artifacts/mil/bags_val.pt \
    --test-bags artifacts/mil/bags_test.pt \
    --output-dir artifacts/mil \
    --epochs 30

echo "======================================================================"
echo " 4. Running Model Evaluation..."
echo "======================================================================"
${PYTHON_BIN} -m src.deep_learning.evaluate \
    --model-dir artifacts/models/deep_learning/wavelm \
    --test-metadata data/metadata/test.csv \
    --audio-root data/processed_audio \
    --bags-path artifacts/mil/bags_test.pt \
    --eval-bags

echo "======================================================================"
echo " Pipeline Finished Successfully!"
echo "======================================================================"
