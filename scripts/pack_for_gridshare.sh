#!/usr/bin/env bash
# ==============================================================================
# Helper script to create a clean tarball for GridShare upload
# Excludes local caches, virtual environments, and benchmark test artifacts
# ==============================================================================
set -euo pipefail

OUTPUT_FILE="${HOME}/parkinsons-detector.tar.gz"

echo "======================================================================"
echo " Packaging parkinsons-detector for GridShare Upload"
echo " Destination: ${OUTPUT_FILE}"
echo "======================================================================"

cd "${HOME}"
tar --exclude='parkinsons-detector/artifacts/models/deep_learning/benchmark_test' \
    --exclude='parkinsons-detector/src/deep_learning/__pycache__' \
    --exclude='parkinsons-detector/.cache' \
    --exclude='parkinsons-detector/.venv' \
    --exclude='parkinsons-detector/*.tar.gz' \
    -czf "${OUTPUT_FILE}" parkinsons-detector/

FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
echo "======================================================================"
echo " Successfully created ${OUTPUT_FILE} (${FILE_SIZE})"
echo " Ready to upload to your GridShare RTX 5060 Ti instance!"
echo " Command: scp -P <PORT> ${OUTPUT_FILE} root@<IP>:/root/"
echo "======================================================================"
