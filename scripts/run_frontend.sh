#!/usr/bin/env bash
# ==============================================================================
# Launch NeuroVocal React / Vite Frontend
# Accessible at http://localhost:3000
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/../frontend"

echo "======================================================================"
echo " Starting NeuroVocal Clinical UI (React + TypeScript + Tailwind)"
echo " Local URL: http://localhost:3000"
echo " Backend:   http://localhost:8000"
echo "======================================================================"

npm run dev -- --host 0.0.0.0 --port 3000
