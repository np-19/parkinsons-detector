# ==============================================================================
# Dockerfile for NeuroVocal FastAPI Backend Service
# Runs GPU-accelerated acoustic inference with PyTorch, Torchaudio & Wav2Vec 2.0
# ==============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Install system audio libraries and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install core and backend dependencies
COPY requirements.txt /workspace/requirements.txt
COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt && \
    pip install --no-cache-dir -r /workspace/backend/requirements.txt

# Copy source tree
COPY . /workspace

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
