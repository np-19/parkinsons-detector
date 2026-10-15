# ==============================================================================
# Dockerfile for Parkinson's Speech Detection Fine-Tuning Pipeline
# Compatible with local NVIDIA GPUs (e.g., GTX 1650 4GB) and Cloud GPUs (GridShare)
# ==============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Install system audio libraries and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# Copy codebase into container image
COPY . /workspace

# Set default command to bash
CMD ["bash"]

