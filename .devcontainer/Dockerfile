FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install system dependencies for Open3D, PyBullet, and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libx11-6 \
    libxcursor1 \
    libxinerama1 \
    libxrandr2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/Bin-Picking-Dataset-Generation