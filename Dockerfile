FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
LABEL maintainer="jingyu sun"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python 3.12. SAM3.1 expects the official
# facebookresearch/sam3 runtime, which currently targets Python 3.12 and CUDA 12.6.
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        git \
        git-lfs \
        curl \
        nvtop \
        screen \
        ca-certificates \
        libsndfile1-dev \
        libgl1 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment.
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install pip / uv.
RUN python -m pip install --no-cache-dir --upgrade pip uv

WORKDIR /data/users/jingyu/

# Copy dependency metadata first to use Docker layer cache.
COPY pyproject.toml ./

# Install CUDA 12.6 PyTorch first. Pinning avoids resolver drift with vLLM/SGLang/SAM3.
RUN uv pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# Install project runtime dependencies without treating pyproject.toml as a requirements file.
# Project code is mounted from NFS and run as scripts; the package itself has no importable modules.
RUN uv pip install --no-cache-dir \
    numpy \
    pillow \
    "pycocotools>=2.0.8" \
    "transformers>=4.57.0" \
    "accelerate>=0.30.0" \
    "openai>=1.40.0" \
    "anthropic>=0.34.0" \
    "google-genai>=1.0.0" \
    "python-dotenv>=1.0.1" \
    "requests>=2.32.0" \
    "huggingface-hub>=0.24.0" \
    "hf-transfer>=0.1.8" \
    "safetensors>=0.4.5" \
    "pandas>=2.2.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "nibabel>=5.2.0" \
    "bitsandbytes>=0.43.1" \
    "vllm>=0.6.0" \
    "sglang>=0.4.0"

# Some optional runtime packages can pull a newer CUDA-13 PyTorch build through
# dependency resolution. Re-pin the CUDA 12.6 build supported by the base image
# and the target cluster driver before optional SAM3 installation.
RUN uv pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# Optional SAM3.1 runtime. Keep it as a build arg because it is a fast-moving,
# gated, CUDA-heavy dependency that should not be required for non-SAM jobs.
ARG INSTALL_SAM3=false
RUN if [ "${INSTALL_SAM3}" = "true" ]; then \
        git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3 && \
        uv pip install --no-cache-dir -e /opt/sam3 && \
        uv pip install --no-cache-dir "pycocotools>=2.0.8" && \
        uv pip install --no-cache-dir --force-reinstall \
            --index-url https://download.pytorch.org/whl/cu126 \
            torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0; \
    fi

# Environment variables.
ENV TORCH_HOME=/data/users/jingyu/.torch
ENV HF_HOME=/data/users/jingyu/Huggingface
ENV HF_DATASETS_CACHE=/data/users/jingyu/Huggingface
ENV HF_XET_HIGH_PERFORMANCE=1
ENV DNNLIB_CACHE_DIR=/data/users/jingyu/.cache/dnnlib
ENV TORCH_EXTENSIONS_DIR=/data/users/jingyu/.torch/torch_extensions
ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"
ENV PYTORCH_KERNEL_CACHE_PATH=/data/users/jingyu/.torch/kernels
ENV UV_CACHE_DIR=/data/users/jingyu/.cache

# Add NFS-compatible user.
ARG USER_ID=35761
ARG GROUP_ID=4451
RUN groupadd -g ${GROUP_ID} usergroup && \
    useradd -m -u ${USER_ID} -g usergroup jingyu
USER ${USER_ID}

# Port and working directory.
EXPOSE 8081
