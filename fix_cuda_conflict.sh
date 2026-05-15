#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpaw-env

echo "1. 卸载 PyTorch 强行塞入的过时 CUDA 12.1 依赖库..."
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 cupy cupy-cuda12x || true

echo "2. 重新安装干净的 cupy-cuda12x (将自动链接到我们之前配好的最新版 Conda CUDA Toolkit)..."
pip install cupy-cuda12x

echo "3. 清理 CuPy 缓存..."
rm -rf ~/.cupy/kernel_cache
rm -rf /tmp/cupy_cache
