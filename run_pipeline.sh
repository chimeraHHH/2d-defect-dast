#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpaw-env

# 检查是否安装了 PyTorch，如果没有则安装
if ! python -c "import torch" &> /dev/null; then
    echo "=================================================="
    echo "检测到环境中缺失 PyTorch，正在为您自动安装 (兼容 CUDA 12.1)..."
    echo "=================================================="
    pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# 确保环境变量设置
export GPAW_NEW=1
export GPAW_USE_GPUS=1
export OMP_NUM_THREADS=1

# 彻底清理 CuPy 编译缓存，防止旧的 JIT 代码残留
rm -rf ~/.cupy/kernel_cache
export CUPY_CACHE_DIR=/tmp/cupy_cache
rm -rf /tmp/cupy_cache
export CUPY_CACHE_IN_MEMORY=1
# 强制 CuPy 进行 JIT 编译时使用匹配 5090 (Blackwell/Ada) 的架构
export CUPY_NVCC_GENERATE_PTX="compute_89"

python run_prediction_pipeline.py
