#!/bin/bash
# 激活 Conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpaw-env

# 彻底卸载旧的 cupy-cuda12x
pip uninstall cupy-cuda12x -y || true
# 强制从官方安装 cupy 源代码并在本地为 5090 编译！
# 这是解决 CUDA_ERROR_NO_BINARY_FOR_GPU 的终极方法
export CUPY_NVCC_GENERATE_PTX="compute_89"
pip install cupy --no-binary cupy -v

# 重新编译 GPAW
echo "=================================================="
echo "检测到 CuPy 更新，正在重新编译 GPAW 以链接新版 CuPy..."
cd /root/autodl-tmp/2d-defect-dast/gpaw-25.7.0
rm -rf build/
pip install . --no-build-isolation
cd /root/autodl-tmp/2d-defect-dast
echo "=================================================="

# 确保环境变量设置
export GPAW_NEW=1
export GPAW_USE_GPUS=1
export OMP_NUM_THREADS=1

# 执行计算
python calculate_defect.py
