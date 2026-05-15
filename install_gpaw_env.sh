#!/bin/bash
set -e

echo "=================================================="
echo "开始配置 GPAW (GPU加速) 与 ASE 环境"
echo "=================================================="

# 1. 创建虚拟环境 (建议使用conda，如果没有conda则用venv)
if command -v conda &> /dev/null; then
    echo "[1/4] 检测到 Conda，正在创建并激活 gpaw-env 环境..."
    conda create -n gpaw-env python=3.10 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate gpaw-env
else
    echo "[1/4] 未检测到 Conda，使用 Python venv 创建环境..."
    python3 -m venv gpaw-env
    source gpaw-env/bin/activate
fi

# 2. 安装基础依赖
echo "[2/4] 安装 numpy, scipy, matplotlib, ase..."
pip install --upgrade pip
pip install numpy scipy matplotlib ase

# 3. 安装 CuPy (适用于 CUDA 12.x，RTX 5090 通常为 CUDA 12)
echo "[3/4] 安装 CuPy (GPU加速核心库)..."
pip install cupy-cuda12x

# 4. 安装 GPAW
echo "[4/4] 安装 GPAW..."
# 推荐通过 conda 安装以自动解决 libxc 等底层 C 库依赖
if command -v conda &> /dev/null; then
    conda install -c conda-forge gpaw -y
else
    pip install gpaw
fi

echo "[5/5] 下载并安装 GPAW PAW 赝势数据集..."
gpaw install-data ~/.gpaw -y || echo "请手动运行 gpaw install-data 下载赝势库"

echo "=================================================="
echo "环境配置完成！"
echo "请在每次使用前运行以下命令激活环境："
if command -v conda &> /dev/null; then
    echo "conda activate gpaw-env"
else
    echo "source gpaw-env/bin/activate"
fi
echo "并且为了启用新版 GPAW 的 GPU 特性，请设置环境变量："
echo "export GPAW_NEW=1"
echo "export GPAW_USE_GPUS=1"
echo "=================================================="
