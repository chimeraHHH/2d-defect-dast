#!/bin/bash
set -e

echo "=================================================="
echo "开始为 RTX 5090 从源码编译带 GPU 支持的 GPAW"
echo "=================================================="

# 1. 补全 AutoDL 常用的 CUDA 环境变量路径
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 2. 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpaw-env

# 3. 卸载之前用 pip 安装的无 GPU 支持版本
echo "[1/4] 卸载旧版本 GPAW..."
pip uninstall gpaw -y || true

# 4. 通过 conda 安装编译 C 扩展必须的底层依赖
echo "[2/4] 安装编译环境和底层依赖 (libxc, fftw, openblas 等)..."
conda install -c conda-forge compilers libxc-c fftw openblas mpi4py mpich -y

# 安装 cuda-toolkit 提供 nvcc
echo "[2.5/4] 安装 nvcc 和 cuda-toolkit..."
conda install -c nvidia cuda-toolkit -y

# 5. 进入已下载的源码目录
cd /root/autodl-tmp/2d-defect-dast/gpaw-25.7.0

# 6. 生成编译配置文件 siteconfig.py
echo "[3/4] 配置 GPU 编译参数 siteconfig.py..."
cat << 'EOF' > siteconfig.py
import os

compiler = 'gcc'
mpi = True        # 开启 MPI
fftw = True
scalapack = False  # 单节点通常不需要 scalapack

# 链接的底层数学和物理库，注意添加 mpi 相关的链接库
libraries = ['xc', 'fftw3', 'openblas', 'mpi']

# 指向 Conda 虚拟环境的库路径
conda_prefix = os.environ.get('CONDA_PREFIX', '/root/miniconda3/envs/gpaw-env')
library_dirs = [os.path.join(conda_prefix, 'lib')]
include_dirs = [os.path.join(conda_prefix, 'include')]
runtime_library_dirs = [os.path.join(conda_prefix, 'lib')]

# 核心 GPU 配置
gpu = True
gpu_target = 'cuda'
gpu_compiler = 'nvcc'

# RTX 5090 属于 Blackwell 架构，这里使用 sm_89 (Ada架构) 以保证最高兼容性，同时能充分调用 Tensor Core
gpu_compile_args = ['-O3', '-g', '-arch=sm_89']

libraries += ['cudart', 'cublas']
EOF

# 7. 清理之前的构建缓存并重新编译
echo "[4/4] 开始硬核编译 GPAW (大概需要 1-3 分钟)，请耐心等待..."
rm -rf build/
pip install . --no-build-isolation

echo "=================================================="
echo "编译完成！"
echo "请运行: python calculate_defect.py"
echo "现在您将真正感受到 RTX 5090 的算力了！"
echo "=================================================="
