#!/bin/bash
# 激活 Conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpaw-env

# 确保环境变量设置
export GPAW_NEW=1
export GPAW_USE_GPUS=1
export OMP_NUM_THREADS=1

# 执行计算
python calculate_defect.py
