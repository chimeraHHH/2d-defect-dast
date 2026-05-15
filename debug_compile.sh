#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpaw-env
cd /root/autodl-tmp/2d-defect-dast/gpaw-25.7.0

echo "正在生成详细的编译日志，请稍候..."
pip install . --no-build-isolation -v > build_log.txt 2>&1 || true

echo "================ 最新编译错误日志 (最后 50 行) ================"
tail -n 50 build_log.txt
echo "================================================================"
