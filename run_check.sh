#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
python check_model.py
python find_best.py
