import os
import subprocess

with open("run_wse2_final.sh", "w") as f:
    f.write('''#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpaw-env
export GPAW_NEW=1
export GPAW_USE_GPUS=1
export OMP_NUM_THREADS=1
python run_prediction_pipeline.py > final_wse2_output.log 2>&1
''')

subprocess.run(["bash", "run_wse2_final.sh"])
