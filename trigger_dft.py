import os
import subprocess

with open("run_dft.sh", "w") as f:
    f.write('''#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpaw-env
python calculate_defect.py > dft_final_output.log 2>&1
''')

subprocess.run(["bash", "run_dft.sh"])
