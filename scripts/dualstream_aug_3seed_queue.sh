#!/usr/bin/env bash
# Sequential 3-seed run of dualstream + leak-free aug (seed 0,1,2).
# Each ~62 min; total ~3.1 h.
set -e
cd /root/2d-defect-dast
source .venv/bin/activate
LOG=results/dualstream_aug_3seed_queue.log
echo "[$(date '+%F %T')] dualstream-aug 3-seed queue starting" >> "$LOG"
for s in 0 1 2; do
  echo "[$(date '+%F %T')] >>> START seed=$s" >> "$LOG"
  python -m src.train --config configs/dualstream_h128_aug_seed${s}.yaml >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] <<< DONE seed=$s" >> "$LOG"
done
echo "[$(date '+%F %T')] queue complete" >> "$LOG"
touch results/dualstream_aug_3seed_queue.done
