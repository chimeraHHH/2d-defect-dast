#!/usr/bin/env bash
# Run dualstream multi-seed (seed 0, 1, 2) sequentially.
# Each ~25 min; total ~75 min wall.
set -e
cd /root/2d-defect-dast
source .venv/bin/activate
LOG=results/dualstream_3seed_queue.log
echo "[$(date '+%F %T')] dualstream 3-seed queue starting" >> "$LOG"

for s in 0 1 2; do
  echo "[$(date '+%F %T')] >>> START seed=$s" >> "$LOG"
  python -m src.train --config configs/dualstream_h128_imp2d_seed${s}.yaml >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] <<< DONE seed=$s" >> "$LOG"
done

echo "[$(date '+%F %T')] queue complete" >> "$LOG"
touch results/dualstream_3seed_queue.done
