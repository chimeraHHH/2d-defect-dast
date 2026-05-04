#!/usr/bin/env bash
# Sequentially train v2 multi-source for 3 additional seeds.
# Each run takes ~25 min; total ~75 min wall.
set -e
cd /root/2d-defect-dast
source .venv/bin/activate
LOG=results/v2_multi_3seed_queue.log
echo "[$(date '+%F %T')] starting 3-seed multi-source v2 queue" >> "$LOG"

for s in 0 1 2; do
  echo "[$(date '+%F %T')] >>> START seed=$s" >> "$LOG"
  python -m scripts.multi_source_v2_seedrun --seed $s >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] <<< DONE seed=$s" >> "$LOG"
done

echo "[$(date '+%F %T')] 3-seed queue complete" >> "$LOG"
# write a sentinel file so external pollers can detect completion
touch results/v2_multi_3seed_queue.done
