#!/usr/bin/env bash
# Run v2-PFA multi-source LOHO for all 5 hosts sequentially.
# Each takes ~25 min; total ~2.1 h.
set -e
cd /root/2d-defect-dast
source .venv/bin/activate
LOG=results/v2_loho_multi_queue.log
echo "[$(date '+%F %T')] starting v2 LOHO multi-source queue (5 hosts)" >> "$LOG"

for h in MoS2 MoSSe TaSe2 Cr2I6 C2H2; do
  echo "[$(date '+%F %T')] >>> START host=$h" >> "$LOG"
  python -m scripts.multi_source_v2_loho --host $h >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] <<< DONE host=$h" >> "$LOG"
done

echo "[$(date '+%F %T')] LOHO queue complete" >> "$LOG"
touch results/v2_loho_multi_queue.done
