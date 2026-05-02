#!/usr/bin/env bash
# Run after all 5 LOHO trainings finish on the remote GPU.
# Downloads results, runs all post-processing, regenerates the paper.

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="root@117.50.183.159"
PORT="23"
RDIR="/root/2d-defect-dast/results"
PY=.venv/bin/python

echo "[1/5] Downloading LOHO results from remote..."
for h in MoS2 Cr2I6 C2H2 TaSe2 MoSSe; do
  echo "  pulling loho_$h"
  scp -P "$PORT" -o StrictHostKeyChecking=no -r "$REMOTE":"$RDIR/loho_$h" results/ \
    2>/dev/null || echo "    (skip / already local)"
done

echo "[2/5] Running loho_summary.py..."
$PY scripts/loho_summary.py

echo "[3/5] Running loho_interp_check on each host (with checkpoint)..."
for h in MoS2 Cr2I6 C2H2 TaSe2 MoSSe; do
  if [ -f "results/loho_$h/best.pt" ]; then
    $PY scripts/loho_interp_check.py --host "$h" || true
  fi
done

echo "[4/5] Re-running aggregate metrics..."
$PY scripts/aggregate_metrics.py

echo "[5/5] Rebuilding paper PDF..."
cd paper
pandoc main.md -o main.pdf --pdf-engine=xelatex \
  -V CJKmainfont="PingFang SC" -V mainfont="Times New Roman" \
  -V geometry:margin=1in --toc 2>&1 | tail -3 || true
cd ..

echo "[done] All v1.2 post-LOHO integration complete."
echo "       results/loho_summary.json + paper/main.pdf updated."
