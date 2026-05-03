#!/usr/bin/env bash
# Run baseline -> improved -> three ablations sequentially. Total wall ~2 hours
# on a 2025 Apple M3 (CPU).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

declare -a CONFIGS=(
  configs/baseline.yaml
  configs/improved.yaml
  configs/ablate_local_only.yaml
  configs/ablate_no_virtual.yaml
  configs/ablate_no_lattice.yaml
)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  echo "===== Running $name ====="
  $PY -m src.train --config "$cfg" 2>&1 | tee "results/logs/${name}.run.log"
done

echo "===== Generating figures ====="
$PY scripts/make_figures.py

echo "All done."
