#!/usr/bin/env bash
# Wait for baseline to finish (metrics.json materialises) then chain through
# the remaining experiments. Skip an experiment if its output already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

wait_for_metrics() {
  local dir="$1"
  while [ ! -f "$dir/metrics.json" ]; do
    sleep 5
  done
}

wait_for_metrics results/baseline
echo "[queue] baseline done -> running improved"

declare -a CONFIGS=(
  "configs/improved.yaml results/improved"
  "configs/ablate_local_only.yaml results/ablate_local_only"
  "configs/ablate_no_virtual.yaml results/ablate_no_virtual"
  "configs/ablate_no_lattice.yaml results/ablate_no_lattice"
)

for entry in "${CONFIGS[@]}"; do
  cfg=$(echo "$entry" | awk '{print $1}')
  out=$(echo "$entry" | awk '{print $2}')
  name=$(basename "$cfg" .yaml)
  if [ -f "$out/metrics.json" ]; then
    echo "[queue] skipping $name (already done)"
    continue
  fi
  echo "[queue] launching $name"
  $PY -m src.train --config "$cfg" 2>&1 | tee "results/logs/${name}.run.log"
done

echo "[queue] generating figures"
$PY scripts/make_figures.py
$PY scripts/analyze_results.py

echo "[queue] all done"
