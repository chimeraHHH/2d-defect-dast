#!/usr/bin/env bash
# v1.2 analysis pipeline: assumes the safe (leak-free) training is done and
# the corresponding checkpoints are in results/<run>/best.pt + .npz.
#
# Run on a CPU laptop in ~30 min (no GPU required for these analyses).

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

echo "===== UQ: 4-seed ensemble + calibration ====="
$PY scripts/uq_calibration.py

echo "===== UQ: 6-member (incl xlong) ensemble ====="
$PY scripts/uq_calibration_xlong.py

echo "===== UQ: ensemble-size ablation k=1..6 ====="
$PY scripts/ensemble_size_ablation.py

echo "===== UQ: MC-Dropout vs deep ensemble ====="
$PY scripts/mc_dropout_uq.py

echo "===== UQ: σ-vs-category alignment ====="
$PY scripts/uq_by_category.py

echo "===== UQ: active-learning oracle demo ====="
$PY scripts/active_learning_demo.py

echo "===== Interp: multi-head attention ====="
$PY scripts/attention_baseline.py

echo "===== Interp: layer-by-layer attention ====="
$PY scripts/attention_layer_compare.py

echo "===== Interp: occlusion attribution ====="
$PY scripts/occlusion_attribution.py

echo "===== Interp: 3-sample panel ====="
$PY scripts/interp_panel.py

echo "===== Interp: feature importance (permutation) ====="
$PY scripts/feature_importance.py

echo "===== Error decomposition (6 axes) ====="
$PY scripts/error_decomposition.py baseline_h128_aug_long_safe

echo "===== LOHO: per-host in-distribution reference ====="
$PY scripts/loho_id_reference.py

echo "===== LOHO: aggregate summary (requires loho_*/metrics.json) ====="
$PY scripts/loho_summary.py || echo "loho_summary.py: skip if results/loho_*/ not yet present"

echo "===== LOHO interp sanity check on each available LOHO host ====="
for h in MoS2 Cr2I6 C2H2 TaSe2 MoSSe; do
  if [ -f "results/loho_${h}/best.pt" ]; then
    $PY scripts/loho_interp_check.py --host "$h"
  fi
done

echo "===== Aggregate metrics ====="
$PY scripts/aggregate_metrics.py

echo "===== All v1.2 analyses done. Outputs in results/*.json + paper/figures/. ====="
