#!/bin/bash
# Launch next experiment from priority queue on a specified GPU.
# Usage: bash scripts/launch_queue.sh <gpu_id> [--skip N]
#
# Priority order (most informative for paper → least):
#   1. v4_moe       — MoE readout for energy-range specialization (ICLR 2025)
#   2. v9_contrast  — Defect-host contrast conditioning (novel, physically motivated)
#   3. v11_lds      — Label Distribution Smoothing (Yang et al., ICML 2021)
#   4. v12_lds_phys — LDS + physics + deftype + contrast (full synergy)
#   5. v2_ema       — EMA baseline (ablation: does EMA help V2?)
#   6. v10_best     — All best innovations combined
#   7. v2_focal     — Focal MAE loss for hard samples
#   8. v2_uncertainty — Heteroscedastic loss (Kendall & Gal, NeurIPS 2017)
#
# Skip N experiments to launch a lower-priority one.

set -e

GPU=${1:-6}
SKIP=${2:-0}

# Safety: never use GPU 2
if [ "$GPU" = "2" ]; then
    echo "ERROR: GPU 2 has hardware issues. Use another GPU."
    exit 1
fi

# Priority queue
declare -a CONFIGS=(
    "configs/v4_moe.yaml"
    "configs/v9_contrast.yaml"
    "configs/v11_lds.yaml"
    "configs/v12_lds_physics.yaml"
    "configs/v13_rnc.yaml"
    "configs/v2_ema.yaml"
    "configs/v10_best_combo.yaml"
    "configs/v2_focal.yaml"
    "configs/v2_uncertainty.yaml"
)

declare -a NAMES=(
    "v4_moe"
    "v9_contrast"
    "v11_lds"
    "v12_lds_physics"
    "v13_rnc"
    "v2_ema"
    "v10_best_combo"
    "v2_focal"
    "v2_uncertainty"
)

# Find next un-launched experiment
LAUNCHED=0
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    name="${NAMES[$i]}"
    outdir="results/$name"

    # Skip if already completed (test_predictions.npz from 20+ epoch run)
    if [ -f "$outdir/test_predictions.npz" ] && [ -f "$outdir/metrics.json" ]; then
        n_epochs=$(python3 -c "import json; m=json.load(open('$outdir/metrics.json')); print(len(m.get('history',[])))" 2>/dev/null || echo "0")
        if [ "$n_epochs" -ge 20 ] 2>/dev/null; then
            echo "SKIP $name: completed ($n_epochs epochs)"
            continue
        fi
    fi
    # Skip if already running
    if pgrep -f "$config" > /dev/null 2>&1; then
        echo "SKIP $name: already running"
        continue
    fi

    # Skip N experiments
    if [ $LAUNCHED -lt $SKIP ]; then
        LAUNCHED=$((LAUNCHED + 1))
        echo "SKIP $name: --skip $SKIP"
        continue
    fi

    echo "Launching: $name on GPU $GPU"
    echo "Config: $config"
    mkdir -p "$outdir"
    CUDA_VISIBLE_DEVICES=$GPU nohup /home/huayiming/.conda/envs/yiminghua/bin/python -u \
        src/train_enhanced.py --config "$config" \
        > "$outdir/nohup.log" 2>&1 &
    PID=$!
    echo "PID=$PID"
    echo "Log: $outdir/nohup.log"
    exit 0
done

echo "All experiments either completed or running!"
