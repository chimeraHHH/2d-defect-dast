#!/bin/bash
# Launch next experiment from priority queue on a specified GPU.
# Usage: bash scripts/launch_queue.sh <gpu_id> [--skip N]
#
# Priority order (most informative → least):
#   1. v4_moe       — MoE readout for extreme-Ef specialization
#   2. v2_asph      — ASPH topological features (55% MAE reduction in literature)
#   3. v7_all       — All innovations combined
#   4. v2_uncertainty — Heteroscedastic loss
#   5. v2_focal     — Focal MAE loss
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
    "configs/v2_asph.yaml"
    "configs/v7_all.yaml"
    "configs/v2_uncertainty.yaml"
    "configs/v2_focal.yaml"
)

declare -a NAMES=(
    "v4_moe"
    "v2_asph"
    "v7_all"
    "v2_uncertainty"
    "v2_focal"
)

# Find next un-launched experiment
LAUNCHED=0
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    name="${NAMES[$i]}"
    outdir="results/$name"

    # Skip if already running or completed
    if [ -f "$outdir/metrics.json" ]; then
        echo "SKIP $name: already completed"
        continue
    fi
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
