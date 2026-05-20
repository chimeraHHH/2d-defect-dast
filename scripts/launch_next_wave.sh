#!/bin/bash
# Launch next wave of experiments on available GPUs.
# Detects which experiments are done/running and launches next from priority queue.
# Respects 3-GPU limit and avoids GPU 2.
#
# Usage:
#   bash scripts/launch_next_wave.sh          # auto-detect free GPUs
#   bash scripts/launch_next_wave.sh 0 4 6    # specify GPU IDs
#
# Priority order (after V3/V4/V6 complete):
#   V9 contrast → V11 LDS → V13 RnC → V14 JK → V2_ema → V2_focal → V12 combined → V10 all

set -e
cd ~/Workspace/yiminghua/project

CONDA_PYTHON=/home/huayiming/.conda/envs/yiminghua/bin/python
MAX_GPUS=3

# Priority queue: config path -> output dir name
declare -a QUEUE_CONFIGS=(
    "configs/v9_contrast.yaml"
    "configs/v11_lds.yaml"
    "configs/v13_rnc.yaml"
    "configs/v14_jk.yaml"
    "configs/v2_ema.yaml"
    "configs/v2_focal.yaml"
    "configs/v12_lds_physics.yaml"
    "configs/v10_best_combo.yaml"
    "configs/v2_uncertainty.yaml"
)

declare -a QUEUE_NAMES=(
    "v9_contrast"
    "v11_lds"
    "v13_rnc"
    "v14_jk"
    "v2_ema"
    "v2_focal"
    "v12_lds_physics"
    "v10_best_combo"
    "v2_uncertainty"
)

# Count currently running training jobs
count_running() {
    pgrep -u $(whoami) -f "train_enhanced.py" 2>/dev/null | wc -l
}

# Check if a specific experiment is running
is_running() {
    pgrep -u $(whoami) -f "$1" > /dev/null 2>&1
}

# Check if experiment is complete (150 epochs with test predictions)
is_complete() {
    local name=$1
    local outdir="results/$name"
    if [ -f "$outdir/test_predictions.npz" ]; then
        # Check if it has reasonable number of epochs in the log
        if [ -f "$outdir/nohup.log" ]; then
            local last_ep=$(grep -oP 'Epoch \K\d+' "$outdir/nohup.log" 2>/dev/null | tail -1)
            if [ -n "$last_ep" ] && [ "$last_ep" -ge 140 ] 2>/dev/null; then
                return 0
            fi
        fi
        if [ -f "$outdir/metrics.json" ]; then
            local n_ep=$($CONDA_PYTHON -c "import json; m=json.load(open('$outdir/metrics.json')); print(len(m.get('history',[])))" 2>/dev/null || echo "0")
            if [ "$n_ep" -ge 140 ] 2>/dev/null; then
                return 0
            fi
        fi
    fi
    return 1
}

# Find available GPUs (not GPU 2, not heavily loaded by others)
find_free_gpus() {
    local free_gpus=()
    for gpu in 0 1 3 4 5 6 7; do  # Skip GPU 2
        local util=$(nvidia-smi -i $gpu --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        local mem=$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        # Consider GPU free if <20% util and <10GB memory used
        if [ -n "$util" ] && [ "$util" -lt 20 ] 2>/dev/null && [ "$mem" -lt 10000 ] 2>/dev/null; then
            free_gpus+=($gpu)
        fi
    done
    echo "${free_gpus[@]}"
}

echo "=========================================="
echo "  Experiment Launch Manager"
echo "=========================================="
echo ""

# Status report
echo "--- Current Status ---"
RUNNING=0
for config in configs/v*.yaml; do
    name=$(basename "$config" .yaml)
    if is_running "$config"; then
        echo "  🔄 $name: RUNNING"
        RUNNING=$((RUNNING + 1))
    elif is_complete "$name"; then
        echo "  ✅ $name: COMPLETE"
    elif [ -d "results/$name" ]; then
        echo "  ⏸  $name: PARTIAL (interrupted?)"
    fi
done
echo ""
echo "Running jobs: $RUNNING / $MAX_GPUS max"

# Determine how many new jobs we can launch
SLOTS=$((MAX_GPUS - RUNNING))
if [ $SLOTS -le 0 ]; then
    echo "No available slots (already at $MAX_GPUS GPU limit)"
    exit 0
fi

# Get available GPUs
if [ $# -gt 0 ]; then
    GPUS=("$@")
else
    read -ra GPUS <<< "$(find_free_gpus)"
fi

if [ ${#GPUS[@]} -eq 0 ]; then
    echo "No free GPUs found!"
    exit 0
fi

echo "Available GPUs: ${GPUS[@]}"
echo "Slots to fill: $SLOTS"
echo ""

# Launch next experiments
GPU_IDX=0
LAUNCHED=0
for i in "${!QUEUE_CONFIGS[@]}"; do
    if [ $LAUNCHED -ge $SLOTS ] || [ $GPU_IDX -ge ${#GPUS[@]} ]; then
        break
    fi

    config="${QUEUE_CONFIGS[$i]}"
    name="${QUEUE_NAMES[$i]}"
    outdir="results/$name"

    # Skip if complete
    if is_complete "$name"; then
        continue
    fi

    # Skip if already running
    if is_running "$config"; then
        continue
    fi

    GPU=${GPUS[$GPU_IDX]}

    # Safety: never use GPU 2
    if [ "$GPU" = "2" ]; then
        GPU_IDX=$((GPU_IDX + 1))
        if [ $GPU_IDX -ge ${#GPUS[@]} ]; then break; fi
        GPU=${GPUS[$GPU_IDX]}
    fi

    echo "🚀 Launching: $name on GPU $GPU"
    mkdir -p "$outdir"
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$GPU \
        nohup $CONDA_PYTHON -u \
        src/train_enhanced.py --config "$config" \
        > "$outdir/nohup.log" 2>&1 &
    PID=$!
    echo "   PID=$PID  Log: $outdir/nohup.log"

    GPU_IDX=$((GPU_IDX + 1))
    LAUNCHED=$((LAUNCHED + 1))
done

if [ $LAUNCHED -eq 0 ]; then
    echo "No new experiments to launch (all queued experiments either running or complete)"
else
    echo ""
    echo "Launched $LAUNCHED new experiment(s)"
fi
