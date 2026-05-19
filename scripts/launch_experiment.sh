#!/bin/bash
# Launch an experiment on a specific GPU.
# Usage: bash scripts/launch_experiment.sh <config> <gpu_id> [--seed N]
# Example: bash scripts/launch_experiment.sh configs/v4_moe.yaml 4
#          bash scripts/launch_experiment.sh configs/v2_focal.yaml 6 --seed 43

set -e

CONFIG=$1
GPU=${2:-4}
SEED_ARG=""
EXTRA_ARGS=""

shift 2 || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED_ARG="--seed $2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Extract output_dir from config
OUT_DIR=$(grep 'output_dir:' $CONFIG | awk '{print $2}')
if [ -z "$OUT_DIR" ]; then
    echo "ERROR: Could not find output_dir in $CONFIG"
    exit 1
fi

# Add seed suffix if specified
if [ -n "$SEED_ARG" ]; then
    SEED_NUM=$(echo $SEED_ARG | awk '{print $2}')
    OUT_DIR="${OUT_DIR}_s${SEED_NUM}"
fi

echo "Config: $CONFIG"
echo "GPU: $GPU"
echo "Output: $OUT_DIR"
echo "Seed arg: $SEED_ARG"

# Check GPU is available (not GPU 2!)
if [ "$GPU" = "2" ]; then
    echo "ERROR: GPU 2 has hardware issues (96W draw, 10x slowdown). Use another GPU."
    exit 1
fi

# Create output dir and launch
mkdir -p "$OUT_DIR"
CUDA_VISIBLE_DEVICES=$GPU nohup /home/huayiming/.conda/envs/yiminghua/bin/python -u \
    src/train_enhanced.py --config "$CONFIG" $SEED_ARG $EXTRA_ARGS \
    > "$OUT_DIR/nohup.log" 2>&1 &
PID=$!
echo "Launched PID=$PID on GPU $GPU"
echo "Log: $OUT_DIR/nohup.log"
echo "Monitor: tail -f $OUT_DIR/nohup.log"
