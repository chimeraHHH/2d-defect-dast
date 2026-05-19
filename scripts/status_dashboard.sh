#!/bin/bash
# Quick status dashboard for all experiments.
# Usage: bash scripts/status_dashboard.sh
# Run from local machine (SSHs to WHU server).

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " DAST Training Dashboard — $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ssh whu 'cd ~/Workspace/yiminghua/project && echo ""

# GPU status
echo "┌─ GPU STATUS ─────────────────────────────────────┐"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | while read line; do
    gpu=$(echo "$line" | cut -d, -f1 | tr -d " ")
    util=$(echo "$line" | cut -d, -f2 | tr -d " ")
    used=$(echo "$line" | cut -d, -f3 | tr -d " ")
    total=$(echo "$line" | cut -d, -f4 | tr -d " ")
    ours=""
    ps aux | grep "CUDA_VISIBLE_DEVICES=$gpu" | grep -v grep | grep -q train_enhanced && ours=" <-- OURS"
    echo "│  GPU $gpu: $util util, $used/$total$ours"
done
echo "└─────────────────────────────────────────────────┘"
echo ""

# Running experiments
echo "┌─ ACTIVE TRAINING ──────────────────────────────────┐"
for dir in v3_deftype v4_moe v6_physics v9_contrast v11_lds v12_lds_physics v13_rnc v2_ema v10_best_combo; do
    logfile="results/$dir/nohup.log"
    if [ -f "$logfile" ]; then
        latest=$(tail -1 "$logfile" 2>/dev/null)
        if echo "$latest" | grep -q "Epoch"; then
            epoch=$(echo "$latest" | grep -oP "Epoch \K\d+")
            total=$(echo "$latest" | grep -oP "Epoch \d+/\K\d+")
            val_mae=$(echo "$latest" | grep -oP "val MAE \K[0-9.]+")
            time_s=$(echo "$latest" | grep -oP "\| \K[0-9.]+(?=s)")
            improved=""
            echo "$latest" | grep -q "\*" && improved=" *BEST*"
            if [ -n "$epoch" ] && [ -n "$total" ]; then
                # Use 10# prefix to avoid octal interpretation of 08/09
                pct=$(( 10#$epoch * 100 / 10#$total ))
                time_int=${time_s%.*}
                remaining_s=$(( (10#$total - 10#$epoch) * 10#$time_int ))
                hours=$(( remaining_s / 3600 ))
                mins=$(( (remaining_s % 3600) / 60 ))
                bar=""
                filled=$(( pct / 5 ))
                for i in $(seq 1 20); do
                    if [ $i -le $filled ]; then bar="${bar}█"; else bar="${bar}░"; fi
                done
                echo "│  $dir:"
                echo "│    [$bar] $pct% (ep $epoch/$total)"
                echo "│    val MAE=$val_mae  ETA ~${hours}h${mins}m$improved"
            fi
        fi
    fi
done
echo "└───────────────────────────────────────────────────┘"
echo ""

# Best checkpoints
echo "┌─ CHECKPOINTS ──────────────────────────────────────┐"
for dir in v3_deftype v4_moe v6_physics v9_contrast v11_lds v12_lds_physics v13_rnc v2_ema; do
    if [ -f "results/$dir/best.pt" ]; then
        size=$(du -h "results/$dir/best.pt" | cut -f1)
        echo "│  $dir: best.pt ($size)"
    fi
done
echo "└───────────────────────────────────────────────────┘"
echo ""

# Queue
echo "┌─ QUEUE (waiting) ──────────────────────────────────┐"
for dir in v9_contrast v11_lds v12_lds_physics v13_rnc v2_ema v10_best_combo v2_focal v2_uncertainty; do
    if [ -f "results/$dir/test_predictions.npz" ] && [ -f "results/$dir/metrics.json" ]; then
        n_ep=$(/home/huayiming/.conda/envs/yiminghua/bin/python -c "import json; m=json.load(open('results/$dir/metrics.json')); print(len(m.get('history',[])))" 2>/dev/null || echo "0")
        if [ "$n_ep" -ge 20 ] 2>/dev/null; then
            echo "│  DONE:    $dir ($n_ep ep)"
        else
            echo "│  PENDING: $dir (smoke test only)"
        fi
    elif pgrep -f "configs/${dir}.yaml" > /dev/null 2>&1; then
        echo "│  RUNNING: $dir"
    else
        echo "│  PENDING: $dir"
    fi
done
echo "└───────────────────────────────────────────────────┘"
'
