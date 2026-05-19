#!/bin/bash
# Monitor all running training experiments.
# Usage: bash scripts/monitor_training.sh
# Run from local machine (SSHs to WHU server).

echo "=================================================="
echo "Training Monitor — $(date)"
echo "=================================================="

ssh whu 'cd ~/Workspace/yiminghua/project && echo ""

# GPU status
echo "=== GPU STATUS ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null
echo ""

# Server load
echo "=== SERVER LOAD ==="
uptime
echo ""

# Running experiments
echo "=== RUNNING EXPERIMENTS ==="
ps aux | grep train_enhanced | grep python | grep -v grep | while read line; do
    config=$(echo "$line" | grep -oP "(?<=--config )\S+")
    gpu=$(echo "$line" | grep -oP "(?<=CUDA_VISIBLE_DEVICES=)\d+")
    echo "  GPU $gpu: $config"
done
echo ""

# Training progress
for dir in v3_deftype v4_moe v6_physics v9_contrast v11_lds v12_lds_physics v2_ema v10_best_combo; do
    logfile="results/$dir/nohup.log"
    if [ -f "$logfile" ]; then
        latest=$(tail -1 "$logfile" 2>/dev/null)
        if echo "$latest" | grep -q "Epoch"; then
            epoch=$(echo "$latest" | grep -oP "Epoch \K\d+")
            total=$(echo "$latest" | grep -oP "Epoch \d+/\K\d+")
            val_mae=$(echo "$latest" | grep -oP "val MAE \K[0-9.]+")
            improved=""
            echo "$latest" | grep -q "\*" && improved=" (NEW BEST)"
            pct=$(( epoch * 100 / total ))
            echo "  $dir: epoch $epoch/$total ($pct%) | val MAE $val_mae$improved"
        fi
    fi
done
echo ""

# Best checkpoints
echo "=== BEST CHECKPOINTS ==="
for dir in v3_deftype v4_moe v6_physics v9_contrast v11_lds v12_lds_physics v2_ema v10_best_combo; do
    if [ -f "results/$dir/best.pt" ]; then
        size=$(du -h "results/$dir/best.pt" | cut -f1)
        modified=$(stat -c %Y "results/$dir/best.pt" 2>/dev/null || stat -f %m "results/$dir/best.pt" 2>/dev/null)
        echo "  $dir: $size (last updated: $(date -d @$modified +%H:%M 2>/dev/null || date -r $modified +%H:%M 2>/dev/null))"
    fi
done
echo ""

# Estimated time remaining (based on avg epoch time from last 3 epochs)
echo "=== ETA ESTIMATE ==="
for dir in v3_deftype v4_moe v6_physics v9_contrast v11_lds v12_lds_physics; do
    logfile="results/$dir/nohup.log"
    if [ -f "$logfile" ]; then
        avg_time=$(tail -3 "$logfile" | grep -oP "\d+\.\ds" | sed "s/s//" | awk "{s+=\$1; n++} END {if(n>0) printf \"%.0f\", s/n; else print 0}")
        latest=$(tail -1 "$logfile" 2>/dev/null)
        epoch=$(echo "$latest" | grep -oP "Epoch \K\d+" 2>/dev/null)
        total=$(echo "$latest" | grep -oP "Epoch \d+/\K\d+" 2>/dev/null)
        if [ -n "$epoch" ] && [ -n "$total" ] && [ "$avg_time" -gt 0 ] 2>/dev/null; then
            remaining=$(( (total - epoch) * avg_time ))
            hours=$(( remaining / 3600 ))
            mins=$(( (remaining % 3600) / 60 ))
            echo "  $dir: ~${hours}h ${mins}m remaining (${avg_time}s/epoch)"
        fi
    fi
done
'
