"""
P1: Full 44x65 Host-Dopant Screening Map
=========================================
Runs ensemble inference on all 10,641 IMP2D samples using 4 best checkpoints.
Aggregates predictions by (host, dopant) pair to produce screening matrix.
"""
import os
import sys
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import CrystalTransformer
from src.dataset import CrystalGraphDataset, collate_fn

# ============= Configuration =============
CHECKPOINT_PATHS = [
    "results/enhanced_online_150ep_uae_mae_warmup_s42/best.pt",
    "results/enhanced_online_150ep_uae_mae_warmup_s43/best.pt",
    "results/enhanced_online_150ep_uae_mae_warmup_s45/best.pt",
    "results/enhanced_online_150ep_uae_mae_warmup_deep_s42/best.pt",
]
DATA_PATH = "data/processed/cleaned_dataset.pkl"
OUTPUT_DIR = "results/p1_screening"
DEVICE = "cuda:0"  # Use with CUDA_VISIBLE_DEVICES=1
BATCH_SIZE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============= Load Data =============
sep = "=" * 60
print(sep)
print("P1: FULL HOST-DOPANT SCREENING MAP")
print(sep)
print("\n[1/5] Loading dataset...")
with open(DATA_PATH, "rb") as f:
    all_data = pickle.load(f)
print(f"  Total samples: {len(all_data)}")

dataset = CrystalGraphDataset(DATA_PATH)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=4, pin_memory=True
)
print(f"  Batches: {len(dataloader)}")

# ============= Load Models =============
print("\n[2/5] Loading ensemble models...")
models = []
normalizers = []

for ckpt_path in CHECKPOINT_PATHS:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    model_kwargs = config["model_kwargs"].copy()
    model = CrystalTransformer(**model_kwargs)
    state_dict = ckpt["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    models.append(model)
    normalizers.append(ckpt["normalizer"])
    print(f"  Loaded: {os.path.basename(os.path.dirname(ckpt_path))} (epoch {ckpt['epoch']})")

print(f"  Ensemble size: {len(models)} models on {DEVICE}")

# ============= Run Inference =============
print("\n[3/5] Running ensemble inference...")
all_predictions = [[] for _ in range(len(models))]

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx+1}/{len(dataloader)}...", flush=True)

        # Move batch tensors to device
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(DEVICE)
            else:
                batch_gpu[k] = v

        for model_idx, model in enumerate(models):
            try:
                output = model(batch_gpu)
                if isinstance(output, tuple):
                    pred = output[0]
                else:
                    pred = output

                # Denormalize
                norm = normalizers[model_idx]
                pred_np = pred.detach().cpu().numpy().flatten()
                pred_denorm = pred_np * norm["std"] + norm["mean"]
                all_predictions[model_idx].extend(pred_denorm.tolist())
            except Exception as e:
                print(f"  ERROR model {model_idx}, batch {batch_idx}: {e}")
                bs = batch_gpu["x"].shape[0]
                all_predictions[model_idx].extend([float("nan")] * bs)

n_samples = len(all_predictions[0])
print(f"  Done. Samples processed: {n_samples}")

# ============= Aggregate by (host, dopant) =============
print("\n[4/5] Aggregating by (host, dopant) pair...")

# Ensemble statistics per sample
preds_array = np.array(all_predictions)  # shape: (n_models, n_samples)
ensemble_mean = np.nanmean(preds_array, axis=0)
ensemble_std = np.nanstd(preds_array, axis=0)

# Ground truth
dft_values = np.array([d["target"] for d in all_data])

# Per-sample MAE
sample_mae = np.abs(ensemble_mean - dft_values)
print(f"  Overall ensemble MAE: {np.nanmean(sample_mae):.4f} eV")
print(f"  Overall ensemble RMSE: {np.sqrt(np.nanmean(sample_mae**2)):.4f} eV")

# Group by (host, dopant)
pair_pred_mean = defaultdict(list)
pair_pred_std = defaultdict(list)
pair_dft = defaultdict(list)

for i, sample in enumerate(all_data):
    host = sample["metadata"]["host"]
    dopant = sample["metadata"]["dopant"]
    pair_pred_mean[(host, dopant)].append(ensemble_mean[i])
    pair_pred_std[(host, dopant)].append(ensemble_std[i])
    pair_dft[(host, dopant)].append(dft_values[i])

# Build matrices
all_hosts = sorted(set(d["metadata"]["host"] for d in all_data))
all_dopants = sorted(set(d["metadata"]["dopant"] for d in all_data))

pred_matrix = np.full((len(all_hosts), len(all_dopants)), np.nan)
unc_matrix = np.full((len(all_hosts), len(all_dopants)), np.nan)
dft_matrix = np.full((len(all_hosts), len(all_dopants)), np.nan)
mae_matrix = np.full((len(all_hosts), len(all_dopants)), np.nan)
count_matrix = np.zeros((len(all_hosts), len(all_dopants)), dtype=int)

host_to_idx = {h: i for i, h in enumerate(all_hosts)}
dopant_to_idx = {d: i for i, d in enumerate(all_dopants)}

for (host, dopant), preds in pair_pred_mean.items():
    hi, di = host_to_idx[host], dopant_to_idx[dopant]
    pred_matrix[hi, di] = np.mean(preds)
    unc_matrix[hi, di] = np.mean(pair_pred_std[(host, dopant)])
    dft_matrix[hi, di] = np.mean(pair_dft[(host, dopant)])
    mae_matrix[hi, di] = np.mean(np.abs(np.array(preds) - np.array(pair_dft[(host, dopant)])))
    count_matrix[hi, di] = len(preds)

print(f"  Matrix size: {len(all_hosts)} x {len(all_dopants)}")
n_filled = int(np.sum(~np.isnan(pred_matrix)))
print(f"  Pairs with data: {n_filled}/{len(all_hosts)*len(all_dopants)}")

# ============= Save Results =============
print("\n[5/5] Saving results...")
import csv

# CSV matrices
for name, matrix in [("screening_pred_ef", pred_matrix),
                     ("screening_uncertainty", unc_matrix),
                     ("screening_dft_ef", dft_matrix),
                     ("screening_mae", mae_matrix)]:
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["host_dopant"] + all_dopants)
        for i, host in enumerate(all_hosts):
            row = [host] + [f"{matrix[i,j]:.4f}" if not np.isnan(matrix[i,j]) else ""
                           for j in range(len(all_dopants))]
            writer.writerow(row)
    print(f"  Saved: {path}")

# Sample count
path = os.path.join(OUTPUT_DIR, "sample_count.csv")
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["host_dopant"] + all_dopants)
    for i, host in enumerate(all_hosts):
        row = [host] + [str(count_matrix[i,j]) for j in range(len(all_dopants))]
        writer.writerow(row)
print(f"  Saved: {path}")

# Per-sample detailed results (JSON)
details = []
for i, sample in enumerate(all_data):
    details.append({
        "idx": i,
        "host": sample["metadata"]["host"],
        "dopant": sample["metadata"]["dopant"],
        "defect_type": sample["metadata"]["defecttype"],
        "dft_ef": float(dft_values[i]),
        "pred_ef": float(ensemble_mean[i]),
        "uncertainty": float(ensemble_std[i]),
        "abs_error": float(sample_mae[i]),
    })
path = os.path.join(OUTPUT_DIR, "per_sample_predictions.json")
with open(path, "w") as f:
    json.dump(details, f, indent=1)
print(f"  Saved: {path}")

# Summary
summary = {
    "n_hosts": len(all_hosts),
    "n_dopants": len(all_dopants),
    "n_pairs_with_data": n_filled,
    "n_pairs_total": len(all_hosts) * len(all_dopants),
    "coverage_pct": round(n_filled / (len(all_hosts) * len(all_dopants)) * 100, 1),
    "ensemble_mae": round(float(np.nanmean(sample_mae)), 4),
    "ensemble_rmse": round(float(np.sqrt(np.nanmean(sample_mae**2))), 4),
    "mean_uncertainty": round(float(np.nanmean(unc_matrix)), 4),
    "pred_ef_range": [round(float(np.nanmin(pred_matrix)), 3), round(float(np.nanmax(pred_matrix)), 3)],
    "hosts": all_hosts,
    "dopants": all_dopants,
    "checkpoints": CHECKPOINT_PATHS,
    "n_samples": n_samples,
    # Top-10 lowest predicted Ef per pair (most favorable)
    "top10_lowest_ef_pairs": sorted(
        [(h, d, round(float(pred_matrix[host_to_idx[h], dopant_to_idx[d]]), 3),
          round(float(unc_matrix[host_to_idx[h], dopant_to_idx[d]]), 3))
         for h, d in pair_pred_mean.keys()],
        key=lambda x: x[2])[:10],
}
path = os.path.join(OUTPUT_DIR, "summary.json")
with open(path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {path}")

# Print final summary
print("\n" + sep)
print("P1 SCREENING COMPLETE")
print(sep)
print(f"  Matrix: {len(all_hosts)} hosts x {len(all_dopants)} dopants")
print(f"  Coverage: {n_filled}/{len(all_hosts)*len(all_dopants)} ({summary['coverage_pct']}%)")
print(f"  Ensemble MAE: {summary['ensemble_mae']} eV")
print(f"  Ensemble RMSE: {summary['ensemble_rmse']} eV")
print(f"  Mean uncertainty: {summary['mean_uncertainty']} eV")
print(f"  Pred Ef range: [{summary['pred_ef_range'][0]}, {summary['pred_ef_range'][1]}] eV")
print(f"\n  Top-5 most favorable (lowest mean pred Ef per pair):")
for host, dopant, ef, unc in summary["top10_lowest_ef_pairs"][:5]:
    print(f"    {host} + {dopant}: Ef = {ef:.3f} eV (unc = {unc:.3f})")
print(f"\n  Results directory: {OUTPUT_DIR}/")
print(sep)
