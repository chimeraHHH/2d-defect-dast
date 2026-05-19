import sys, os, json, time
import numpy as np
import torch
from pathlib import Path

ROOT = Path(os.path.expanduser("~/Workspace/yiminghua/project"))
sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.models import CrystalTransformerV2
from src.train_enhanced import Normalizer, move_batch

device = torch.device("cuda:0")

dataset = CrystalGraphDataset(ROOT / "data/processed/cleaned_dataset.pkl")
_, _, test_set = make_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=64, shuffle=False, collate_fn=collate_fn
)

ckpt_paths = [
    ROOT / "results/v2_gated_pooling/best.pt",
    ROOT / "results/v2_gated_pooling_s43/best.pt",
    ROOT / "results/v2_gated_pooling_s44/best.pt",
    ROOT / "results/v2_gated_pooling_s45/best.pt",
]

model_kwargs = dict(
    atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2,
    num_heads=4, rcut_local=5.0, dmax_global=12.0, defect_embedding=True,
    dropout=0.0, ct_uae_path=str(ROOT / "data/ct_uae_mt3_embeddings.pt"),
    use_gated_pooling=True, use_env_enrichment=True, use_prenorm_local=True,
)

models = []
normalizers = []
for p in ckpt_paths:
    ckpt = torch.load(p, map_location=device, weights_only=False)
    model = CrystalTransformerV2(**model_kwargs)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    models.append(model)
    
    norm = Normalizer(torch.zeros(1))
    norm.mean = ckpt["normalizer"]["mean"]
    norm.std = ckpt["normalizer"]["std"]
    normalizers.append(norm)
    print(f"Loaded {p.parent.name} (norm: mean={norm.mean:.4f}, std={norm.std:.4f})")

all_preds = [[] for _ in range(len(models))]
all_targets = []

t0 = time.time()
with torch.no_grad():
    for batch in test_loader:
        batch = move_batch(batch, device)
        target = batch["target"]
        all_targets.append(target.cpu().numpy())
        for i, (model, norm) in enumerate(zip(models, normalizers)):
            pred_norm = model(batch)
            pred = norm.denorm(pred_norm)
            all_preds[i].append(pred.cpu().numpy())

dt = time.time() - t0
print(f"\nInference time: {dt:.1f}s")

targets = np.concatenate(all_targets)
preds_per_model = [np.concatenate(p) for p in all_preds]

print("\n=== Individual Model Metrics ===")
for i, (preds, path) in enumerate(zip(preds_per_model, ckpt_paths)):
    mae = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets)**2).mean())
    print(f"  {path.parent.name}: MAE={mae:.4f} RMSE={rmse:.4f}")

print("\n=== Ensemble Metrics ===")
for k in range(2, len(models) + 1):
    ens_pred = np.mean(preds_per_model[:k], axis=0)
    mae = np.abs(ens_pred - targets).mean()
    rmse = np.sqrt(((ens_pred - targets)**2).mean())
    print(f"  {k}-model ensemble: MAE={mae:.4f} RMSE={rmse:.4f}")

print("\n=== Greedy Best Ensemble ===")
best_mae = float("inf")
best_combo = []
remaining = list(range(len(models)))
for step in range(len(models)):
    best_i, best_mae_step = -1, float("inf")
    for i in remaining:
        combo = best_combo + [i]
        ens = np.mean([preds_per_model[j] for j in combo], axis=0)
        mae = np.abs(ens - targets).mean()
        if mae < best_mae_step:
            best_mae_step = mae
            best_i = i
    best_combo.append(best_i)
    remaining.remove(best_i)
    ens = np.mean([preds_per_model[j] for j in best_combo], axis=0)
    mae = np.abs(ens - targets).mean()
    rmse = np.sqrt(((ens - targets)**2).mean())
    names = [ckpt_paths[j].parent.name for j in best_combo]
    print(f"  k={len(best_combo)}: MAE={mae:.4f} RMSE={rmse:.4f}")

ens_4 = np.mean(preds_per_model, axis=0)
ens_std = np.std(preds_per_model, axis=0)
ens_err = np.abs(ens_4 - targets)
corr = np.corrcoef(ens_std, ens_err)[0, 1]
print(f"\nUncertainty: mean_std={ens_std.mean():.4f}, corr(std,|err|)={corr:.3f}")

results = {
    "individual_mae": {ckpt_paths[i].parent.name: float(np.abs(preds_per_model[i] - targets).mean()) 
                       for i in range(len(models))},
    "ensemble_4_mae": float(np.abs(ens_4 - targets).mean()),
    "ensemble_4_rmse": float(np.sqrt(((ens_4 - targets)**2).mean())),
    "uncertainty_corr": float(corr),
    "n_test": len(targets),
}
with open(str(ROOT / "results/v2_ensemble_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved")
