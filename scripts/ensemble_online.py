"""Ensemble evaluation: load all checkpoints, predict on the SAME test set (seed=42)."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.models import CrystalTransformer


def load_model(ckpt_path, model_kwargs, device):
    model = CrystalTransformer(**model_kwargs)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:
        model.load_state_dict(state["model"])
    elif "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    return model, state


def predict(model, loader, normalizer, device):
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            p = normalizer.denorm(out)
            preds.append(p.cpu())
            targets.append(batch["target"].cpu())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


class Normalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def denorm(self, x):
        return x * self.std + self.mean


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CrystalGraphDataset(ROOT / "data/processed/cleaned_dataset.pkl")
    _, _, test_set = make_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42)
    train_set, _, _ = make_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

    base_kwargs = dict(
        atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2,
        num_heads=4, rcut_local=5.0, dmax_global=12.0, defect_embedding=True, dropout=0.0)

    uae_kwargs = dict(**base_kwargs, ct_uae_path=str(ROOT / "data/ct_uae_mt3_embeddings.pt"))

    runs = {
        "100ep_s42": ("enhanced_online_100ep_s42", base_kwargs),
        "100ep_s43": ("enhanced_online_100ep_s43", base_kwargs),
        "100ep_s44": ("enhanced_online_100ep_s44", base_kwargs),
        "uae_s42": ("enhanced_online_100ep_uae_s42", uae_kwargs),
        "uae_s43": ("enhanced_online_100ep_uae_s43", uae_kwargs),
        "uae_s44": ("enhanced_online_100ep_uae_s44", uae_kwargs),
        "uae_huber_s42": ("enhanced_online_100ep_uae_huber_s42", uae_kwargs),
        "uae_huber_s43": ("enhanced_online_100ep_uae_huber_s43", uae_kwargs),
        "uae_huber_s44": ("enhanced_online_100ep_uae_huber_s44", uae_kwargs),
        "uae_mae_s42": ("enhanced_online_100ep_uae_mae_s42", uae_kwargs),
        "uae_mae_s43": ("enhanced_online_100ep_uae_mae_s43", uae_kwargs),
        "uae_mae_s44": ("enhanced_online_100ep_uae_mae_s44", uae_kwargs),
        "uae_mae_warmup_s42": ("enhanced_online_100ep_uae_mae_warmup_s42", uae_kwargs),
        "uae_mae_warmup_s43": ("enhanced_online_100ep_uae_mae_warmup_s43", uae_kwargs),
    }

    all_preds = {}
    targets = None

    for name, (run_dir, model_kwargs) in runs.items():
        ckpt = ROOT / "results" / run_dir / "best.pt"
        if not ckpt.exists():
            print(f"  SKIP {name}: no checkpoint")
            continue

        model, state = load_model(ckpt, model_kwargs, device)

        norm_info = state.get("normalizer", {})
        norm_mean = norm_info.get("mean", None)
        norm_std = norm_info.get("std", None)
        if norm_mean is None:
            metrics = ROOT / "results" / run_dir / "metrics.json"
            if metrics.exists():
                m = json.loads(metrics.read_text())
                norm_mean = m.get("target_mean", 0)
                norm_std = m.get("target_std", 1)
            else:
                print(f"  SKIP {name}: no normalizer info")
                continue

        normalizer = Normalizer(
            mean=torch.tensor(norm_mean, dtype=torch.float32).to(device),
            std=torch.tensor(norm_std, dtype=torch.float32).to(device))

        p, t = predict(model, test_loader, normalizer, device)
        all_preds[name] = p
        if targets is None:
            targets = t
        mae = np.abs(p - t).mean()
        print(f"  {name:12s}: MAE {mae:.4f}")

    if len(all_preds) < 2:
        print("Not enough models for ensemble")
        return

    print(f"\n=== Ensemble ({len(all_preds)} models) ===")
    from itertools import combinations
    names = list(all_preds.keys())

    for k in range(2, len(names) + 1):
        best_mae, best_combo = 999, None
        for combo in combinations(names, k):
            P = np.stack([all_preds[n] for n in combo])
            mu = P.mean(axis=0)
            mae = np.abs(mu - targets).mean()
            if mae < best_mae:
                best_mae = mae
                best_combo = combo
        print(f"  k={k}: best MAE {best_mae:.4f} <- {list(best_combo)}")

    P_all = np.stack(list(all_preds.values()))
    mu = P_all.mean(axis=0)
    sigma = P_all.std(axis=0, ddof=1)
    mae = np.abs(mu - targets).mean()
    rmse = np.sqrt(((mu - targets) ** 2).mean())
    corr = np.corrcoef(sigma, np.abs(mu - targets))[0, 1]
    print(f"\n  Full ensemble: MAE {mae:.4f} | RMSE {rmse:.4f} | corr(σ,|err|) {corr:.3f}")

    np.savez(ROOT / "results" / "ensemble_online.npz",
             preds=mu, targets=targets, sigma=sigma,
             individual_preds=P_all,
             model_names=np.array(names))
    print(f"\n  Saved -> results/ensemble_online.npz")


if __name__ == "__main__":
    main()
