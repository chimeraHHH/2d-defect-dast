"""C17 Stage 4: Augmented training with 4 strategies.

Tests whether adding pseudo-labeled candidates can break the data bottleneck
identified in §5.17 (β ≈ 0). Compares:

  Strategy A: Baseline (no augmentation, real train only)
  Strategy B: Random 100 candidates as pseudo (with ensemble μ label)
  Strategy C: Adversarial 100 candidates (top-σ) as pseudo
  Strategy D: Confidence-filtered 20 candidates (σ_cal < 2.0) as pseudo

All strategies share the same architecture, lr, epoch budget.
Pseudo samples are down-weighted by exp(-σ_cal / scale) to penalize
unreliable labels.

Outputs
-------
- results/c17_augmented_training.json
- paper/figures/fig_c17_augmented.png
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import (CrystalGraphDataset, collate_fn,  # noqa: E402
                          get_atom_feature_table, split_indices)
from src.models import CrystalTransformer  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
CAND_PATH = ROOT / "data" / "processed" / "candidates_c17.pkl"
PRED_PATH = ROOT / "results" / "candidates_c17_predictions.json"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 60
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
SEED = 42
SIGMA_SCALE = 2.0   # for weighting:  w_pseudo = exp(-σ_cal / SIGMA_SCALE)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model():
    return CrystalTransformer(
        atom_fea_len=9, hidden_dim=128,
        n_local_layers=3, n_global_layers=2,
        num_heads=4, dropout=0.1,
    )


class WeightedConcatDataset(Dataset):
    """Concat real + pseudo with per-sample weights (real weight=1)."""
    def __init__(self, real_ds, real_idx, pseudo_ds, pseudo_idx,
                 pseudo_weights):
        self.real_ds = real_ds
        self.real_idx = list(real_idx)
        self.pseudo_ds = pseudo_ds
        self.pseudo_idx = list(pseudo_idx)
        self.pseudo_weights = list(pseudo_weights)

    def __len__(self):
        return len(self.real_idx) + len(self.pseudo_idx)

    def __getitem__(self, i):
        if i < len(self.real_idx):
            d = self.real_ds[self.real_idx[i]]
            return {**d, "sample_weight": torch.tensor(1.0, dtype=torch.float32)}
        else:
            j = i - len(self.real_idx)
            d = self.pseudo_ds[self.pseudo_idx[j]]
            return {**d, "sample_weight": torch.tensor(
                self.pseudo_weights[j], dtype=torch.float32)}


def collate_fn_weighted(batch):
    weights = torch.stack([b["sample_weight"] for b in batch])
    out = collate_fn([{k: v for k, v in b.items() if k != "sample_weight"}
                      for b in batch])
    out["sample_weight"] = weights
    return out


def evaluate(model, loader, device, mean, std):
    model.eval()
    err, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            pred = model(batch) * std + mean
            target = batch["target"]   # already in eV
            err += (pred - target).abs().sum().item()
            n += target.numel()
    return err / max(n, 1)


def train_one(model, train_loader, val_loader, device, mean, std, name):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    best_val, bad = float("inf"), 0
    best_state = None
    history = []
    for ep in range(EPOCHS):
        ep_start = time.time()
        model.train()
        loss_sum, n_seen = 0.0, 0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            pred = model(batch)
            target_norm = (batch["target"] - mean) / std
            w = batch["sample_weight"]
            losses = loss_fn(pred, target_norm)
            loss = (losses * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.item()) * len(target_norm)
            n_seen += len(target_norm)
        sched.step()

        val_mae = evaluate(model, val_loader, device, mean, std)
        history.append({"epoch": ep, "val_mae_eV": val_mae,
                        "train_loss": loss_sum / max(n_seen, 1)})
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep < 3 or ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"    [{name}] ep {ep:2d}  loss={loss_sum/max(n_seen,1):.4f}  "
                  f"val_mae={val_mae:.4f}  ({time.time()-ep_start:.0f}s)",
                  flush=True)
        if bad >= PATIENCE:
            print(f"    [{name}] early stop at ep {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


def main():
    t_start = time.time()
    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load real ──────────────────────────────────────────────────
    print(f"Loading real IMP2D ...")
    real = CrystalGraphDataset(DATA_PATH)
    train_idx, val_idx, test_idx = split_indices(len(real), 0.8, 0.1, SEED)
    print(f"  splits: train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")

    train_targets = np.array([real.data[i]["target"] for i in train_idx])
    mean = float(train_targets.mean())
    std = float(train_targets.std() + 1e-6)
    print(f"  train target μ={mean:.3f}  σ={std:.3f}")

    val_loader = DataLoader(
        WeightedConcatDataset(real, val_idx, real, [], []),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_weighted)
    test_loader = DataLoader(
        WeightedConcatDataset(real, test_idx, real, [], []),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_weighted)

    # ── Load candidates + predictions ──────────────────────────────
    print(f"Loading candidates + predictions ...")
    cand_ds = CrystalGraphDataset.__new__(CrystalGraphDataset)
    with open(CAND_PATH, "rb") as f:
        cand_ds.data = pickle.load(f)
    cand_ds.meta = None
    cand_ds.atom_features = get_atom_feature_table(None)
    cand_ds.defect_mark_neighbors = 0
    print(f"  {len(cand_ds.data)} candidates")

    preds = json.load(open(PRED_PATH))
    pred_arr = preds["predictions"]
    cand_mu = np.array([p["mu"] for p in pred_arr])
    cand_sigma = np.array([p["sigma_cal"] for p in pred_arr])

    # set the target on each candidate to ensemble μ (in eV)
    for s, mu_i in zip(cand_ds.data, cand_mu):
        s["target"] = float(mu_i)

    # weight = exp(-σ / scale)
    cand_w = np.exp(-cand_sigma / SIGMA_SCALE)
    print(f"  pseudo weights: min={cand_w.min():.3f}, max={cand_w.max():.3f}, "
          f"mean={cand_w.mean():.3f}")

    # ── Define strategies ──────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    n_cand = len(cand_ds.data)

    strategies = {}

    # A: baseline (no aug)
    strategies["A_baseline"] = ([], [])

    # B: random 100
    rand100 = rng.choice(n_cand, size=min(100, n_cand), replace=False).tolist()
    strategies["B_random100"] = (rand100, cand_w[rand100].tolist())

    # C: top-σ 100 (most adversarial)
    adv100 = list(np.argsort(-cand_sigma)[:min(100, n_cand)])
    strategies["C_adversarial100"] = (adv100, cand_w[adv100].tolist())

    # D: confidence-filtered (σ < 2)
    conf_idx = list(np.where(cand_sigma < 2.0)[0])
    strategies["D_conf_filtered"] = (conf_idx, cand_w[conf_idx].tolist())

    print(f"\nStrategies:")
    for k, (idx, w) in strategies.items():
        if w:
            print(f"  {k}: n_pseudo={len(idx)}, mean_w={np.mean(w):.3f}")
        else:
            print(f"  {k}: no pseudo")

    # ── Run all strategies ─────────────────────────────────────────
    results = {}
    for name, (cand_idx, cand_weights) in strategies.items():
        print(f"\n=== Strategy {name} ===")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        train_ds = WeightedConcatDataset(
            real, train_idx, cand_ds, cand_idx, cand_weights)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn_weighted)
        model = make_model()
        model, hist, best_val = train_one(model, train_loader, val_loader,
                                          device, mean, std, name)
        test_mae = evaluate(model, test_loader, device, mean, std)
        print(f"  → Test MAE = {test_mae:.4f} eV")
        results[name] = {
            "n_real": len(train_idx),
            "n_pseudo": len(cand_idx),
            "best_val_mae_eV": float(best_val),
            "test_mae_eV": float(test_mae),
            "history": hist,
        }
        # incremental save
        with open(RESULTS / "c17_augmented_training.json", "w") as f:
            json.dump({
                "config": {"epochs": EPOCHS, "lr": LR,
                           "sigma_scale": SIGMA_SCALE, "seed": SEED},
                "candidate_pred_path": str(PRED_PATH.name),
                "results": results,
                "wall_time_min": (time.time() - t_start) / 60,
            }, f, indent=2)

    # ── Summary + figure ────────────────────────────────────────────
    print(f"\n=== Summary ===")
    for name, r in results.items():
        delta = (r["test_mae_eV"] - results["A_baseline"]["test_mae_eV"]) / results["A_baseline"]["test_mae_eV"] * 100
        print(f"  {name}: Test MAE {r['test_mae_eV']:.4f} eV  Δ={delta:+.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(results.keys())
    vals = [results[n]["test_mae_eV"] for n in names]
    colors = ["#888"] + ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.4f}", ha="center", fontsize=9)
    ax.axhline(results["A_baseline"]["test_mae_eV"], color="gray",
               ls="--", alpha=0.5, label="baseline")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("C17: Self-distillation augmentation strategies")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_fig = FIG_DIR / "fig_c17_augmented.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"figure saved -> {out_fig}")
    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
