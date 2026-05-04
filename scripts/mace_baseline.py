"""C11: MACE equivariant baseline on IMP2D, same leak-free split as §5.1.

Trains a small E(3)-equivariant MACE model from scratch and reports
test MAE for direct comparison with our CrystalTransformer / SchNet /
ViSNet (§5.16). Closes the open question of whether higher-order
equivariance (max_ell ≥ 2 with tensor products) helps the scalar
defect-formation-energy regression.

Outputs
-------
- results/mace_baseline.json
- paper/figures/fig_mace_baseline.png
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# -- mace imports --
from mace.data import AtomicData, Configuration
from mace.modules import (
    MACE,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.utils import compute_avg_num_neighbors
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_atomic_energies
from mace.tools.utils import AtomicNumberTable
from e3nn.o3 import Irreps

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import split_indices  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────
R_MAX = 5.0
EPOCHS = 60
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
GRAD_CLIP = 5.0
SEED = 42


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_configuration(sample, z_table):
    """Convert one IMP2D sample dict into a mace.data.Configuration."""
    z = sample["numbers"].astype(np.int64)
    pos = sample["positions"].astype(np.float32)
    cell = sample["cell"].astype(np.float32)
    return Configuration(
        atomic_numbers=z,
        positions=pos,
        properties={"energy": float(sample["target"])},
        property_weights={"energy": 1.0},
        cell=cell,
        pbc=(True, True, True),
        config_type="Default",
        head="Default",
    )


def build_dataset(samples, z_table, r_max):
    out = []
    for s in samples:
        cfg = build_configuration(s, z_table)
        try:
            ad = AtomicData.from_config(cfg, z_table=z_table, cutoff=r_max)
            out.append(ad)
        except Exception as e:
            # skip pathological samples (e.g. zero edges)
            continue
    return out


def evaluate(model, loader, device, mean, std):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.to_dict(), training=False, compute_force=False,
                         compute_stress=False, compute_virials=False,
                         compute_displacement=False)
            # de-normalise
            pred = out["energy"].detach().cpu().numpy() * std + mean
            true = batch.energy.detach().cpu().numpy() * std + mean
            preds.append(pred)
            trues.append(true)
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    return float(np.abs(p - t).mean()), p, t


def main():
    t_start = time.time()
    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load samples ────────────────────────────────────────────────
    print(f"Loading {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        samples = pickle.load(f)
    train_idx, val_idx, test_idx = split_indices(len(samples), 0.8, 0.1, SEED)
    print(f"  splits: train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    # ── Build z_table ───────────────────────────────────────────────
    all_z = set()
    for s in samples:
        all_z.update(int(z) for z in s["numbers"])
    z_table = AtomicNumberTable(sorted(all_z))
    print(f"  unique elements: {len(z_table.zs)} (min Z={min(z_table.zs)} max Z={max(z_table.zs)})")

    # ── Normalise targets ───────────────────────────────────────────
    train_targets = np.array([s["target"] for s in train_samples])
    mean = float(train_targets.mean())
    std = float(train_targets.std() + 1e-6)
    print(f"  train target μ={mean:.3f}  σ={std:.3f}")

    # apply normalisation in-place
    for s in train_samples + val_samples + test_samples:
        s["target_orig"] = float(s["target"])
        s["target"] = (float(s["target"]) - mean) / std

    print("Building MACE datasets (this may take a minute)...")
    t0 = time.time()
    train_data = build_dataset(train_samples, z_table, R_MAX)
    val_data = build_dataset(val_samples, z_table, R_MAX)
    test_data = build_dataset(test_samples, z_table, R_MAX)
    print(f"  train {len(train_data)}, val {len(val_data)}, test {len(test_data)}  "
          f"({time.time()-t0:.0f}s)")

    train_loader = torch_geometric.dataloader.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = torch_geometric.dataloader.DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch_geometric.dataloader.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False)

    # ── Compute average number of neighbours ────────────────────────
    avg_n_neigh = compute_avg_num_neighbors(train_loader)
    print(f"  avg_num_neighbors = {avg_n_neigh:.2f}")

    # ── Build MACE model ────────────────────────────────────────────
    # Small but capable: 32 scalar + 32 vector + 32 l=2 channels
    hidden_irreps = Irreps("32x0e + 32x1o")
    mlp_irreps = Irreps("16x0e")

    atomic_energies = np.zeros(len(z_table.zs), dtype=np.float64)

    model = MACE(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=2,
        num_elements=len(z_table.zs),
        hidden_irreps=hidden_irreps,
        MLP_irreps=mlp_irreps,
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_n_neigh,
        atomic_numbers=z_table.zs,
        correlation=3,
        gate=torch.nn.functional.silu,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MACE: {n_params/1e6:.3f}M parameters")

    # ── Optimiser ────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.SmoothL1Loss()

    best_val, bad = float("inf"), 0
    best_state = None
    history = []
    for ep in range(EPOCHS):
        ep_start = time.time()
        model.train()
        n_batches = 0
        loss_sum = 0.0
        n_skip = 0
        for batch in train_loader:
            batch = batch.to(device)
            try:
                out = model(batch.to_dict(), training=True,
                            compute_force=False, compute_stress=False,
                            compute_virials=False, compute_displacement=False)
            except Exception as e:
                print(f"  WARN forward failed batch: {e}")
                n_skip += 1
                continue
            pred = out["energy"]
            target = batch.energy
            loss = loss_fn(pred, target)
            if not torch.isfinite(loss):
                n_skip += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            loss_sum += float(loss.item())
            n_batches += 1
        sched.step()

        val_mae_norm, _, _ = evaluate(model, val_loader, device, mean=0, std=1)
        # de-normalise val MAE for display
        val_mae = val_mae_norm * std

        history.append({
            "epoch": ep,
            "train_loss": loss_sum / max(n_batches, 1),
            "val_mae_eV": val_mae,
        })
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep < 3 or ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  ep {ep:2d}  train_loss={loss_sum/max(n_batches,1):.4f}  "
                  f"val_mae={val_mae:.4f} eV  skip={n_skip}  "
                  f"({time.time()-ep_start:.0f}s)", flush=True)

        if bad >= PATIENCE:
            print(f"  early stop at ep {ep}, best_val={best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Final test evaluation ────────────────────────────────────────
    test_mae_norm, preds, trues = evaluate(model, test_loader, device,
                                            mean=0, std=1)
    test_mae = test_mae_norm * std
    test_rmse = float(np.sqrt(((preds - trues) ** 2).mean())) * std
    print(f"\n=== MACE final test results ===")
    print(f"  Test MAE  = {test_mae:.4f} eV")
    print(f"  Test RMSE = {test_rmse:.4f} eV")
    print(f"  Wall time = {(time.time()-t_start)/60:.1f} min")

    # ── Save ─────────────────────────────────────────────────────────
    out = {
        "model": "MACE (small, max_ell=2, 2 interactions)",
        "n_params": int(n_params),
        "n_params_M": round(n_params / 1e6, 3),
        "test_mae_eV": float(test_mae),
        "test_rmse_eV": float(test_rmse),
        "best_val_mae_eV": float(best_val),
        "wall_time_min": (time.time() - t_start) / 60,
        "config": {
            "r_max": R_MAX,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_ell": 2,
            "num_interactions": 2,
            "hidden_irreps": "32x0e + 32x1o",
            "correlation": 3,
            "n_elements": len(z_table.zs),
            "avg_num_neighbors": float(avg_n_neigh),
            "target_mean": mean,
            "target_std": std,
        },
        "history": history,
    }
    out_json = RESULTS / "mace_baseline.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_json}")

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    eps = [h["epoch"] for h in history]
    val_maes = [h["val_mae_eV"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    axes[0].plot(eps, val_maes, "-", color="#1f77b4", lw=2, label="Val MAE (eV)")
    axes[0].plot(eps, train_losses, "--", color="#aaa", lw=1, label="Train SmoothL1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val MAE (eV)")
    axes[0].set_title(f"MACE training curve (best={best_val:.3f})")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Bar: comparison vs other GNN baselines
    others = {
        "LightGBM\n(classical)": 1.158,
        "SchNet\n(0.46M)": 0.585,
        "ViSNet\n(1.16M, lmax=1)": 0.86,
        f"MACE\n({n_params/1e6:.2f}M, lmax=2)": test_mae,
        "ALIGNN\n(4.0M)": 0.540,
        "Ours\n(0.75M)": 0.516,
    }
    keys = list(others.keys())
    vals = list(others.values())
    colors = ["#888"] + ["#1f77b4"] * 3 + ["#1f77b4", "#d62728"]
    bars = axes[1].bar(keys, vals, color=colors)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02,
                     f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_ylabel("Test MAE (eV)")
    axes[1].set_title("Architecture comparison (same leak-free split)")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_fig = FIG_DIR / "fig_mace_baseline.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"figure saved -> {out_fig}")


if __name__ == "__main__":
    main()
