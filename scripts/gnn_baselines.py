"""C12: SOTA GNN baselines from torch_geometric on IMP2D.

Compares three families of inductive bias on the same leak-free split:
  - SchNet              (continuous-filter conv, scalar)
  - DimeNet++           (3-body, scalar)
  - ViSNet              (E(3) equivariant vector-scalar)

Outputs
-------
- results/gnn_baselines.json
- paper/figures/fig_gnn_baselines.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, split_indices  # noqa: E402


# ── monkey-patch radius_graph (avoids torch-cluster ABI issue) ─────────
def _radius_graph_pure(x, r, batch=None, loop=False, max_num_neighbors=32,
                       flow="source_to_target", num_workers=None):
    """Vectorized batched radius graph — no torch-cluster needed.

    Computes full pairwise distances within each graph (small graphs ~30 atoms),
    masks across-graph pairs and beyond-cutoff, then returns edge_index."""
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    N = x.size(0)
    # full pairwise
    d2 = torch.cdist(x, x)                               # (N, N)
    # mask: same graph & within cutoff
    same = batch.unsqueeze(0) == batch.unsqueeze(1)      # (N, N) bool
    if not loop:
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        same = same & ~eye
    valid = same & (d2 <= r)
    # cap neighbors via topk per row (only for rows that exceed)
    if max_num_neighbors > 0:
        # rank per row; mark beyond-topk as invalid
        d_for_sort = torch.where(valid, d2, torch.full_like(d2, float("inf")))
        _, topk = torch.topk(d_for_sort, k=min(max_num_neighbors, N),
                              dim=1, largest=False)
        keep = torch.zeros_like(valid)
        keep.scatter_(1, topk, True)
        valid = valid & keep
    rows, cols = valid.nonzero(as_tuple=True)
    return torch.stack([rows, cols], dim=0)


# patch torch_geometric
import torch_geometric.nn.pool as _pyg_pool  # noqa: E402
_pyg_pool.radius_graph = _radius_graph_pure
import torch_geometric.nn.models.schnet as _schnet_mod  # noqa: E402
_schnet_mod.radius_graph = _radius_graph_pure
import torch_geometric.nn.models.dimenet as _dimenet_mod  # noqa: E402
_dimenet_mod.radius_graph = _radius_graph_pure
import torch_geometric.nn.models.visnet as _visnet_mod  # noqa: E402
_visnet_mod.radius_graph = _radius_graph_pure

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
SEED = 42


# ── PyG-flavoured collate ─────────────────────────────────────────────
def collate_pyg(samples):
    """Return (z, pos, batch_idx, target) where z is concat atomic numbers."""
    z_all, pos_all, batch_idx, targets = [], [], [], []
    for i, s in enumerate(samples):
        z = torch.as_tensor(s["numbers"], dtype=torch.long)
        pos = torch.as_tensor(s["positions"], dtype=torch.float32)
        z_all.append(z)
        pos_all.append(pos)
        batch_idx.append(torch.full((z.numel(),), i, dtype=torch.long))
        targets.append(float(s["target"]))
    return (
        torch.cat(z_all, dim=0),
        torch.cat(pos_all, dim=0),
        torch.cat(batch_idx, dim=0),
        torch.tensor(targets, dtype=torch.float32),
    )


# ── eval / train helpers ──────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model, z, pos, batch, model_name):
    if model_name in ("SchNet", "DimeNet++"):
        return model(z, pos, batch).squeeze(-1)
    elif model_name == "ViSNet":
        x, _v = model(z, pos, batch)
        return x.squeeze(-1)
    else:
        raise ValueError(model_name)


def evaluate(model, loader, device, name):
    model.eval()
    err_sum, n = 0.0, 0
    with torch.no_grad():
        for z, pos, b, y in loader:
            z, pos, b, y = z.to(device), pos.to(device), b.to(device), y.to(device)
            pred = predict(model, z, pos, b, name)
            err_sum += (pred - y).abs().sum().item()
            n += y.numel()
    return err_sum / max(n, 1)


def train_one(model, train_loader, val_loader, device, name, epochs):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()
    best_val, bad = float("inf"), 0
    best_state = None
    history = []
    for ep in range(epochs):
        ep_start = time.time()
        model.train()
        for z, pos, b, y in train_loader:
            z, pos, b, y = z.to(device), pos.to(device), b.to(device), y.to(device)
            pred = predict(model, z, pos, b, name)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        val_mae = evaluate(model, val_loader, device, name)
        history.append({"epoch": ep, "val_mae": val_mae})
        if ep < 3 or ep % 5 == 0 or ep == epochs - 1:
            print(f"    ep {ep:2d}  val_mae={val_mae:.4f}  ({time.time()-ep_start:.0f}s)", flush=True)
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"    early stop at ep {ep}, best_val={best_val:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def build_models():
    from torch_geometric.nn.models import SchNet, DimeNetPlusPlus, ViSNet

    schnet = SchNet(
        hidden_channels=128, num_filters=128,
        num_interactions=6, num_gaussians=50,
        cutoff=8.0, max_num_neighbors=32, readout="add",
    )

    dimenetpp = DimeNetPlusPlus(
        hidden_channels=128, out_channels=1,
        num_blocks=4, int_emb_size=64,
        basis_emb_size=8, out_emb_channels=256,
        num_spherical=7, num_radial=6,
        cutoff=5.0, max_num_neighbors=32,
    )

    visnet = ViSNet(
        lmax=1, num_heads=8, num_layers=6,
        hidden_channels=128, num_rbf=32,
        cutoff=5.0, max_num_neighbors=32,
    )
    return [("SchNet", schnet), ("DimeNet++", dimenetpp), ("ViSNet", visnet)]


def main():
    t_start = time.time()
    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(SEED)

    dataset = CrystalGraphDataset(DATA_PATH)
    train_idx, val_idx, test_idx = split_indices(len(dataset), 0.8, 0.1, SEED)
    print(f"Splits: train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")

    train_set = [dataset.data[i] for i in train_idx]
    val_set = [dataset.data[i] for i in val_idx]
    test_set = [dataset.data[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_pyg)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_pyg)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_pyg)

    models = build_models()
    results = []
    for name, model in models:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {name}  ({n_params/1e6:.3f}M params) ===")
        t0 = time.time()
        try:
            model, hist = train_one(model, train_loader, val_loader, device,
                                    name, EPOCHS)
            test_mae = evaluate(model, test_loader, device, name)
            elapsed = time.time() - t0
            print(f"  Test MAE = {test_mae:.4f}  ({elapsed:.0f}s)")
            results.append({
                "model": name,
                "n_params": int(n_params),
                "n_params_M": round(n_params / 1e6, 3),
                "test_mae": float(test_mae),
                "wall_sec": float(elapsed),
                "epochs_run": len(hist),
                "best_val_mae": float(min(h["val_mae"] for h in hist)),
            })
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "model": name,
                "n_params": int(n_params),
                "n_params_M": round(n_params / 1e6, 3),
                "error": f"{type(e).__name__}: {e}",
            })

        # incremental save
        with open(RESULTS / "gnn_baselines.json", "w") as f:
            json.dump({"results": results,
                       "config": {
                           "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                           "lr": LR, "seed": SEED,
                       },
                       "wall_time_min": (time.time() - t_start) / 60},
                      f, indent=2)

    # add reference rows
    results.append({
        "model": "CrystalTransformer (ours)",
        "n_params": 747000, "n_params_M": 0.747,
        "test_mae": 0.516, "wall_sec": 720, "note": "leak-free aug long",
    })
    results.append({
        "model": "ALIGNN (ref)",
        "n_params": 4030000, "n_params_M": 4.03,
        "test_mae": 0.540, "wall_sec": None, "note": "team prior reproduction",
    })

    out = {"results": results,
           "config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                      "lr": LR, "seed": SEED},
           "wall_time_min": (time.time() - t_start) / 60}
    with open(RESULTS / "gnn_baselines.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {RESULTS / 'gnn_baselines.json'}")

    # figure
    plot = [r for r in results if "test_mae" in r]
    plot.sort(key=lambda r: r["test_mae"])
    names = [r["model"] for r in plot]
    maes = [r["test_mae"] for r in plot]
    params = [r["n_params_M"] for r in plot]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(names, maes, color=["#1f77b4" if "ours" not in n.lower()
                                        and "ALIGNN" not in n else "#d62728"
                                        for n in names])
    ax.set_xlabel("Test MAE (eV)", fontsize=11)
    ax.set_title(f"GNN baselines on IMP2D (same leak-free split, {EPOCHS} ep)",
                 fontsize=12)
    for bar, m, p in zip(bars, maes, params):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}  ({p:.2f}M)", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_fig = FIG_DIR / "fig_gnn_baselines.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"figure saved -> {out_fig}")
    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
