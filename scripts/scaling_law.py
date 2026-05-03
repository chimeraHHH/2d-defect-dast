"""C15: Empirical scaling laws for CrystalTransformer on IMP2D.

Sweeps train-set size (N) x model hidden dim (H) and fits power-law
exponents to test MAE.  Standard "AI4Science" finding format:
   MAE ~ A * N^{-alpha} * H^{-beta}

Outputs
-------
- results/scaling_law.json
- paper/figures/fig_scaling_law.png
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, split_indices  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# sweep grid
DATA_SIZES = [500, 1000, 2000, 4000, 8000]
HIDDEN_DIMS = [64, 96, 128, 192]
EPOCHS = 30                  # short but enough to converge with early stop
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
SEED = 42


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(hidden_dim: int) -> nn.Module:
    # scale local/global layers proportionally with hidden_dim band
    if hidden_dim <= 80:
        n_local, n_global, num_heads = 2, 1, 2
    elif hidden_dim <= 128:
        n_local, n_global, num_heads = 3, 2, 4
    else:
        n_local, n_global, num_heads = 3, 2, 4
    return CrystalTransformer(
        atom_fea_len=9,
        hidden_dim=hidden_dim,
        n_local_layers=n_local,
        n_global_layers=n_global,
        num_heads=num_heads,
        dropout=0.1,
    )


def evaluate(model, loader, device):
    model.eval()
    mae = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            pred = model(batch)
            tgt = batch["target"]
            mae += (pred - tgt).abs().sum().item()
            n += tgt.numel()
    return mae / max(n, 1)


def train_one(model, train_loader, val_loader, device, max_epochs):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.SmoothL1Loss()
    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            pred = model(batch)
            loss = loss_fn(pred, batch["target"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        val_mae = evaluate(model, val_loader, device)
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ep + 1


def main():
    t_start = time.time()
    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading dataset: {DATA_PATH}")
    dataset = CrystalGraphDataset(DATA_PATH)
    train_idx, val_idx, test_idx = split_indices(len(dataset), 0.8, 0.1, SEED)
    print(f"  splits: train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")

    val_loader = DataLoader(Subset(dataset, val_idx),
                            batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(Subset(dataset, test_idx),
                             batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn)

    rng = np.random.default_rng(SEED)
    runs = []
    grid_total = len(DATA_SIZES) * len(HIDDEN_DIMS)
    grid_done = 0

    for n_train in DATA_SIZES:
        if n_train > len(train_idx):
            continue
        # subsample train set deterministically
        sub = rng.permutation(train_idx)[:n_train].tolist()
        sub_loader = DataLoader(Subset(dataset, sub),
                                batch_size=min(BATCH_SIZE, max(1, n_train // 8)),
                                shuffle=True, collate_fn=collate_fn,
                                drop_last=False)
        for hidden in HIDDEN_DIMS:
            grid_done += 1
            t0 = time.time()
            model = make_model(hidden)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n[{grid_done}/{grid_total}] N={n_train}  H={hidden}  "
                  f"params={n_params/1e6:.3f}M")
            model, n_ep = train_one(model, sub_loader, val_loader, device,
                                    EPOCHS)
            test_mae = evaluate(model, test_loader, device)
            elapsed = time.time() - t0
            print(f"  -> test MAE = {test_mae:.4f}  ({n_ep} ep, {elapsed:.0f}s)")
            runs.append({
                "n_train": int(n_train),
                "hidden_dim": int(hidden),
                "n_params": int(n_params),
                "n_params_M": round(n_params / 1e6, 4),
                "epochs_run": int(n_ep),
                "test_mae": float(test_mae),
                "wall_sec": float(elapsed),
            })
            # incremental save
            with open(RESULTS / "scaling_law.json", "w") as f:
                json.dump({"runs": runs,
                           "config": {
                               "data_sizes": DATA_SIZES,
                               "hidden_dims": HIDDEN_DIMS,
                               "epochs": EPOCHS,
                               "lr": LR,
                               "seed": SEED,
                           }}, f, indent=2)

    # ── fit power laws ────────────────────────────────────────────────
    arr = np.array([(r["n_train"], r["n_params"], r["test_mae"]) for r in runs])
    N, P, Y = arr[:, 0], arr[:, 1], arr[:, 2]

    # log-log fit:  log Y = a + alpha * log N + beta * log P
    A = np.column_stack([np.ones_like(N), np.log(N), np.log(P)])
    coefs, *_ = np.linalg.lstsq(A, np.log(Y), rcond=None)
    a, alpha, beta = float(coefs[0]), float(coefs[1]), float(coefs[2])
    pred_log = A @ coefs
    r2 = 1.0 - np.var(np.log(Y) - pred_log) / np.var(np.log(Y))
    print("\n=== Scaling-law fit  log(MAE) = a + alpha*log(N) + beta*log(P) ===")
    print(f"  a={a:.3f}  alpha (data)={alpha:.3f}  beta (params)={beta:.3f}")
    print(f"  R² = {r2:.3f}")

    out = {
        "runs": runs,
        "scaling_fit": {
            "form": "log(MAE) = a + alpha*log(N_train) + beta*log(N_params)",
            "a": a, "alpha_data": alpha, "beta_params": beta, "r2": float(r2),
        },
        "config": {
            "data_sizes": DATA_SIZES,
            "hidden_dims": HIDDEN_DIMS,
            "epochs": EPOCHS,
            "lr": LR,
            "seed": SEED,
        },
        "wall_time_min": (time.time() - t_start) / 60,
    }
    with open(RESULTS / "scaling_law.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {RESULTS / 'scaling_law.json'}")

    # ── figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel A: data scaling at fixed hidden dim = 128
    ax = axes[0]
    for hidden in HIDDEN_DIMS:
        sel = [r for r in runs if r["hidden_dim"] == hidden]
        sel.sort(key=lambda r: r["n_train"])
        if not sel:
            continue
        ns = [r["n_train"] for r in sel]
        ms = [r["test_mae"] for r in sel]
        ax.loglog(ns, ms, "o-", label=f"H={hidden}",
                  linewidth=1.5, markersize=6)
    ax.set_xlabel("Training samples N", fontsize=11)
    ax.set_ylabel("Test MAE (eV)", fontsize=11)
    ax.set_title("Data scaling", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Panel B: model scaling at fixed N = 8000 (largest)
    ax = axes[1]
    for n_train in DATA_SIZES:
        sel = [r for r in runs if r["n_train"] == n_train]
        sel.sort(key=lambda r: r["n_params"])
        if not sel:
            continue
        ps = [r["n_params"] / 1e6 for r in sel]
        ms = [r["test_mae"] for r in sel]
        ax.loglog(ps, ms, "s-", label=f"N={n_train}",
                  linewidth=1.5, markersize=6)
    ax.set_xlabel("Model parameters (M)", fontsize=11)
    ax.set_ylabel("Test MAE (eV)", fontsize=11)
    ax.set_title("Model scaling", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"IMP2D scaling laws: MAE ~ N^{{{alpha:.2f}}} P^{{{beta:.2f}}}  "
        f"(R²={r2:.2f})", fontsize=12,
    )
    fig.tight_layout()
    out_fig = FIG_DIR / "fig_scaling_law.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"figure saved -> {out_fig}")
    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
