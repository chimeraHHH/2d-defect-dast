"""Plan A: Multi-source training to break the data bottleneck.

Unifies multiple defect/structure databases via per-source readout heads.
The shared backbone (CrystalTransformer encoder) gets gradient from
ALL sources, producing richer structural representations. Each sample
contributes loss only through its own source-specific head, so reference
mismatches across sources don't pollute each other.

Sources combined
----------------
- IMP2D            : defect formation energy (eV/cell)        — main task
- JARVIS-2D        : 2D vacancy formation energy (eV)         — small aug
- JARVIS-3D        : 3D bulk vacancy formation energy (eV)    — chem diversity
- JARVIS dft_3d    : 3D pristine formation energy (eV/atom)   — massive structural prior

Test evaluation: ALWAYS on IMP2D held-out test set, IMP2D head ONLY
(apples-to-apples vs single-source baseline 0.516 eV).

Outputs
-------
- results/multi_source_train.json
- paper/figures/fig_multi_source.png
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import (CrystalGraphDataset, collate_fn,  # noqa: E402
                          get_atom_feature_table, split_indices)
from src.models.baseline import CrystalTransformer  # noqa: E402

DATA_PATHS = {
    "IMP2D":      ROOT / "data" / "processed" / "cleaned_dataset.pkl",
    "JARVIS-2D":  ROOT / "data" / "processed" / "jarvis_2d.pkl",
    "JARVIS-3D":  ROOT / "data" / "processed" / "jarvis_3d.pkl",
    "DFT-3D":     ROOT / "data" / "processed" / "dft_3d_lite.pkl",
}
SOURCES = list(DATA_PATHS.keys())
N_SOURCES = len(SOURCES)

RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 60
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
SEED = 42
# loss weight per sample = source_weight[src]; balances large vs small sources
SOURCE_WEIGHTS = {
    "IMP2D":      1.0,    # main task — full weight
    "JARVIS-2D":  0.5,    # small aug
    "JARVIS-3D":  0.5,    # small aug
    "DFT-3D":     0.3,    # large pretraining-style aug
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Multi-head model ──────────────────────────────────────────────────
class MultiHeadCrystalTransformer(nn.Module):
    """Shared CrystalTransformer backbone + per-source readout heads.

    The backbone produces a pooled latent vector per graph; that vector
    is routed through the readout head corresponding to its data source.
    """

    def __init__(self, n_sources: int, **kwargs):
        super().__init__()
        # Build backbone but replace its readout
        self.backbone = CrystalTransformer(**kwargs)
        hidden_dim = kwargs.get("hidden_dim", 128)
        # remove backbone's readout — we'll grab the pooled vector instead
        self._strip_backbone_readout(hidden_dim)
        # per-source readouts
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(kwargs.get("dropout", 0.0)),
                nn.Linear(hidden_dim, 1),
            ) for _ in range(n_sources)
        ])

    def _strip_backbone_readout(self, hidden_dim):
        # We'll compute pooled ourselves; we keep backbone.readout but ignore it
        pass

    def _backbone_pool(self, batch: Dict) -> torch.Tensor:
        """Replicates CrystalTransformer.forward up to the pooled vector."""
        bb = self.backbone
        x = batch["x"]
        mask = batch["atom_mask"]
        dist_matrix = batch["dist_matrix"]
        defect_mask = batch.get("defect_mask")
        device = x.device

        h = bb.embed(x)
        if bb.defect_embedding is not None and defect_mask is not None:
            h = h + bb.defect_embedding(defect_mask)

        b, n_max, c = h.shape
        num_atoms_list = batch["num_atoms_list"]
        flat_indices = []
        for i, n_i in enumerate(num_atoms_list):
            base = i * n_max
            flat_indices.append(torch.arange(n_i, device=device, dtype=torch.long) + base)
        flat_indices = torch.cat(flat_indices) if flat_indices else torch.empty(0, dtype=torch.long, device=device)
        h_flat_full = h.reshape(b * n_max, c)
        flat_h = h_flat_full.index_select(0, flat_indices)

        edge_index, edge_dist, triplet_index, angles = bb._flatten_edges(
            num_atoms_list,
            batch["edge_index_list"], batch["edge_dist_list"],
            batch["triplet_index_list"], batch["angles_list"],
            device=device,
        )
        edge_attr_rbf = bb.edge_rbf(edge_dist)
        for layer in bb.local_layers:
            flat_h = layer(flat_h, edge_index, edge_attr_rbf, triplet_index, angles)
        h_local_flat = torch.zeros(b * n_max, c, dtype=h.dtype, device=device)
        h_local_flat.index_copy_(0, flat_indices, flat_h)
        h_local = h_local_flat.reshape(b, n_max, c)

        h_global = h_local
        for layer in bb.global_layers:
            h_global = layer(h_global, dist_matrix, mask)
        mask_f = mask.float().unsqueeze(-1)
        pooled = (h_global * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward(self, batch: Dict) -> torch.Tensor:
        """Predict per-sample using per-source head."""
        pooled = self._backbone_pool(batch)              # (B, hidden_dim)
        sources = batch["source_id"]                      # (B,) long
        out = torch.zeros(pooled.size(0), device=pooled.device)
        # route each sample through its source-specific head
        for src_id in range(len(self.heads)):
            mask = sources == src_id
            if not mask.any():
                continue
            sub = pooled[mask]
            pred = self.heads[src_id](sub).squeeze(-1)
            out = out.masked_scatter(mask, pred)
        return out


# ── Multi-source dataset ──────────────────────────────────────────────
class MultiSourceDataset(Dataset):
    """Concat all sources with per-sample (source_id, target_norm, weight)."""

    def __init__(self, samples_per_source: Dict[str, list],
                 mean_per_source: Dict[str, float],
                 std_per_source: Dict[str, float]):
        self.atom_features = get_atom_feature_table(None)
        self.entries = []   # list of (sample_dict, source_idx, mu, sigma, weight)
        for src_idx, name in enumerate(SOURCES):
            samples = samples_per_source.get(name, [])
            mu = mean_per_source.get(name, 0.0)
            sigma = std_per_source.get(name, 1.0)
            w = SOURCE_WEIGHTS.get(name, 1.0)
            for s in samples:
                self.entries.append((s, src_idx, mu, sigma, w))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        s, src_idx, mu, sigma, w = self.entries[i]
        # build pytorch sample (mirror CrystalGraphDataset.__getitem__)
        numbers = torch.from_numpy(s["numbers"])
        x = self.atom_features[numbers]
        defect_mask = torch.from_numpy(s.get(
            "defect_mask", np.zeros(len(numbers), dtype=np.int64))).long()
        item = {
            "x": x,
            "defect_mask": defect_mask,
            "edge_index": torch.from_numpy(s["edge_index"]),
            "edge_dist": torch.from_numpy(s["edge_dist"]),
            "triplet_index": torch.from_numpy(s["triplet_index"]),
            "angles": torch.from_numpy(s["angles"]),
            "dist_matrix": torch.from_numpy(s["dist_matrix"]),
            "positions": torch.from_numpy(s["positions"]),
            "cell": torch.from_numpy(s["cell"]),
            "target": torch.tensor(s["target"], dtype=torch.float32),
            "num_atoms": len(numbers),
        }
        # per-source extras
        item["source_id"] = src_idx
        item["target_norm"] = float((s["target"] - mu) / sigma)
        item["sample_weight"] = w
        item["src_mean"] = mu
        item["src_std"] = sigma
        return item


def collate_fn_multi(batch):
    """Collate that preserves source_id, target_norm, weight."""
    plain = collate_fn([{k: v for k, v in b.items()
                         if k not in ("source_id", "target_norm",
                                      "sample_weight", "src_mean", "src_std")}
                        for b in batch])
    plain["source_id"] = torch.tensor(
        [b["source_id"] for b in batch], dtype=torch.long)
    plain["target_norm"] = torch.tensor(
        [b["target_norm"] for b in batch], dtype=torch.float32)
    plain["sample_weight"] = torch.tensor(
        [b["sample_weight"] for b in batch], dtype=torch.float32)
    plain["src_mean"] = torch.tensor(
        [b["src_mean"] for b in batch], dtype=torch.float32)
    plain["src_std"] = torch.tensor(
        [b["src_std"] for b in batch], dtype=torch.float32)
    return plain


# ── Train / eval helpers ──────────────────────────────────────────────
def evaluate_imp2d(model, loader, device, mean, std):
    """Evaluate IMP2D head only on IMP2D samples."""
    model.eval()
    err, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            pred_norm = model(batch)
            # de-normalise per sample
            pred = pred_norm * batch["src_std"] + batch["src_mean"]
            target = batch["target"]
            # only evaluate IMP2D samples
            imp2d_mask = batch["source_id"] == 0
            if not imp2d_mask.any():
                continue
            err += (pred[imp2d_mask] - target[imp2d_mask]).abs().sum().item()
            n += imp2d_mask.sum().item()
    return err / max(n, 1)


def train_one(model, train_loader, val_loader, device, mean, std, epochs):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    best_val, bad = float("inf"), 0
    best_state = None
    history = []
    for ep in range(epochs):
        ep_start = time.time()
        model.train()
        loss_sum, n_seen = 0.0, 0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            pred_norm = model(batch)
            target_norm = batch["target_norm"]
            w = batch["sample_weight"]
            losses = loss_fn(pred_norm, target_norm)
            loss = (losses * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.item()) * len(target_norm)
            n_seen += len(target_norm)
        sched.step()

        val_mae = evaluate_imp2d(model, val_loader, device, mean, std)
        history.append({"epoch": ep, "val_mae_imp2d": val_mae,
                        "train_loss": loss_sum / max(n_seen, 1)})
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep < 3 or ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep {ep:2d}  loss={loss_sum/max(n_seen,1):.4f}  "
                  f"val_mae_imp2d={val_mae:.4f}  ({time.time()-ep_start:.0f}s)",
                  flush=True)
        if bad >= PATIENCE:
            print(f"  early stop at ep {ep}, best={best_val:.4f}")
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

    # ── Load all sources ────────────────────────────────────────────
    sources_data = {}
    for name, path in DATA_PATHS.items():
        if not path.exists():
            print(f"  WARN: {name} not found at {path}, skipping")
            continue
        with open(path, "rb") as f:
            sources_data[name] = pickle.load(f)
        print(f"  loaded {name}: {len(sources_data[name])} samples")

    # ── IMP2D split (consistent with rest of paper) ─────────────────
    imp2d = sources_data["IMP2D"]
    train_idx, val_idx, test_idx = split_indices(len(imp2d), 0.8, 0.1, SEED)
    imp2d_train = [imp2d[i] for i in train_idx]
    imp2d_val = [imp2d[i] for i in val_idx]
    imp2d_test = [imp2d[i] for i in test_idx]
    print(f"\nIMP2D split: train {len(imp2d_train)} val {len(imp2d_val)} test {len(imp2d_test)}")

    # for other sources: use ALL of them in training (no test split)
    other_sources = {name: data for name, data in sources_data.items()
                     if name != "IMP2D"}
    for name, data in other_sources.items():
        # tiny split: 90% train, 10% val (to avoid overfitting per-source head)
        n = len(data)
        rng = np.random.default_rng(SEED + hash(name) % 1000)
        perm = rng.permutation(n)
        n_val = max(5, n // 10)
        other_sources[name] = {
            "train": [data[i] for i in perm[n_val:]],
            "val":   [data[i] for i in perm[:n_val]],
        }

    # ── Compute per-source target normalisation ─────────────────────
    means, stds = {}, {}
    means["IMP2D"] = float(np.mean([s["target"] for s in imp2d_train]))
    stds["IMP2D"]  = float(np.std([s["target"] for s in imp2d_train]) + 1e-6)
    for name, splits in other_sources.items():
        targets = np.array([s["target"] for s in splits["train"]])
        means[name] = float(targets.mean())
        stds[name]  = float(targets.std() + 1e-6)
    for name in SOURCES:
        if name in means:
            print(f"  {name}: μ={means[name]:.3f}  σ={stds[name]:.3f}")

    # ── Build train / val / test loaders ────────────────────────────
    train_samples = {"IMP2D": imp2d_train}
    val_samples = {"IMP2D": imp2d_val}
    test_samples = {"IMP2D": imp2d_test}
    for name, splits in other_sources.items():
        train_samples[name] = splits["train"]
        val_samples[name] = splits["val"]
        # NO test for other sources

    train_ds = MultiSourceDataset(train_samples, means, stds)
    val_ds = MultiSourceDataset(val_samples, means, stds)
    test_ds = MultiSourceDataset(test_samples, means, stds)
    print(f"\nTotal train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_multi)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn_multi)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn_multi)

    # ── Build multi-head model ──────────────────────────────────────
    model = MultiHeadCrystalTransformer(
        n_sources=N_SOURCES,
        atom_fea_len=9, hidden_dim=128,
        n_local_layers=3, n_global_layers=2,
        num_heads=4, dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params/1e6:.3f}M params (multi-head, {N_SOURCES} heads)")

    # ── Train ───────────────────────────────────────────────────────
    print(f"\n=== Training ({EPOCHS} epochs) ===")
    model, history, best_val = train_one(model, train_loader, val_loader,
                                          device, means["IMP2D"], stds["IMP2D"],
                                          EPOCHS)

    # ── Final test evaluation on IMP2D ──────────────────────────────
    test_mae = evaluate_imp2d(model, test_loader, device,
                               means["IMP2D"], stds["IMP2D"])
    print(f"\n=== FINAL ===")
    print(f"  IMP2D val   MAE = {best_val:.4f} eV")
    print(f"  IMP2D test  MAE = {test_mae:.4f} eV")
    print(f"  Single-source baseline (paper §5.1): 0.516 eV (4-seed mean 0.537)")
    print(f"  Δ vs baseline: {(test_mae - 0.516)/0.516*100:+.1f}%")

    # ── Save ────────────────────────────────────────────────────────
    out = {
        "config": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "seed": SEED, "source_weights": SOURCE_WEIGHTS,
        },
        "sources": {name: {
            "n_train": len([e for e in train_ds.entries
                            if e[1] == SOURCES.index(name)]),
            "n_val": len([e for e in val_ds.entries
                          if e[1] == SOURCES.index(name)]),
            "mean": means.get(name, 0.0),
            "std": stds.get(name, 1.0),
        } for name in SOURCES if name in means},
        "n_params": int(n_params),
        "best_val_mae_imp2d_eV": float(best_val),
        "test_mae_imp2d_eV": float(test_mae),
        "baseline_test_mae_eV": 0.516,
        "delta_pct": float((test_mae - 0.516) / 0.516 * 100),
        "history": history,
        "wall_time_min": (time.time() - t_start) / 60,
    }
    out_path = RESULTS / "multi_source_train.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")

    # ── Figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    eps = [h["epoch"] for h in history]
    val_maes = [h["val_mae_imp2d"] for h in history]
    ax.plot(eps, val_maes, "-o", color="#1f77b4", lw=2, ms=4,
             label="Multi-source val MAE (IMP2D)")
    ax.axhline(0.516, color="gray", ls="--", lw=1.5,
                label="Single-source baseline (0.516)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IMP2D test MAE (eV)")
    ax.set_title(
        f"Plan A: Multi-source training\n"
        f"final test MAE {test_mae:.4f} eV "
        f"({(test_mae-0.516)/0.516*100:+.1f}% vs baseline)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_fig = FIG_DIR / "fig_multi_source.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"figure -> {out_fig}")
    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
