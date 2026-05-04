"""Multi-source joint training with v2 (PFA-only) backbone.

Identical to ``scripts/multi_source_train.py`` except the shared backbone is
``PeriodicCrystalTransformer`` configured with ``use_pfa=True``,
``use_long_range=False``, ``use_defect_bias=False`` — i.e. only the
direction-aware Periodic Fourier bias from Phase 1 is kept (the multi-scale
distance and defect-pair biases were ablated as net-zero in single-source).

Each of {IMP2D, JARVIS-2D, JARVIS-3D, DFT-3D} gets its own readout head; the
IMP2D test MAE is the headline metric we compare against single-source 0.516
(v1 baseline) and 0.555 (v1 multi-source no-aug).
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
from src.models.attention_v2 import (  # noqa: E402
    PeriodicCrystalTransformer,
    compute_frac_disp,
)

DATA_PATHS = {
    "IMP2D":      ROOT / "data" / "processed" / "cleaned_dataset.pkl",
    "JARVIS-2D":  ROOT / "data" / "processed" / "jarvis_2d.pkl",
    "JARVIS-3D":  ROOT / "data" / "processed" / "jarvis_3d.pkl",
    "DFT-3D":     ROOT / "data" / "processed" / "dft_3d_lite.pkl",
}
SOURCES = list(DATA_PATHS.keys())

RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 60
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
SEED = 42
SOURCE_WEIGHTS = {
    "IMP2D":      1.0,
    "JARVIS-2D":  0.5,
    "JARVIS-3D":  0.5,
    "DFT-3D":     0.3,
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Multi-head v2 model ───────────────────────────────────────────────
class MultiHeadPeriodicTransformer(nn.Module):
    """Shared PeriodicCrystalTransformer backbone + per-source readout heads."""

    def __init__(self, n_sources: int, **kwargs):
        super().__init__()
        # PFA-only configuration based on Phase 1 ablation winner
        kwargs.setdefault("use_pfa", True)
        kwargs.setdefault("use_long_range", False)
        kwargs.setdefault("use_defect_bias", False)
        self.backbone = PeriodicCrystalTransformer(**kwargs)
        hidden_dim = kwargs.get("hidden_dim", 128)
        dropout = kwargs.get("dropout", 0.0)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(n_sources)
            ]
        )

    def _backbone_pool(self, batch: Dict) -> torch.Tensor:
        """Replicate PeriodicCrystalTransformer.forward up to the pooled vector."""
        bb = self.backbone
        x = batch["x"]
        mask = batch["atom_mask"]
        dist_matrix = batch["dist_matrix"]
        defect_mask = batch.get("defect_mask")
        positions = batch.get("positions")
        cell = batch.get("cell")
        device = x.device

        h = bb.embed(x)
        if bb.defect_embedding is not None and defect_mask is not None:
            h = h + bb.defect_embedding(defect_mask)

        b, n_max, c = h.shape
        num_atoms_list = batch["num_atoms_list"]
        flat_indices = []
        for i, n_i in enumerate(num_atoms_list):
            base = i * n_max
            flat_indices.append(
                torch.arange(n_i, device=device, dtype=torch.long) + base
            )
        flat_indices_t = (
            torch.cat(flat_indices)
            if flat_indices
            else torch.empty(0, dtype=torch.long, device=device)
        )
        h_flat_full = h.reshape(b * n_max, c)
        flat_h = h_flat_full.index_select(0, flat_indices_t)

        edge_index, edge_dist, triplet_index, angles = bb._flatten_edges(
            num_atoms_list,
            batch["edge_index_list"],
            batch["edge_dist_list"],
            batch["triplet_index_list"],
            batch["angles_list"],
            device=device,
        )
        edge_attr_rbf = bb.edge_rbf(edge_dist)
        for layer in bb.local_layers:
            flat_h = layer(flat_h, edge_index, edge_attr_rbf, triplet_index, angles)
        h_local_flat = torch.zeros(b * n_max, c, dtype=h.dtype, device=device)
        h_local_flat.index_copy_(0, flat_indices_t, flat_h)
        h_local = h_local_flat.reshape(b, n_max, c)

        # Geometry tensors
        frac_disp = None
        if bb.use_pfa and positions is not None and cell is not None:
            try:
                frac_disp = compute_frac_disp(positions, cell)
            except RuntimeError:
                frac_disp = None

        h_global = h_local
        for layer in bb.global_layers:
            h_global = layer(h_global, dist_matrix, frac_disp, defect_mask, mask)

        mask_f = mask.float().unsqueeze(-1)
        pooled = (h_global * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward(self, batch: Dict) -> torch.Tensor:
        pooled = self._backbone_pool(batch)
        sources = batch["source_id"]
        out = torch.zeros(pooled.size(0), device=pooled.device)
        for src_id in range(len(self.heads)):
            sel = sources == src_id
            if not sel.any():
                continue
            sub = pooled[sel]
            pred = self.heads[src_id](sub).squeeze(-1)
            out = out.masked_scatter(sel, pred)
        return out


# ── Multi-source dataset (mirror v1 multi_source_train.py) ────────────
class MultiSourceDataset(Dataset):
    def __init__(
        self,
        samples_per_source: Dict[str, list],
        mean_per_source: Dict[str, float],
        std_per_source: Dict[str, float],
    ):
        self.atom_features = get_atom_feature_table(None)
        self.entries = []
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
        numbers = torch.from_numpy(s["numbers"])
        x = self.atom_features[numbers]
        defect_mask = torch.from_numpy(
            s.get("defect_mask", np.zeros(len(numbers), dtype=np.int64))
        ).long()
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
        item["source_id"] = src_idx
        item["target_norm"] = float((s["target"] - mu) / sigma)
        item["sample_weight"] = w
        item["src_mean"] = mu
        item["src_std"] = sigma
        return item


def collate_fn_multi(batch):
    plain = collate_fn(
        [
            {
                k: v
                for k, v in b.items()
                if k
                not in ("source_id", "target_norm", "sample_weight", "src_mean", "src_std")
            }
            for b in batch
        ]
    )
    plain["source_id"] = torch.tensor(
        [b["source_id"] for b in batch], dtype=torch.long
    )
    plain["target_norm"] = torch.tensor(
        [b["target_norm"] for b in batch], dtype=torch.float32
    )
    plain["sample_weight"] = torch.tensor(
        [b["sample_weight"] for b in batch], dtype=torch.float32
    )
    plain["src_mean"] = torch.tensor(
        [b["src_mean"] for b in batch], dtype=torch.float32
    )
    plain["src_std"] = torch.tensor(
        [b["src_std"] for b in batch], dtype=torch.float32
    )
    return plain


def evaluate_imp2d(model, loader, device):
    model.eval()
    err, n = 0.0, 0
    sq, mae_list = 0.0, []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            pred_norm = model(batch)
            pred = pred_norm * batch["src_std"] + batch["src_mean"]
            target = batch["target"]
            imp2d_mask = batch["source_id"] == 0
            if not imp2d_mask.any():
                continue
            d = (pred[imp2d_mask] - target[imp2d_mask]).abs()
            err += d.sum().item()
            sq += (d ** 2).sum().item()
            n += imp2d_mask.sum().item()
    mae = err / max(n, 1)
    rmse = (sq / max(n, 1)) ** 0.5
    return mae, rmse, n


def train_one(model, train_loader, val_loader, device, epochs):
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
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
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

        val_mae, val_rmse, _ = evaluate_imp2d(model, val_loader, device)
        history.append(
            {
                "epoch": ep,
                "val_mae_imp2d": val_mae,
                "val_rmse_imp2d": val_rmse,
                "train_loss": loss_sum / max(n_seen, 1),
                "wall_sec": time.time() - ep_start,
            }
        )
        improved = val_mae < best_val - 1e-4
        if improved:
            best_val = val_mae
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
        print(
            f"  ep {ep:2d}  loss={loss_sum / max(n_seen, 1):.4f}  "
            f"val_mae_imp2d={val_mae:.4f}  ({time.time() - ep_start:.0f}s)"
            f"{'  *' if improved else ''}",
            flush=True,
        )
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

    sources_data = {}
    for name, path in DATA_PATHS.items():
        if not path.exists():
            print(f"  WARN: {name} not found at {path}, skipping")
            continue
        with open(path, "rb") as f:
            sources_data[name] = pickle.load(f)
        print(f"  loaded {name}: {len(sources_data[name])} samples")

    imp2d = sources_data["IMP2D"]
    train_idx, val_idx, test_idx = split_indices(len(imp2d), 0.8, 0.1, SEED)
    imp2d_train = [imp2d[i] for i in train_idx]
    imp2d_val = [imp2d[i] for i in val_idx]
    imp2d_test = [imp2d[i] for i in test_idx]
    print(
        f"\nIMP2D split: train {len(imp2d_train)} val {len(imp2d_val)} test {len(imp2d_test)}"
    )

    other_sources = {
        name: data for name, data in sources_data.items() if name != "IMP2D"
    }

    samples_per_source = {"IMP2D": imp2d_train, **other_sources}
    val_samples = {"IMP2D": imp2d_val}
    test_samples = {"IMP2D": imp2d_test}

    mean_per_source = {}
    std_per_source = {}
    for name, samples in samples_per_source.items():
        targets = np.array([s["target"] for s in samples], dtype=np.float64)
        mean_per_source[name] = float(targets.mean())
        std_per_source[name] = float(targets.std() + 1e-6)
        print(
            f"  {name}: n={len(samples)} mean={mean_per_source[name]:.4f} "
            f"std={std_per_source[name]:.4f}"
        )

    train_set = MultiSourceDataset(samples_per_source, mean_per_source, std_per_source)
    val_set = MultiSourceDataset(val_samples, mean_per_source, std_per_source)
    test_set = MultiSourceDataset(test_samples, mean_per_source, std_per_source)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_multi
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_multi
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_multi
    )

    print(
        f"\nDataset sizes: train {len(train_set)}  val {len(val_set)}  test {len(test_set)}"
    )

    model = MultiHeadPeriodicTransformer(
        n_sources=len(SOURCES),
        atom_fea_len=9,
        hidden_dim=128,
        n_local_layers=3,
        n_global_layers=2,
        num_heads=4,
        rcut_local=5.0,
        dmax_global=12.0,
        defect_embedding=True,
        dropout=0.1,
        # PFA-only (Phase 1 winner among components)
        use_pfa=True,
        use_long_range=False,
        use_defect_bias=False,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {n_params / 1e6:.4f}M ({n_params:,})")

    print(f"\nTraining v2 (PFA-only) multi-source for up to {EPOCHS} epochs...")
    model, history, best_val = train_one(
        model, train_loader, val_loader, device, epochs=EPOCHS
    )

    test_mae, test_rmse, n_test = evaluate_imp2d(model, test_loader, device)
    print(f"\nFinal IMP2D test MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  (N={n_test})")

    out = {
        "config": {
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "source_weights": SOURCE_WEIGHTS,
            "backbone": "PeriodicCrystalTransformer (PFA-only)",
        },
        "sources": {
            name: {
                "n_train": len(samples_per_source.get(name, [])),
                "n_val": len(val_samples.get(name, [])),
                "mean": mean_per_source.get(name),
                "std": std_per_source.get(name),
            }
            for name in SOURCES
        },
        "n_params": n_params,
        "best_val_mae_imp2d_eV": best_val,
        "test_mae_imp2d_eV": test_mae,
        "test_rmse_imp2d_eV": test_rmse,
        "v1_baseline_test_mae_eV": 0.516,  # CrystalTransformer single-seed leak-free aug
        "v1_multi_source_test_mae_eV": 0.555,
        "delta_pct_vs_v1_baseline": (test_mae - 0.516) / 0.516 * 100,
        "delta_pct_vs_v1_multisource": (test_mae - 0.555) / 0.555 * 100,
        "history": history,
        "wall_min": (time.time() - t_start) / 60.0,
    }
    out_path = RESULTS / "multi_source_train_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    # also save best checkpoint
    ckpt_dir = RESULTS / "multi_source_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": out["config"],
            "sources": SOURCES,
            "src_means": mean_per_source,
            "src_stds": std_per_source,
        },
        ckpt_dir / "best.pt",
    )
    print(f"\nWritten {out_path}")
    print(f"Wall time: {out['wall_min']:.1f} min")


if __name__ == "__main__":
    main()
