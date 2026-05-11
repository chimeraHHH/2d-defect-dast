"""Multi-source v4: Apply enhanced training recipe to 4-DB joint training.

Key differences vs v2_aug:
  - Backbone: CrystalTransformer (baseline) with ct-UAE embeddings (not PFA)
  - Loss: MAE (L1) instead of SmoothL1
  - Scheduler: Cosine annealing with 10-epoch warmup
  - Training: 150 epochs with SWA from ep120
  - Regularization: label_noise 0.03, drop_path 0.1
  - Unchanged: per-source readout heads, source weighting

Expected improvement: v2_aug achieved 0.486±0.025; single-source v4 achieved
0.407. Multi-source should benefit from the enhanced recipe while also
leveraging cross-domain regularization.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import (CrystalGraphDataset, collate_fn,  # noqa: E402
                          get_atom_feature_table, split_indices)
from src.models import CrystalTransformer  # noqa: E402

# ── Config ───────────────────────────────────────────────────────────────
DATA_PATHS = {
    "IMP2D":      ROOT / "data" / "processed" / "cleaned_dataset.pkl",
    "JARVIS-2D":  ROOT / "data" / "processed" / "jarvis_2d.pkl",
    "JARVIS-3D":  ROOT / "data" / "processed" / "jarvis_3d.pkl",
    "DFT-3D":     ROOT / "data" / "processed" / "dft_3d_lite.pkl",
}
SOURCES = list(DATA_PATHS.keys())
RESULTS = ROOT / "results"

EPOCHS = 150
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1.0e-4
WARMUP_EPOCHS = 10
SWA_START = 120
SWA_LR = 1e-4
LABEL_NOISE_STD = 0.03
GRAD_CLIP = 5.0
PATIENCE = 30  # longer patience for 150ep

SOURCE_WEIGHTS = {"IMP2D": 1.0, "JARVIS-2D": 0.5, "JARVIS-3D": 0.5, "DFT-3D": 0.1}

MODEL_KWARGS = dict(
    atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2,
    num_heads=4, rcut_local=5.0, dmax_global=12.0, defect_embedding=True,
    dropout=0.1, ct_uae_path=str(ROOT / "data" / "ct_uae_mt3_embeddings.pt"),
)

DEEP_MODEL_KWARGS = dict(
    atom_fea_len=9, hidden_dim=128, n_local_layers=4, n_global_layers=3,
    num_heads=4, rcut_local=5.0, dmax_global=12.0, defect_embedding=True,
    dropout=0.12, ct_uae_path=str(ROOT / "data" / "ct_uae_mt3_embeddings.pt"),
)


# ── Multi-head model wrapping CrystalTransformer ─────────────────────────
class MultiHeadCrystalTransformer(nn.Module):
    """Shared CrystalTransformer backbone + per-source readout heads.

    This replaces the single readout MLP in the base model with N source-specific
    readout heads while sharing the entire local+global backbone.
    """

    def __init__(self, n_sources: int, **backbone_kwargs):
        super().__init__()
        self.backbone = CrystalTransformer(**backbone_kwargs)
        hidden_dim = backbone_kwargs.get("hidden_dim", 128)
        dropout = backbone_kwargs.get("dropout", 0.0)

        # Remove the default readout from backbone - we'll use per-source heads
        # Keep backbone.readout for single-source inference compatibility
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_sources)
        ])

    def _get_pooled(self, batch: Dict) -> torch.Tensor:
        """Run backbone up to the pooled representation (before readout)."""
        bb = self.backbone
        x = batch["x"]
        mask = batch["atom_mask"]
        dist_matrix = batch["dist_matrix"]
        defect_mask = batch.get("defect_mask")
        device = x.device

        # ct-UAE embedding
        if bb.ct_uae_table is not None:
            z = batch.get("atomic_numbers")
            if z is not None:
                z_clamped = z.clamp(0, bb.ct_uae_table.shape[0] - 1)
                uae_fea = bb.ct_uae_table[z_clamped]
                x = torch.cat([x, uae_fea], dim=-1)

        h = bb.embed(x)
        if bb.defect_embedding is not None and defect_mask is not None:
            h = h + bb.defect_embedding(defect_mask)

        # Local interaction (sparse)
        b, n_max, c = h.shape
        num_atoms_list = batch["num_atoms_list"]
        flat_indices = []
        for i, n_i in enumerate(num_atoms_list):
            base_offset = i * n_max
            flat_indices.append(
                torch.arange(n_i, device=device, dtype=torch.long) + base_offset
            )
        flat_indices_t = torch.cat(flat_indices) if flat_indices else torch.empty(0, dtype=torch.long, device=device)
        h_flat = h.reshape(b * n_max, c).index_select(0, flat_indices_t)

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
            h_flat = layer(h_flat, edge_index, edge_attr_rbf, triplet_index, angles)

        h_local = torch.zeros(b * n_max, c, dtype=h.dtype, device=device)
        h_local.index_copy_(0, flat_indices_t, h_flat)
        h_local = h_local.reshape(b, n_max, c)

        # Global transformer
        h_global = h_local
        for layer in bb.global_layers:
            h_global = layer(h_global, dist_matrix, mask)

        # Pool
        mask_f = mask.float().unsqueeze(-1)
        pooled = (h_global * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward(self, batch: Dict) -> torch.Tensor:
        pooled = self._get_pooled(batch)
        sources = batch["source_id"]
        out = torch.zeros(pooled.size(0), device=pooled.device)
        for src_id in range(len(self.heads)):
            sel = sources == src_id
            if not sel.any():
                continue
            pred = self.heads[src_id](pooled[sel]).squeeze(-1)
            out = out.masked_scatter(sel, pred)
        return out


# ── Dataset ──────────────────────────────────────────────────────────────
class MultiSourceDataset(Dataset):
    def __init__(self, samples_per_source, mean_per_source, std_per_source):
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
            "atomic_numbers": numbers.long(),
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
    """Collate that preserves source metadata."""
    plain = collate_fn([
        {k: v for k, v in b.items()
         if k not in ("source_id", "target_norm", "sample_weight", "src_mean", "src_std")}
        for b in batch
    ])
    plain["source_id"] = torch.tensor([b["source_id"] for b in batch], dtype=torch.long)
    plain["target_norm"] = torch.tensor([b["target_norm"] for b in batch], dtype=torch.float32)
    plain["sample_weight"] = torch.tensor([b["sample_weight"] for b in batch], dtype=torch.float32)
    plain["src_mean"] = torch.tensor([b["src_mean"] for b in batch], dtype=torch.float32)
    plain["src_std"] = torch.tensor([b["src_std"] for b in batch], dtype=torch.float32)
    return plain


# ── Training ─────────────────────────────────────────────────────────────
def evaluate_imp2d(model, loader, device):
    model.eval()
    err, sq, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred_norm = model(batch)
            pred = pred_norm * batch["src_std"].to(device) + batch["src_mean"].to(device)
            target = batch["target"].to(device)
            imp2d_mask = batch["source_id"].to(device) == 0
            if not imp2d_mask.any():
                continue
            d = (pred[imp2d_mask] - target[imp2d_mask]).abs()
            err += d.sum().item()
            sq += (d ** 2).sum().item()
            n += imp2d_mask.sum().item()
    mae = err / max(n, 1)
    rmse = (sq / max(n, 1)) ** 0.5
    return mae, rmse, n


def train(model, train_loader, val_loader, device, epochs, seed):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup + cosine annealing
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - WARMUP_EPOCHS, eta_min=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler],
        milestones=[WARMUP_EPOCHS])

    # SWA
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=SWA_LR)

    best_val, bad = float("inf"), 0
    best_state = None
    history = []

    for ep in range(epochs):
        ep_start = time.time()
        model.train()
        loss_sum, n_seen = 0.0, 0

        use_swa = ep >= SWA_START

        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            target_norm = batch["target_norm"]
            w = batch["sample_weight"]

            # Label noise
            if LABEL_NOISE_STD > 0:
                noise = torch.randn_like(target_norm) * LABEL_NOISE_STD
                target_norm = target_norm + noise

            pred_norm = model(batch)

            # MAE loss with source weighting
            losses = (pred_norm - target_norm).abs()
            loss = (losses * w).sum() / w.sum().clamp(min=1e-6)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            loss_sum += loss.item() * len(target_norm)
            n_seen += len(target_norm)

        # Scheduler step
        if use_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validation
        val_mae, val_rmse, _ = evaluate_imp2d(model, val_loader, device)
        history.append({
            "epoch": ep, "val_mae": val_mae, "val_rmse": val_rmse,
            "train_loss": loss_sum / max(n_seen, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "wall_sec": time.time() - ep_start,
        })

        improved = val_mae < best_val - 1e-4
        if improved:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % 10 == 0 or improved:
            print(f"  ep {ep:3d}  loss={loss_sum/max(n_seen,1):.4f}  "
                  f"val_mae={val_mae:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
                  f"{'  *' if improved else ''}", flush=True)

        if bad >= PATIENCE and ep >= SWA_START:
            print(f"  early stop at ep {ep}, best={best_val:.4f}")
            break

    # Final: use SWA model if available
    if SWA_START < epochs:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_state = {k.replace("module.", ""): v for k, v in swa_model.state_dict().items()
                     if not k.startswith("n_averaged")}
        # Evaluate SWA
        model.load_state_dict(swa_state, strict=False)
        swa_mae, _, _ = evaluate_imp2d(model, val_loader, device)
        print(f"  SWA val_mae={swa_mae:.4f} vs best={best_val:.4f}")
        if swa_mae < best_val:
            best_val = swa_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deep", action="store_true", help="Use deep 4+3 model")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    depth_tag = "_deep" if args.deep else ""
    tag = args.tag or f"multi_source_v4{depth_tag}_s{args.seed}"

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed={args.seed}  tag={tag}  deep={args.deep}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    sources_data = {}
    for name, path in DATA_PATHS.items():
        if not path.exists():
            print(f"  WARN: {name} not found at {path}, skipping")
            continue
        with open(path, "rb") as f:
            sources_data[name] = pickle.load(f)
        print(f"  loaded {name}: {len(sources_data[name])} samples")

    # IMP2D split (same as all other experiments)
    imp2d = sources_data["IMP2D"]
    train_idx, val_idx, test_idx = split_indices(len(imp2d), 0.8, 0.1, 42)  # always seed=42 for split
    imp2d_train = [imp2d[i] for i in train_idx]
    imp2d_val = [imp2d[i] for i in val_idx]
    imp2d_test = [imp2d[i] for i in test_idx]
    print(f"\n  IMP2D split: train {len(imp2d_train)} val {len(imp2d_val)} test {len(imp2d_test)}")

    # Other sources: use all for training (no val/test needed)
    other_sources = {name: data for name, data in sources_data.items() if name != "IMP2D"}

    samples_per_source = {"IMP2D": imp2d_train, **other_sources}
    val_samples = {"IMP2D": imp2d_val}
    test_samples = {"IMP2D": imp2d_test}

    # Normalizer per source
    mean_per_source, std_per_source = {}, {}
    for name, samples in samples_per_source.items():
        targets = np.array([s["target"] for s in samples], dtype=np.float64)
        mean_per_source[name] = float(targets.mean())
        std_per_source[name] = float(targets.std() + 1e-6)
        print(f"    {name}: n={len(samples)} mean={mean_per_source[name]:.4f} std={std_per_source[name]:.4f}")

    train_set = MultiSourceDataset(samples_per_source, mean_per_source, std_per_source)
    val_set = MultiSourceDataset(val_samples, mean_per_source, std_per_source)
    test_set = MultiSourceDataset(test_samples, mean_per_source, std_per_source)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_multi, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn_multi, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn_multi, num_workers=2)

    print(f"\n  Dataset sizes: train {len(train_set)} val {len(val_set)} test {len(test_set)}")

    # Model
    model_kwargs = DEEP_MODEL_KWARGS if args.deep else MODEL_KWARGS
    model = MultiHeadCrystalTransformer(n_sources=len(SOURCES), **model_kwargs)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"\n  Model params: {n_params/1e6:.4f}M")

    # Train
    print(f"\n  Training {tag} for up to {EPOCHS} epochs...")
    model, history, best_val = train(model, train_loader, val_loader, device, EPOCHS, args.seed)

    # Test
    test_mae, test_rmse, n_test = evaluate_imp2d(model, test_loader, device)
    print(f"\n  Final IMP2D test MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  (N={n_test})")

    # Save
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "config": {
            "seed": args.seed, "epochs": EPOCHS, "lr": LR, "deep": args.deep,
            "batch_size": BATCH_SIZE, "warmup_epochs": WARMUP_EPOCHS,
            "swa_start": SWA_START, "label_noise": LABEL_NOISE_STD,
            "source_weights": SOURCE_WEIGHTS,
            "model_kwargs": model_kwargs,
            "backbone": "CrystalTransformer + ct-UAE + per-source heads",
        },
        "n_params": n_params,
        "best_val_mae_imp2d_eV": best_val,
        "test_mae_imp2d_eV": test_mae,
        "test_rmse_imp2d_eV": test_rmse,
        "references": {
            "v1_single_source": 0.516,
            "v2_multi_source_4seed": 0.486,
            "v4_single_source_best": 0.407,
        },
        "history": history,
        "wall_min": (time.time() - t_start) / 60.0,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    torch.save({
        "model": model.state_dict(),
        "config": out["config"],
        "normalizer": {"mean": mean_per_source["IMP2D"], "std": std_per_source["IMP2D"]},
        "src_means": mean_per_source,
        "src_stds": std_per_source,
    }, out_dir / "best.pt")

    # Save IMP2D test predictions
    preds_arr, targets_arr = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred_norm = model(batch)
            pred = pred_norm * batch["src_std"].to(device) + batch["src_mean"].to(device)
            mask = batch["source_id"].to(device) == 0
            if not mask.any():
                continue
            preds_arr.append(pred[mask].cpu().numpy())
            targets_arr.append(batch["target"][mask].cpu().numpy())

    np.savez(out_dir / "test_predictions.npz",
             preds=np.concatenate(preds_arr),
             targets=np.concatenate(targets_arr))

    print(f"\n  Saved -> {out_dir}")
    print(f"  Wall time: {out['wall_min']:.1f} min")


if __name__ == "__main__":
    main()
