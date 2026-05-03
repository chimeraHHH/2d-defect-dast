"""C14: Third dataset — JARVIS dft_2d (1.1k pristine 2D materials).

Tests cross-task transfer: model trained on IMP2D defect formation energy,
evaluated zero-shot and few-shot on dft_2d pristine formation-energy-per-atom
(a related but distinct task).

Pipeline
--------
1. Download dft_2d via jarvis-tools.
2. Convert each entry into our internal sample dict (atoms, edges, defect-mask
   = all zeros since pristine).
3. Zero-shot evaluation with the IMP2D-pretrained checkpoint.
4. Few-shot (k=10/30/100/300) fine-tuning vs random init.

Outputs
-------
- data/processed/dft_2d.pkl
- results/dft2d_transfer.json
- paper/figures/fig_dft2d_transfer.png
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

DATA_OUT = ROOT / "data" / "processed" / "dft_2d.pkl"
CKPT = ROOT / "results" / "baseline_h128_aug_long_safe" / "best.pt"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = 5.0          # neighbor cutoff for graph construction
MAX_NEIGH = 32        # cap neighbors
SEED = 42

K_FEW_SHOT = [10, 30, 100, 300]
EPOCHS_FT = 60
BATCH_SIZE = 16
LR_BACKBONE = 1e-5
LR_READOUT = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8


# ── data download + conversion ───────────────────────────────────────
def download_dft2d() -> list:
    if DATA_OUT.exists():
        with open(DATA_OUT, "rb") as f:
            return pickle.load(f)
    print("Downloading JARVIS dft_2d (1.1k pristine 2D materials)...")
    from jarvis.db.figshare import data
    raw = data("dft_2d")
    print(f"  got {len(raw)} entries; converting to internal format...")

    converted = []
    for i, entry in enumerate(raw):
        if entry.get("formation_energy_peratom") is None:
            continue
        try:
            atoms = entry["atoms"]
            elements = atoms["elements"]
            coords = np.asarray(atoms["coords"], dtype=np.float32)
            cell = np.asarray(atoms["lattice_mat"], dtype=np.float32)
            # convert frac->cart if needed (jarvis uses fractional)
            if atoms.get("cartesian", False) is False:
                coords = coords @ cell

            # element symbol -> Z
            from ase.data import atomic_numbers
            Z = np.array([atomic_numbers[e] for e in elements], dtype=np.int64)
            n = len(Z)
            if n < 2 or n > 60:
                continue

            # build neighbor edges with periodic boundary conditions
            from ase import Atoms
            ase_atoms = Atoms(numbers=Z, positions=coords, cell=cell, pbc=True)
            # all pairs within cutoff
            from ase.neighborlist import NeighborList, neighbor_list
            ii, jj, dd, oo = neighbor_list("ijdS", ase_atoms,
                                            cutoff=CUTOFF,
                                            self_interaction=False)
            if len(ii) == 0:
                continue
            edge_index = np.stack([ii, jj], axis=0).astype(np.int64)
            edge_dist = dd.astype(np.float32)
            edge_offset = oo.astype(np.float32)

            # triplets: (center, neighbor1, neighbor2) - atom indices, like IMP2D
            triplet_index = []
            angles = []
            from collections import defaultdict
            inc = defaultdict(list)  # center_atom -> list of (neighbor_idx, edge_id)
            for k, (u, v, d, o) in enumerate(zip(ii, jj, dd, oo)):
                inc[u].append((v, k))
            for u, neigh_list in inc.items():
                if len(neigh_list) < 2:
                    continue
                if len(neigh_list) > 6:
                    chosen = list(np.random.default_rng(0).choice(
                        len(neigh_list), 6, replace=False))
                    neigh_list = [neigh_list[i] for i in chosen]
                for a in range(len(neigh_list)):
                    for b in range(a + 1, len(neigh_list)):
                        v_a_atom, ka = neigh_list[a]
                        v_b_atom, kb = neigh_list[b]
                        triplet_index.append([u, v_a_atom, v_b_atom])
                        v_a = (coords[v_a_atom] + oo[ka] @ cell - coords[u])
                        v_b = (coords[v_b_atom] + oo[kb] @ cell - coords[u])
                        cos = float(np.dot(v_a, v_b)
                                    / (np.linalg.norm(v_a) * np.linalg.norm(v_b)
                                       + 1e-9))
                        cos = max(-1.0, min(1.0, cos))
                        angles.append(np.arccos(cos))
            if not triplet_index:
                triplet_index = np.zeros((0, 3), dtype=np.int64)
                angles = np.zeros((0,), dtype=np.float32)
            else:
                triplet_index = np.asarray(triplet_index, dtype=np.int64)
                angles = np.asarray(angles, dtype=np.float32)

            # full distance matrix (for global attention)
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(coords, coords).astype(np.float32)

            sample = {
                "id": i,
                "unique_id": entry.get("jid", str(i)),
                "numbers": Z,
                "positions": coords.astype(np.float32),
                "cell": cell,
                "edge_index": edge_index,
                "edge_dist": edge_dist,
                "edge_offset": edge_offset,
                "triplet_index": triplet_index,
                "angles": angles,
                "dist_matrix": dist_matrix,
                "target": float(entry["formation_energy_peratom"]),
                "metadata": {
                    "host": entry.get("formula", ""),
                    "dopant": "",
                    "site": "",
                    "defecttype": "pristine",
                    "natoms": int(n),
                    "spacegroup": str(entry.get("spg_number", "1")),
                    "supercell": "111",
                    "jid": entry.get("jid"),
                },
            }
            converted.append(sample)
        except Exception as e:
            if i < 5:
                print(f"  skip {i}: {type(e).__name__}: {e}")
            continue

    print(f"  converted {len(converted)}/{len(raw)} entries")
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_OUT, "wb") as f:
        pickle.dump(converted, f)
    print(f"  saved -> {DATA_OUT}")
    return converted


# ── train / eval helpers ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    err, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            pred = model(batch)
            tgt = batch["target"]
            err += (pred - tgt).abs().sum().item()
            n += tgt.numel()
    return err / max(n, 1)


def make_model():
    return CrystalTransformer(
        atom_fea_len=9, hidden_dim=128,
        n_local_layers=3, n_global_layers=2,
        num_heads=4, dropout=0.1,
    )


def load_pretrained():
    model = make_model()
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if "model_state" in state:
        state = state["model_state"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model


def train(model, train_loader, val_loader, device, epochs, init_lr_b, init_lr_r):
    model.to(device)
    # differential LR
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if "readout" in n:
            head_params.append(p)
        else:
            backbone_params.append(p)
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": init_lr_b},
        {"params": head_params, "lr": init_lr_r},
    ], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()
    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
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
    return model


def main():
    t_start = time.time()
    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    samples = download_dft2d()
    print(f"\nTotal converted samples: {len(samples)}")

    # build dataset object
    cls = CrystalGraphDataset.__new__(CrystalGraphDataset)
    cls.data = samples
    cls.meta = None
    from src.dataset import get_atom_feature_table
    cls.atom_features = get_atom_feature_table(None)
    cls.defect_mark_neighbors = 0
    # validate + fix: drop samples with malformed indices
    valid_data = []
    for s in cls.data:
        n = len(s["numbers"])
        if "defect_mask" not in s:
            s["defect_mask"] = np.zeros(n, dtype=np.int64)
        else:
            s["defect_mask"] = s["defect_mask"].astype(np.int64)
        # sanity: edge_index must be < n
        ei = s["edge_index"]
        if ei.size == 0 or ei.max() >= n:
            continue
        # triplet_index references atom indices (must be < n)
        ti = s["triplet_index"]
        if ti.size > 0 and ti.max() >= n:
            continue
        # check Z values are in [1, 100]
        if (s["numbers"] < 1).any() or (s["numbers"] > 100).any():
            continue
        valid_data.append(s)
    cls.data = valid_data
    print(f"After validation: {len(cls.data)}/{len(samples)} samples remain")
    samples = cls.data

    # split
    n = len(samples)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_test = max(50, n // 5)
    test_idx = idx[:n_test].tolist()
    pool_idx = idx[n_test:].tolist()
    print(f"Test: {len(test_idx)}, available pool for FT: {len(pool_idx)}")

    test_loader = DataLoader(Subset(cls, test_idx), batch_size=16,
                             shuffle=False, collate_fn=collate_fn)

    results = {"k_zero_shot": None, "few_shot": {}}

    # ── zero-shot ─────────────────────────────────────────────────
    print("\n=== Zero-shot evaluation ===")
    model = load_pretrained().to(device)
    zs_mae = evaluate(model, test_loader, device)
    print(f"  Zero-shot MAE = {zs_mae:.4f} eV/atom")
    results["k_zero_shot"] = float(zs_mae)

    # baseline target stats for context
    targets = np.array([s["target"] for s in samples])
    mean_pred_mae = float(np.mean(np.abs(targets[test_idx] - targets[pool_idx].mean())))
    print(f"  Mean-predictor MAE (reference) = {mean_pred_mae:.4f}")
    results["mean_predictor_mae"] = mean_pred_mae

    # ── few-shot ──────────────────────────────────────────────────
    val_n = max(50, len(pool_idx) // 5)
    val_idx = pool_idx[-val_n:]
    val_loader = DataLoader(Subset(cls, val_idx), batch_size=16,
                            shuffle=False, collate_fn=collate_fn)

    for k in K_FEW_SHOT:
        if k > len(pool_idx) - val_n:
            continue
        print(f"\n=== Few-shot k={k} ===")
        train_seed = pool_idx[:k]
        train_loader = DataLoader(Subset(cls, train_seed),
                                   batch_size=min(BATCH_SIZE, k),
                                   shuffle=True, collate_fn=collate_fn,
                                   drop_last=False)

        # FT (pretrained)
        model_ft = load_pretrained().to(device)
        model_ft = train(model_ft, train_loader, val_loader, device,
                          EPOCHS_FT, LR_BACKBONE, LR_READOUT)
        ft_mae = evaluate(model_ft, test_loader, device)

        # SC (from scratch)
        model_sc = make_model().to(device)
        model_sc = train(model_sc, train_loader, val_loader, device,
                          EPOCHS_FT, 5e-4, 5e-4)
        sc_mae = evaluate(model_sc, test_loader, device)

        improvement = (sc_mae - ft_mae) / sc_mae * 100
        print(f"  FT MAE = {ft_mae:.4f}  SC MAE = {sc_mae:.4f}  "
              f"improvement = {improvement:+.1f}%")
        results["few_shot"][f"k={k}"] = {
            "ft_mae": float(ft_mae),
            "scratch_mae": float(sc_mae),
            "improvement_pct": float(improvement),
        }

        # incremental save
        with open(RESULTS / "dft2d_transfer.json", "w") as f:
            json.dump(results, f, indent=2)

    results["wall_time_min"] = (time.time() - t_start) / 60
    with open(RESULTS / "dft2d_transfer.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {RESULTS / 'dft2d_transfer.json'}")

    # figure
    if results["few_shot"]:
        ks = sorted(int(k.split("=")[1]) for k in results["few_shot"])
        ft = [results["few_shot"][f"k={k}"]["ft_mae"] for k in ks]
        sc = [results["few_shot"][f"k={k}"]["scratch_mae"] for k in ks]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogx(ks, ft, "o-", label="FT (IMP2D pretrained)",
                    color="#1f77b4", linewidth=2, markersize=7)
        ax.semilogx(ks, sc, "s--", label="From scratch",
                    color="#d62728", linewidth=2, markersize=7)
        ax.axhline(zs_mae, color="gray", linestyle=":",
                   label=f"Zero-shot ({zs_mae:.2f})")
        ax.axhline(mean_pred_mae, color="black", linestyle=":",
                   label=f"Mean-predictor ({mean_pred_mae:.2f})")
        ax.set_xlabel("Few-shot k (training samples on dft_2d)", fontsize=11)
        ax.set_ylabel("Test MAE (eV/atom)", fontsize=11)
        ax.set_title("Cross-task transfer: IMP2D defect → dft_2d pristine",
                     fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_fig = FIG_DIR / "fig_dft2d_transfer.png"
        fig.savefig(out_fig, dpi=180)
        plt.close(fig)
        print(f"figure saved -> {out_fig}")

    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
