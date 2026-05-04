"""Build leak-free augmented (defect, pristine) pairs.

For each cleaned IMP2D sample we generate three pairs in the train split:
  (i)   the original (defect, pristine)
  (ii)  a rotated (R @ defect, R @ pristine) with R ∈ SO(2) (same R)
  (iii) a perturbed (defect + N(0, σ²), pristine + N(0, σ²)) — same noise tensor
        applied to host atoms in both, so Δr stays exactly zero between the
        atoms shared by defect and pristine. (The dopant atom in defect gets
        its own noise, pristine has no dopant atom.)

We follow the leak-free protocol from `build_leak_free_aug.py`:
  - Split first (8512 train / 1064 val / 1065 test, seed=42)
  - Augment ONLY the train fold; val / test stay as the original cleaned
    pairs.

Output: data/processed/aug_pristine_dataset_safe.pkl with metadata
  meta = {
    "version": "leak_free_aug_pristine_v1",
    "n_train": 25536, "n_val": 1064, "n_test": 1065,
    "split_seed": 42,
    "aug_seed": 42,
  }
The data list is ORDERED: train (25536) | val (1064) | test (1065).
"""
from __future__ import annotations

import math
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import split_indices  # noqa: E402
from src.graph import build_graph  # noqa: E402
from scripts.build_pristine_pairs import find_defect_index  # noqa: E402


def _atoms(numbers, positions, cell):
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def _rebuild_graph(numbers, positions, cell):
    g = build_graph(_atoms(numbers, positions, cell), cutoff=5.0)
    return {
        "numbers": g["numbers"],
        "positions": g["positions"],
        "cell": g["cell"],
        "edge_index": g["edge_index"],
        "edge_dist": g["edge_dist"],
        "edge_offset": g["edge_offset"],
        "triplet_index": g["triplet_index"],
        "angles": g["angles"],
        "dist_matrix": g["dist_matrix"],
    }


def _rotate_pair(s, rng):
    angle = rng.uniform(0.0, 2 * math.pi)
    c, sn = math.cos(angle), math.sin(angle)
    R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    defect = deepcopy(s)
    defect_pos = (s["positions"].astype(np.float64) @ R.T).astype(np.float32)
    defect_cell = (s["cell"].astype(np.float64) @ R.T).astype(np.float32)
    g_def = _rebuild_graph(s["numbers"], defect_pos, defect_cell)
    defect.update(g_def)
    defect["unique_id"] = f"{s['unique_id']}_rot"
    defect.setdefault("metadata", {})["augmented"] = "rot"

    p = s["pristine"]
    pristine_pos = (p["positions"].astype(np.float64) @ R.T).astype(np.float32)
    pristine_cell = (p["cell"].astype(np.float64) @ R.T).astype(np.float32)
    g_pri = _rebuild_graph(p["numbers"], pristine_pos, pristine_cell)
    defect["pristine"] = g_pri
    return defect


def _perturb_pair(s, rng, sigma=0.02):
    # Apply the SAME noise tensor to host atoms shared by both supercells
    # so Δr_host between defect and pristine remains exactly zero.
    didx = s["defect_atom_index"]
    n_def = s["positions"].shape[0]
    noise_def = rng.normal(0.0, sigma, (n_def, 3)).astype(np.float64)
    # pristine atoms are defect atoms with the dopant (index didx) removed,
    # in the same order.  So noise_pri = noise_def with index didx deleted.
    keep = np.ones(n_def, dtype=bool)
    keep[didx] = False
    noise_pri = noise_def[keep]

    defect = deepcopy(s)
    defect_pos = (s["positions"].astype(np.float64) + noise_def).astype(np.float32)
    g_def = _rebuild_graph(s["numbers"], defect_pos, s["cell"])
    defect.update(g_def)
    defect["unique_id"] = f"{s['unique_id']}_pert"
    defect.setdefault("metadata", {})["augmented"] = "pert"

    p = s["pristine"]
    pristine_pos = (p["positions"].astype(np.float64) + noise_pri).astype(np.float32)
    g_pri = _rebuild_graph(p["numbers"], pristine_pos, p["cell"])
    defect["pristine"] = g_pri
    return defect


def main():
    src = ROOT / "data" / "processed" / "cleaned_dataset_with_pristine.pkl"
    dst = ROOT / "data" / "processed" / "aug_pristine_dataset_safe.pkl"

    print(f"Loading {src} ...")
    with open(src, "rb") as f:
        blob = pickle.load(f)
    data = blob["data"]
    print(f"  N = {len(data)} samples")

    train_idx, val_idx, test_idx = split_indices(len(data), 0.8, 0.1, 42)
    print(f"  split: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    rng = np.random.default_rng(42)
    t0 = time.time()
    train_aug = []
    for i, idx in enumerate(train_idx):
        s = data[idx]
        train_aug.append(s)
        train_aug.append(_rotate_pair(s, rng))
        train_aug.append(_perturb_pair(s, rng))
        if (i + 1) % 1000 == 0:
            print(f"  augmented {i+1}/{len(train_idx)} train  ({time.time()-t0:.0f}s)")
    val_section = [data[i] for i in val_idx]
    test_section = [data[i] for i in test_idx]
    out_data = train_aug + val_section + test_section

    out_meta = {
        "version": "leak_free_aug_pristine_v1",
        "n_train": len(train_aug),
        "n_val": len(val_section),
        "n_test": len(test_section),
        "split_seed": 42,
        "aug_seed": 42,
        "build_time_min": (time.time() - t0) / 60.0,
    }

    print(f"\nWriting {len(out_data)} samples ({out_meta['n_train']} train aug, "
          f"{out_meta['n_val']} val, {out_meta['n_test']} test) to {dst} ...")
    with open(dst, "wb") as f:
        pickle.dump({"data": out_data, "meta": out_meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  size: {dst.stat().st_size / 1e9:.2f} GB; total time {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
