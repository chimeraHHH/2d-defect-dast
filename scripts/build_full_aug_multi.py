"""Build leak-free 3x augmented versions of all 4 multi-source datasets.

Multiprocessing-parallelised graph rebuilder: uses up to N_WORKERS CPU
cores for the rotation + perturbation augmentations, since the
existing ASE-based ``src.graph.build_graph`` does not have a GPU
backend and the per-sample workload is embarrassingly parallel.

For each source we apply the same split-then-augment protocol used in
`build_leak_free_aug.py`:
  1. split_indices(seed=42, 0.8, 0.1, 0.1) → train / val / test indices
  2. apply rotation + Gaussian perturbation (3x: original + rotated +
     perturbed) to the TRAIN fold only, in parallel across N_WORKERS
     processes
  3. write the source pkl as ordered (train_aug | val | test) with
     explicit n_train/n_val/n_test in meta so make_splits triggers
     the leak-free path automatically (see commit b2450d0).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph  # noqa: E402

# --- inline imports of split_indices to avoid pulling whole src.dataset ---
import random


def split_indices(n_samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    indices = list(range(n_samples))
    rng.shuffle(indices)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:],
    )


SOURCES = {
    "IMP2D": ROOT / "data" / "processed" / "cleaned_dataset.pkl",
    "JARVIS-2D": ROOT / "data" / "processed" / "jarvis_2d.pkl",
    "JARVIS-3D": ROOT / "data" / "processed" / "jarvis_3d.pkl",
    "DFT-3D": ROOT / "data" / "processed" / "dft_3d_lite.pkl",
}
OUT_NAMES = {
    "IMP2D": "aug_imp2d_safe.pkl",
    "JARVIS-2D": "aug_jarvis_2d_safe.pkl",
    "JARVIS-3D": "aug_jarvis_3d_safe.pkl",
    "DFT-3D": "aug_dft_3d_safe.pkl",
}

N_WORKERS = max(1, min(12, mp.cpu_count() - 2))


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


def _augment_one(payload):
    """Worker function: take one sample, return (orig, rot, pert) triple."""
    s, seed = payload
    if "positions" not in s:
        return [s, None, None]  # caller will filter Nones
    rng = np.random.default_rng(seed)
    out = [s]

    # rotation (in-plane SO(2))
    try:
        angle = rng.uniform(0, 2 * math.pi)
        c, sn = math.cos(angle), math.sin(angle)
        R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        new_pos = (s["positions"].astype(np.float64) @ R.T).astype(np.float32)
        new_cell = (s["cell"].astype(np.float64) @ R.T).astype(np.float32)
        g = _rebuild_graph(s["numbers"], new_pos, new_cell)
        rot = deepcopy(s)
        rot.update(g)
        rot["unique_id"] = f"{s.get('unique_id', s.get('id', '?'))}_rot"
        rot.setdefault("metadata", {})["augmented"] = "rot"
        out.append(rot)
    except Exception:
        out.append(None)

    # perturb (Gaussian sigma=0.02)
    try:
        pos = s["positions"].astype(np.float64)
        noise = rng.normal(0.0, 0.02, pos.shape)
        new_pos = (pos + noise).astype(np.float32)
        g = _rebuild_graph(s["numbers"], new_pos, s["cell"])
        pert = deepcopy(s)
        pert.update(g)
        pert["unique_id"] = f"{s.get('unique_id', s.get('id', '?'))}_pert"
        pert.setdefault("metadata", {})["augmented"] = "pert"
        out.append(pert)
    except Exception:
        out.append(None)
    return out


def process_source(name: str, src_path: Path, out_path: Path) -> None:
    if not src_path.exists():
        print(f"  SKIP {name}: {src_path} not found", flush=True)
        return
    print(f"\n=== {name} from {src_path.name} ===", flush=True)
    with open(src_path, "rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and "data" in blob:
        data = blob["data"]
    else:
        data = blob
    print(f"  N total = {len(data)}", flush=True)
    train_idx, val_idx, test_idx = split_indices(len(data), 0.8, 0.1, 42)
    print(f"  split: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}", flush=True)

    payloads = [(data[idx], 42 + i) for i, idx in enumerate(train_idx)]

    t0 = time.time()
    print(f"  augmenting {len(payloads)} train samples in pool of {N_WORKERS} workers...", flush=True)
    with mp.Pool(processes=N_WORKERS) as pool:
        results = []
        for i, triple in enumerate(pool.imap_unordered(_augment_one, payloads, chunksize=8)):
            results.append(triple)
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(payloads) - i - 1) / rate
                print(f"    {i+1}/{len(payloads)} done  {rate:.0f}/s  ETA {eta:.0f}s", flush=True)

    train_aug = []
    for triple in results:
        for s in triple:
            if s is not None:
                train_aug.append(s)

    val_section = [data[i] for i in val_idx]
    test_section = [data[i] for i in test_idx]
    out_data = train_aug + val_section + test_section

    out_meta = {
        "version": "leak_free_aug_v1_per_source",
        "source": name,
        "n_train": len(train_aug),
        "n_val": len(val_section),
        "n_test": len(test_section),
        "split_seed": 42,
        "aug_seed": 42,
        "n_orig_train": len(train_idx),
        "build_time_min": (time.time() - t0) / 60.0,
        "n_workers": N_WORKERS,
    }
    print(
        f"  → train_aug={len(train_aug)} val={len(val_section)} test={len(test_section)}  "
        f"({(time.time()-t0)/60:.1f} min)", flush=True,
    )
    with open(out_path, "wb") as f:
        pickle.dump({"data": out_data, "meta": out_meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)


def main():
    out_dir = ROOT / "data" / "processed"
    print(f"Using {N_WORKERS} workers (cpu_count={mp.cpu_count()})", flush=True)
    for name, src in SOURCES.items():
        out_path = out_dir / OUT_NAMES[name]
        if out_path.exists():
            try:
                with open(out_path, "rb") as f:
                    blob = pickle.load(f)
                if (isinstance(blob, dict)
                        and blob.get("meta", {}).get("version") == "leak_free_aug_v1_per_source"):
                    print(f"= {name}: existing {out_path.name} already at v1; skipping", flush=True)
                    continue
            except Exception:
                pass
        process_source(name, src, out_path)


if __name__ == "__main__":
    main()
