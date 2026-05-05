"""Build leak-free 3x augmented versions of all 4 multi-source datasets.

For each source we apply the same split-then-augment protocol used for IMP2D
in `build_leak_free_aug.py`:
  1. split_indices(seed=42, 0.8, 0.1, 0.1) → train / val / test indices
  2. apply rotation + Gaussian perturbation (3x total: original + rotated +
     perturbed) to the TRAIN fold only
  3. write the source pkl as ordered (train_aug | val | test) with explicit
     n_train/n_val/n_test in meta so make_splits triggers the leak-free path

Output: one .pkl per source under data/processed/
  - aug_imp2d_safe.pkl       (already exists as augmented_dataset_safe.pkl;
                              we reuse it through build_leak_free_aug if
                              missing, but most pipelines have it cached)
  - aug_jarvis_2d_safe.pkl
  - aug_jarvis_3d_safe.pkl
  - aug_dft_3d_safe.pkl

Each carries meta:
  version: leak_free_aug_v1_per_source
  source: <name>
  n_train, n_val, n_test
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


def rotate_sample(s, rng):
    angle = rng.uniform(0, 2 * math.pi)
    c, sn = math.cos(angle), math.sin(angle)
    R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    new_pos = (s["positions"].astype(np.float64) @ R.T).astype(np.float32)
    new_cell = (s["cell"].astype(np.float64) @ R.T).astype(np.float32)
    g = _rebuild_graph(s["numbers"], new_pos, new_cell)
    out = deepcopy(s)
    out.update(g)
    out["unique_id"] = f"{s.get('unique_id', s.get('id', '?'))}_rot"
    out.setdefault("metadata", {})["augmented"] = "rot"
    return out


def perturb_sample(s, rng, sigma=0.02):
    pos = s["positions"].astype(np.float64)
    noise = rng.normal(0.0, sigma, pos.shape)
    new_pos = (pos + noise).astype(np.float32)
    g = _rebuild_graph(s["numbers"], new_pos, s["cell"])
    out = deepcopy(s)
    out.update(g)
    out["unique_id"] = f"{s.get('unique_id', s.get('id', '?'))}_pert"
    out.setdefault("metadata", {})["augmented"] = "pert"
    return out


def process_source(name: str, src_path: Path, out_path: Path) -> None:
    if not src_path.exists():
        print(f"  SKIP {name}: {src_path} not found")
        return
    print(f"\n=== {name} from {src_path.name} ===")
    with open(src_path, "rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and "data" in blob:
        data = blob["data"]
    else:
        data = blob
    print(f"  N total = {len(data)}")
    train_idx, val_idx, test_idx = split_indices(len(data), 0.8, 0.1, 42)
    print(f"  split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    rng = np.random.default_rng(42)
    train_aug = []
    t0 = time.time()
    for i, idx in enumerate(train_idx):
        s = data[idx]
        if "positions" not in s:
            print(f"  WARN sample {idx} missing positions; skipping aug")
            train_aug.append(s)
            continue
        train_aug.append(s)
        try:
            train_aug.append(rotate_sample(s, rng))
            train_aug.append(perturb_sample(s, rng))
        except Exception as e:
            print(f"  WARN sample {idx} aug failed: {e}; keeping original only")
        if (i + 1) % 2000 == 0:
            print(f"    augmented {i+1}/{len(train_idx)} ({time.time()-t0:.0f}s)")

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
    }
    print(f"  → train_aug={len(train_aug)} val={len(val_section)} test={len(test_section)}")
    with open(out_path, "wb") as f:
        pickle.dump({"data": out_data, "meta": out_meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    out_dir = ROOT / "data" / "processed"
    for name, src in SOURCES.items():
        out_path = out_dir / OUT_NAMES[name]
        if out_path.exists():
            with open(out_path, "rb") as f:
                blob = pickle.load(f)
            meta = blob.get("meta", {})
            if (
                isinstance(blob, dict)
                and meta.get("version") == "leak_free_aug_v1_per_source"
            ):
                print(f"= {name}: existing {out_path.name} already at v1; skipping")
                continue
        process_source(name, src, out_path)


if __name__ == "__main__":
    main()
