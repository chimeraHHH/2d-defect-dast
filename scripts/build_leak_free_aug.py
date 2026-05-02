"""Build a leak-free augmented dataset.

Workflow:
1. Load cleaned_dataset.pkl (10641 originals).
2. Reproduce the canonical 80/10/10 split with seed=42 — same 1065 cleaned
   originals as the no-aug baseline test set.
3. Apply rotation + Gaussian-perturbation augmentation **only** to the train
   originals (8512 samples). Augmentation produces 2 extra copies per train
   sample → 25536 train, plus the original 1064 val and 1065 test.
4. Concatenate `train_aug + val_orig + test_orig` and save as
   ``data/processed/augmented_dataset_safe.pkl``.

Crucially the mapping back to splits is preserved by sample order: the first
25536 samples are train (orig + 2 augmentations), then 1064 val originals,
then 1065 test originals. ``CrystalGraphDataset`` sees this as a flat list of
27665 samples; we expose a custom split function that respects the layout.
"""
from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import split_indices  # noqa: E402
from src.graph import build_graph  # noqa: E402


def _atoms(sample) -> Atoms:
    return Atoms(
        numbers=sample["numbers"],
        positions=sample["positions"],
        cell=sample["cell"],
        pbc=True,
    )


def _rebuild(sample, atoms: Atoms, tag: str) -> dict:
    g = build_graph(atoms, cutoff=5.0)
    new = deepcopy(sample)
    new.update(
        {
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
    )
    new["unique_id"] = f"{sample['unique_id']}_{tag}"
    new.setdefault("metadata", {})
    new["metadata"]["augmented"] = tag
    return new


def rotate(sample, rng: np.random.Generator) -> dict:
    angle = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    atoms = _atoms(sample)
    atoms.set_positions(atoms.get_positions() @ R.T)
    atoms.set_cell(np.asarray(atoms.get_cell()) @ R.T)
    return _rebuild(sample, atoms, "rot")


def perturb(sample, rng: np.random.Generator, sigma: float = 0.02) -> dict:
    atoms = _atoms(sample)
    pos = atoms.get_positions()
    atoms.set_positions(pos + rng.normal(0.0, sigma, pos.shape))
    return _rebuild(sample, atoms, "pert")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--output", default="data/processed/augmented_dataset_safe.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.02)
    args = parser.parse_args()

    src = ROOT / args.input
    dst = ROOT / args.output

    print(f"Loading {src}")
    with open(src, "rb") as f:
        data = pickle.load(f)
    print(f"Originals: {len(data)}")

    train_idx, val_idx, test_idx = split_indices(len(data), 0.8, 0.1, args.seed)
    print(f"Splits (cleaned, seed={args.seed}): "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    rng = np.random.default_rng(args.seed)

    train_section = []
    for i in train_idx:
        s = data[i]
        train_section.append(s)
        train_section.append(rotate(s, rng))
        train_section.append(perturb(s, rng, sigma=args.sigma))

    val_section = [data[i] for i in val_idx]
    test_section = [data[i] for i in test_idx]

    final = train_section + val_section + test_section
    n_train, n_val, n_test = len(train_section), len(val_section), len(test_section)
    print(f"Built leak-free aug: train={n_train} (3x orig) + val={n_val} + test={n_test}"
          f" = {len(final)} samples")

    meta = {
        "version": "leak_free_v1",
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": args.seed,
        "sigma": args.sigma,
    }

    with open(dst, "wb") as f:
        pickle.dump({"data": final, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
