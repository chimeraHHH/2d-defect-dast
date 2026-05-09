"""Extended physics-informed augmentation: multi-σ perturbation + lattice strain.

Extends the leak-free augmentation from 3× to ~10× by adding:
1. Multiple perturbation sigmas (0.01, 0.02, 0.03, 0.05 Å) — thermal jitter at
   different effective temperatures.  Ef is invariant under small displacements
   around the DFT-relaxed minimum.
2. In-plane biaxial strain (±1%, ±2%) — elastic regime where Ef change is
   second-order and negligible for the label.
3. Two independent random rotations (instead of one).

All augmentation is applied ONLY to the training split (leak-free contract).
Output format matches leak_free_v1 so the existing training loop picks it up
without changes.
"""
from __future__ import annotations

import argparse
import copy
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph


def _atoms_from_sample(sample):
    from ase import Atoms
    return Atoms(
        numbers=sample["numbers"],
        positions=sample["positions"],
        cell=sample["cell"],
        pbc=True,
    )


def _rebuild(sample, atoms, tag: str) -> dict:
    g = build_graph(atoms, cutoff=5.0)
    new = copy.deepcopy(sample)
    new.update({
        "numbers": g["numbers"],
        "positions": g["positions"],
        "cell": g["cell"],
        "edge_index": g["edge_index"],
        "edge_dist": g["edge_dist"],
        "edge_offset": g["edge_offset"],
        "triplet_index": g["triplet_index"],
        "angles": g["angles"],
        "dist_matrix": g["dist_matrix"],
    })
    new["unique_id"] = f"{sample['unique_id']}_{tag}"
    new.setdefault("metadata", {})
    new["metadata"]["augmented"] = tag
    return new


def rotate_in_plane(sample, rng):
    angle = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    atoms = _atoms_from_sample(sample)
    atoms.set_positions(atoms.get_positions() @ R.T)
    atoms.set_cell(np.asarray(atoms.get_cell()) @ R.T)
    return _rebuild(sample, atoms, f"rot{angle:.2f}")


def perturb_positions(sample, rng, sigma: float = 0.02):
    atoms = _atoms_from_sample(sample)
    pos = atoms.get_positions()
    noise = rng.normal(0.0, sigma, pos.shape)
    atoms.set_positions(pos + noise)
    return _rebuild(sample, atoms, f"pert{sigma}")


def strain_in_plane(sample, rng, strain_pct: float):
    """Apply uniform in-plane biaxial strain to cell and positions."""
    factor = 1.0 + strain_pct / 100.0
    atoms = _atoms_from_sample(sample)
    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    pos = atoms.get_positions().copy()
    # scale a and b vectors (indices 0, 1), keep c (index 2) for 2D
    cell[0, :2] *= factor
    cell[1, :2] *= factor
    pos[:, :2] *= factor
    atoms.set_cell(cell)
    atoms.set_positions(pos)
    return _rebuild(sample, atoms, f"strain{strain_pct:+.1f}")


def augment_single(sample, rng, sigmas, strains):
    """Generate all augmented variants for one sample."""
    augmented = []
    # 2 random rotations
    for i in range(2):
        augmented.append(rotate_in_plane(sample, rng))
    # multi-sigma perturbations
    for sigma in sigmas:
        augmented.append(perturb_positions(sample, rng, sigma=sigma))
    # in-plane strains
    for s in strains:
        augmented.append(strain_in_plane(sample, rng, s))
    return augmented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--output", default="data/processed/extended_aug_dataset_safe.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--strains", type=float, nargs="+", default=[-2.0, -1.0, 1.0, 2.0])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    src = ROOT / args.input
    dst = ROOT / args.output

    print(f"Loading {src}")
    with open(src, "rb") as f:
        data = pickle.load(f)

    n = len(data)
    n_train = int(args.train_ratio * n)
    n_val = int(args.val_ratio * n)
    n_test = n - n_train - n_val

    # deterministic split (same as leak-free pipeline)
    rng_split = np.random.default_rng(args.seed)
    indices = np.arange(n)
    rng_split.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_orig = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]

    # augment train only
    rng_aug = np.random.default_rng(args.seed + 1)
    n_augs_per_sample = 2 + len(args.sigmas) + len(args.strains)
    total_train = len(train_orig) * (1 + n_augs_per_sample)
    print(f"Augmenting {len(train_orig)} train samples × {1 + n_augs_per_sample} "
          f"= {total_train} total train")
    print(f"  Sigmas: {args.sigmas}")
    print(f"  Strains: {args.strains}%")

    augmented_train = list(train_orig)  # originals first
    t0 = time.time()
    for sample in tqdm(train_orig, desc="Augmenting"):
        augmented_train.extend(
            augment_single(sample, rng_aug, args.sigmas, args.strains)
        )
    dt = time.time() - t0
    print(f"Augmentation done in {dt:.0f}s, {len(augmented_train)} train samples")

    # shuffle augmented train
    rng_aug.shuffle(augmented_train)

    # pack into leak-free format
    all_data = augmented_train + val_data + test_data
    meta = {
        "version": "extended_aug_v1",
        "n_train": len(augmented_train),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "augmentation": {
            "rotations": 2,
            "sigmas": args.sigmas,
            "strains": args.strains,
            "multiplier": 1 + n_augs_per_sample,
        },
        "seed": args.seed,
    }
    blob = {"data": all_data, "meta": meta}
    with open(dst, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    sz = dst.stat().st_size / 1e9
    print(f"Wrote {len(all_data)} samples to {dst} ({sz:.2f} GB)")
    print(f"  Train: {meta['n_train']}, Val: {meta['n_val']}, Test: {meta['n_test']}")


if __name__ == "__main__":
    main()
