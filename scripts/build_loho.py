"""Build a leave-one-host-out (LOHO) dataset.

For a held-out host, partition the cleaned IMP2D into:
  * train: all samples whose host != holdout, with per-train aug (×3)
  * val:   10% of train (random, seed 42)
  * test:  ALL samples whose host == holdout (no aug)

Saved as ``data/processed/loho_<host>.pkl`` with the same ``leak_free_v1``
meta format used by ``build_leak_free_aug.py``, so that ``CrystalGraphDataset``
+ ``make_splits`` already give the right split out of the box.
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

from src.graph import build_graph  # noqa: E402


def _atoms(s) -> Atoms:
    return Atoms(
        numbers=s["numbers"], positions=s["positions"], cell=s["cell"], pbc=True
    )


def _rebuild(sample, atoms: Atoms, tag: str) -> dict:
    g = build_graph(atoms, cutoff=5.0)
    new = deepcopy(sample)
    new.update({k: g[k] for k in (
        "numbers", "positions", "cell", "edge_index", "edge_dist",
        "edge_offset", "triplet_index", "angles", "dist_matrix",
    )})
    new["unique_id"] = f"{sample['unique_id']}_{tag}"
    new.setdefault("metadata", {})["augmented"] = tag
    return new


def rotate(s, rng):
    a = rng.uniform(0, 2 * math.pi)
    c, sn = math.cos(a), math.sin(a)
    R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    atoms = _atoms(s)
    atoms.set_positions(atoms.get_positions() @ R.T)
    atoms.set_cell(np.asarray(atoms.get_cell()) @ R.T)
    return _rebuild(s, atoms, "rot")


def perturb(s, rng, sigma=0.02):
    atoms = _atoms(s)
    atoms.set_positions(atoms.get_positions() + rng.normal(0.0, sigma, atoms.get_positions().shape))
    return _rebuild(s, atoms, "pert")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/cleaned_dataset.pkl")
    p.add_argument("--holdout", required=True, help="host string to leave out")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = ROOT / args.input
    with open(src, "rb") as f:
        data = pickle.load(f)

    in_holdout, others = [], []
    for s in data:
        host = s["metadata"].get("host", "?") or "?"
        (in_holdout if host == args.holdout else others).append(s)
    if len(in_holdout) < 30:
        raise ValueError(f"holdout host '{args.holdout}' has only {len(in_holdout)} samples")
    print(f"holdout host '{args.holdout}': {len(in_holdout)} test samples")
    print(f"remaining: {len(others)} samples (will become train + val)")

    rng = random.Random(args.seed)
    rng.shuffle(others)
    n_val = max(1, int(0.1 * len(others)))
    val_section = others[:n_val]
    train_originals = others[n_val:]
    print(f"-> val from random: {len(val_section)}; train originals: {len(train_originals)}")

    np_rng = np.random.default_rng(args.seed)
    train_section = []
    for s in train_originals:
        train_section.append(s)
        train_section.append(rotate(s, np_rng))
        train_section.append(perturb(s, np_rng))
    print(f"train after 3x aug: {len(train_section)}")

    final = train_section + val_section + in_holdout
    meta = {
        "version": "leak_free_v1",
        "holdout_host": args.holdout,
        "n_train": len(train_section),
        "n_val": len(val_section),
        "n_test": len(in_holdout),
        "seed": args.seed,
    }

    safe_host = args.holdout.replace("/", "_")
    dst = ROOT / args.output_dir / f"loho_{safe_host}.pkl"
    with open(dst, "wb") as f:
        pickle.dump({"data": final, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = dst.stat().st_size / 1e6
    print(f"saved -> {dst} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
