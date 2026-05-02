"""Physics-informed data augmentation for the cleaned defect dataset.

Two strategies, both guaranteed to leave formation energy invariant:

* random in-plane rotation: rotate atomic coordinates and lattice basis by
  the same SO(2) action. Rotation invariance of formation energy is exact.
* Gaussian coordinate perturbation: add an isotropic noise of standard
  deviation ``sigma`` (default 0.02 Å). The dataset already contains DFT-
  relaxed positions, so this is a small ``approximation'' equivalent to
  thermal jitter and helps the model learn smooth potential surfaces.

Both functions return a fresh list of dictionaries that share the same
schema as the cleaned dataset, including pre-computed graph features. The
graph is rebuilt for the augmented sample with ``src.graph.build_graph``
so that ``edge_dist``, ``triplet`` etc. stay consistent.
"""
from __future__ import annotations

import argparse
import copy
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph  # noqa: E402


def _atoms_from_sample(sample) -> Atoms:
    return Atoms(
        numbers=sample["numbers"],
        positions=sample["positions"],
        cell=sample["cell"],
        pbc=True,
    )


def _rebuild(sample, atoms: Atoms, tag: str) -> dict:
    g = build_graph(atoms, cutoff=5.0)
    new = copy.deepcopy(sample)
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


def rotate_in_plane(sample, rng: np.random.Generator) -> dict:
    angle = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    atoms = _atoms_from_sample(sample)
    atoms.set_positions(atoms.get_positions() @ R.T)
    atoms.set_cell(np.asarray(atoms.get_cell()) @ R.T)
    return _rebuild(sample, atoms, "rot")


def perturb_positions(sample, rng: np.random.Generator, sigma: float = 0.02) -> dict:
    atoms = _atoms_from_sample(sample)
    pos = atoms.get_positions()
    noise = rng.normal(0.0, sigma, pos.shape)
    atoms.set_positions(pos + noise)
    return _rebuild(sample, atoms, "pert")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--output", default="data/processed/augmented_dataset.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.02)
    args = parser.parse_args()

    src = ROOT / args.input
    dst = ROOT / args.output

    print(f"Loading {src}")
    with open(src, "rb") as f:
        data = pickle.load(f)
    rng = np.random.default_rng(args.seed)

    augmented = list(data)
    for sample in data:
        augmented.append(rotate_in_plane(sample, rng))
        augmented.append(perturb_positions(sample, rng, sigma=args.sigma))

    rng.shuffle(augmented)
    with open(dst, "wb") as f:
        pickle.dump(augmented, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(augmented)} samples to {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
