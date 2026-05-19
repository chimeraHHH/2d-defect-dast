#!/usr/bin/env python3
"""Compute Atom-Specific Persistent Homology (ASPH) features for defect sites.

Persistent homology captures topological features of the local atomic
neighborhood that coordination numbers and distances cannot:
  - 0-dim PH: connected components → measures local crowding/isolation
  - 1-dim PH: loops/rings → detects cage-like structures and voids

These are especially relevant for defect formation energy because:
  - Interstitials create crowded, distorted neighborhoods
  - Adsorbates sit on surfaces with specific void patterns

References:
  - Fang & Yan, Chem. Mater. 2025 (55% MAE reduction on defect Ef)
  - Hiraoka et al., PNAS 2016 (persistent homology for materials)

Usage:
    python scripts/compute_asph.py [--rcut 6.0] [--n-components 8]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from ripser import ripser

ROOT = Path(__file__).resolve().parent.parent


def compute_asph_single(positions, defect_idx, rcut=6.0, maxdim=1):
    """Compute ASPH features for a single defect site.

    Args:
        positions: (N, 3) atomic positions
        defect_idx: index of the defect atom (or -1 for none)
        rcut: cutoff radius for local neighborhood
        maxdim: max homology dimension (0=components, 1=loops)

    Returns:
        features: (n_features,) ASPH feature vector
    """
    N = len(positions)

    # Find defect center
    if defect_idx >= 0 and defect_idx < N:
        center = positions[defect_idx]
    else:
        # No defect marked — use centroid
        center = positions.mean(axis=0)

    # Extract local neighborhood
    dists = np.linalg.norm(positions - center, axis=1)
    local_mask = dists < rcut
    local_pts = positions[local_mask]
    n_local = len(local_pts)

    if n_local < 3:
        # Too few atoms for meaningful topology
        return np.zeros(get_n_features(maxdim))

    # Compute pairwise distance matrix for local neighborhood
    diff = local_pts[:, None, :] - local_pts[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    # Run ripser for persistent homology
    try:
        result = ripser(dist_matrix, maxdim=maxdim, distance_matrix=True)
    except Exception:
        return np.zeros(get_n_features(maxdim))

    features = []
    for dim in range(maxdim + 1):
        dgm = result['dgms'][dim]
        # Filter out infinite-death points (except for dim-0 which has one)
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))
        lifetimes = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([0.0])

        # Statistics of persistence diagram
        features.extend([
            len(finite),                              # count of features
            lifetimes.max() if len(lifetimes) > 0 else 0,     # max lifetime
            lifetimes.mean() if len(lifetimes) > 0 else 0,    # mean lifetime
            lifetimes.sum() if len(lifetimes) > 0 else 0,     # total lifetime
            lifetimes.std() if len(lifetimes) > 1 else 0,     # lifetime std
            np.median(lifetimes) if len(lifetimes) > 0 else 0, # median lifetime
            finite[:, 0].mean() if len(finite) > 0 else 0,    # mean birth
            finite[:, 1].mean() if len(finite) > 0 else 0,    # mean death
        ])

    # Add local structure statistics
    features.extend([
        n_local,                                    # number of local atoms
        n_local / max(N, 1),                        # fraction of atoms in neighborhood
        dist_matrix[dist_matrix > 0].mean(),        # mean pairwise distance
        dist_matrix[dist_matrix > 0].std(),         # pairwise distance std
    ])

    return np.array(features, dtype=np.float32)


def get_n_features(maxdim=1):
    """Return number of features per sample."""
    return 8 * (maxdim + 1) + 4  # 8 stats per dim + 4 structure stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--output", default="data/asph_features.pkl")
    parser.add_argument("--rcut", type=float, default=6.0)
    parser.add_argument("--n-components", type=int, default=8,
                        help="PCA components for dimensionality reduction")
    parser.add_argument("--maxdim", type=int, default=1,
                        help="Max homology dimension (0 or 1)")
    args = parser.parse_args()

    data_path = ROOT / args.data_path
    output_path = ROOT / args.output

    print(f"Loading dataset from {data_path}...")
    with open(data_path, "rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and "data" in blob:
        data = blob["data"]
    else:
        data = blob

    n_samples = len(data)
    n_raw = get_n_features(args.maxdim)
    print(f"Computing ASPH features for {n_samples} samples "
          f"(rcut={args.rcut}, maxdim={args.maxdim}, raw_dim={n_raw})...")

    raw_features = np.zeros((n_samples, n_raw), dtype=np.float32)
    t0 = time.time()

    for i, sample in enumerate(data):
        positions = sample["positions"]
        defect_mask = sample.get("defect_mask", np.zeros(len(positions), dtype=np.int64))
        defect_idx = np.flatnonzero(defect_mask)
        defect_idx = defect_idx[-1] if len(defect_idx) > 0 else -1

        raw_features[i] = compute_asph_single(
            positions, defect_idx, rcut=args.rcut, maxdim=args.maxdim
        )

        if (i + 1) % 500 == 0 or i == n_samples - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"  [{i+1}/{n_samples}] {rate:.1f} samples/s, ETA {eta:.0f}s")

    # Normalize features
    mean = raw_features.mean(axis=0)
    std = raw_features.std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (raw_features - mean) / std

    # PCA reduction
    from sklearn.decomposition import PCA
    n_comp = min(args.n_components, n_raw, n_samples)
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(normalized)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_raw} → {n_comp} components, explained variance = {explained:.3f}")

    # Broadcast to per-atom features (same value for all atoms in a sample)
    asph_per_atom = []
    for i, sample in enumerate(data):
        n_atoms = len(sample["positions"])
        asph_per_atom.append(
            np.tile(reduced[i], (n_atoms, 1)).astype(np.float32)
        )

    # Save
    save_dict = {
        "asph_features": asph_per_atom,
        "n_components": n_comp,
        "explained_variance": float(explained),
        "pca_mean": mean,
        "pca_std": std,
        "pca_components": pca.components_,
        "rcut": args.rcut,
        "maxdim": args.maxdim,
    }
    with open(output_path, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Saved ASPH features to {output_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
