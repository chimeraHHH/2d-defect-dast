#!/usr/bin/env python3
"""Compute Atom-Specific Persistent Homology (ASPH) features for all structures.

Implements the ASPH approach from Fang & Yan, Chem. Mater. 2025:
  - For each atom, build point clouds of each element type within cutoff
  - Compute Rips persistent homology in dimensions H0, H1, H2
  - Extract statistical summaries (mean, std, max, min, weighted sum)
  - PCA reduce to N principal components per atom

Output: .pkl file with ASPH features that can be loaded by the dataset class.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def compute_rips_features(point_cloud: np.ndarray, maxdim: int = 2,
                          thresh: float = 10.0) -> np.ndarray:
    """Compute persistent homology features for a single point cloud.

    Returns a fixed-length feature vector summarizing the persistence diagram.
    H0: 5 features (death only — birth is always 0)
    H1: 15 features (birth, death, persistence × 5 stats each)
    H2: 15 features (birth, death, persistence × 5 stats each)
    Total: 35 features per point cloud.
    """
    from ripser import ripser

    features = np.zeros(35, dtype=np.float32)

    if len(point_cloud) < 2:
        return features

    try:
        result = ripser(point_cloud, maxdim=min(maxdim, 2), thresh=thresh)
        diagrams = result['dgms']
    except Exception:
        return features

    idx = 0
    for dim in range(min(maxdim + 1, len(diagrams))):
        dgm = diagrams[dim]
        # Remove infinite persistence points
        finite_mask = np.isfinite(dgm[:, 1])
        dgm = dgm[finite_mask]

        if len(dgm) == 0:
            if dim == 0:
                idx += 5
            else:
                idx += 15
            continue

        births = dgm[:, 0]
        deaths = dgm[:, 1]
        persistence = deaths - births

        def stats(arr):
            """5 statistics: mean, std, max, min, weighted sum."""
            if len(arr) == 0:
                return [0.0] * 5
            return [
                np.mean(arr),
                np.std(arr) if len(arr) > 1 else 0.0,
                np.max(arr),
                np.min(arr),
                np.sum(arr * persistence / (persistence.sum() + 1e-10)),
            ]

        if dim == 0:
            # H0: only death values are meaningful (birth is always 0)
            features[idx:idx + 5] = stats(deaths)
            idx += 5
        else:
            # H1, H2: birth, death, persistence
            features[idx:idx + 5] = stats(births)
            idx += 5
            features[idx:idx + 5] = stats(deaths)
            idx += 5
            features[idx:idx + 5] = stats(persistence)
            idx += 5

    return features


def compute_asph_for_structure(positions: np.ndarray,
                               atomic_numbers: np.ndarray,
                               cell: np.ndarray,
                               cutoff: float = 10.0,
                               maxdim: int = 2) -> np.ndarray:
    """Compute ASPH features for all atoms in a structure.

    For each atom i, for each unique element type Z in the structure:
      1. Find all atoms of type Z within cutoff of atom i
      2. Build point cloud from their relative positions
      3. Compute Rips PH → 35 features

    Then sum/average across element types → 35 features per atom.

    Args:
        positions: (N, 3) fractional or Cartesian coordinates
        atomic_numbers: (N,) atomic numbers
        cell: (3, 3) unit cell matrix
        cutoff: radius cutoff for neighbor search
        maxdim: maximum homology dimension

    Returns:
        (N, 35) ASPH features for each atom
    """
    N = len(positions)
    unique_elements = np.unique(atomic_numbers)
    n_elements = len(unique_elements)

    # Build minimum-image distance matrix
    # For each atom, find neighbors within cutoff using periodic boundary conditions
    cart_pos = positions  # assume already Cartesian

    # All-pairs distances with minimum image convention
    all_features = np.zeros((N, 35), dtype=np.float32)

    for i in range(N):
        elem_features = []
        for Z in unique_elements:
            # Find atoms of element Z
            z_mask = atomic_numbers == Z
            z_indices = np.where(z_mask)[0]

            if len(z_indices) == 0:
                continue

            # Compute distances from atom i to all atoms of type Z
            # Use minimum image convention
            diffs = cart_pos[z_indices] - cart_pos[i]

            # Apply minimum image convention
            try:
                cell_inv = np.linalg.inv(cell)
                frac_diffs = diffs @ cell_inv
                frac_diffs -= np.round(frac_diffs)
                diffs = frac_diffs @ cell
            except np.linalg.LinAlgError:
                pass

            distances = np.linalg.norm(diffs, axis=1)

            # Filter by cutoff and exclude self
            within_cutoff = (distances < cutoff) & (distances > 0.1)
            if not within_cutoff.any():
                continue

            # Build point cloud from relative positions
            cloud = diffs[within_cutoff]

            # Include the center atom at origin
            cloud = np.vstack([[0, 0, 0], cloud])

            # Compute PH features
            fea = compute_rips_features(cloud, maxdim=maxdim, thresh=cutoff)
            elem_features.append(fea)

        if elem_features:
            all_features[i] = np.mean(elem_features, axis=0)

    return all_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--cutoff", type=float, default=8.0,
                        help="Cutoff radius for PH (default 8A, paper uses 10A)")
    parser.add_argument("--maxdim", type=int, default=1,
                        help="Max homology dimension (0,1,2). 2 is slow, 1 is faster.")
    parser.add_argument("--n-components", type=int, default=8,
                        help="PCA components per atom")
    parser.add_argument("--output", default="data/processed/asph_features.pkl")
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    data_path = ROOT / args.data_path
    print(f"Loading dataset from {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    n_samples = len(data)
    print(f"Computing ASPH features for {n_samples} structures...")
    print(f"  cutoff={args.cutoff}A, maxdim={args.maxdim}")

    t0 = time.time()

    # Sequential computation (multiprocessing has pickling issues with dataset)
    all_features = []
    for i in range(n_samples):
        sample = data[i]
        positions = sample["positions"].numpy() if hasattr(sample["positions"], "numpy") else np.array(sample["positions"])
        atomic_numbers = sample["numbers"].numpy() if hasattr(sample["numbers"], "numpy") else np.array(sample["numbers"])
        cell = sample["cell"].numpy() if hasattr(sample["cell"], "numpy") else np.array(sample["cell"])
        fea = compute_asph_for_structure(positions, atomic_numbers, cell,
                                          cutoff=args.cutoff, maxdim=args.maxdim)
        all_features.append(fea)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate
            print(f"  [{i+1}/{n_samples}] {rate:.1f} structures/s, "
                  f"ETA {remaining:.0f}s")

    elapsed = time.time() - t0
    print(f"ASPH computation done in {elapsed:.0f}s")

    # Concatenate all features for PCA
    all_asph = []
    atom_counts = []
    for fea in all_features:
        all_asph.append(fea)
        atom_counts.append(len(fea))

    all_asph_flat = np.concatenate(all_asph, axis=0)
    print(f"Total atoms: {len(all_asph_flat)}, features: {all_asph_flat.shape[1]}")

    # Remove zero-variance features
    var = all_asph_flat.var(axis=0)
    nonzero_mask = var > 1e-10
    print(f"Non-zero variance features: {nonzero_mask.sum()}/{len(var)}")
    all_asph_filtered = all_asph_flat[:, nonzero_mask]

    # PCA
    n_comp = min(args.n_components, all_asph_filtered.shape[1])
    pca = PCA(n_components=n_comp)
    all_pca_flat = pca.fit_transform(all_asph_filtered)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_comp} components explain {explained*100:.1f}% variance")

    # Split back into per-structure features
    asph_per_structure = []
    offset = 0
    for count in atom_counts:
        asph_per_structure.append(
            all_pca_flat[offset:offset + count].astype(np.float32)
        )
        offset += count

    # Save
    output = {
        "asph_features": asph_per_structure,  # list of (N_i, n_comp) arrays
        "pca_model": pca,
        "nonzero_mask": nonzero_mask,
        "n_components": n_comp,
        "cutoff": args.cutoff,
        "maxdim": args.maxdim,
        "explained_variance": float(explained),
        "n_samples": n_samples,
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
