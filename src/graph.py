"""Periodic graph construction utilities.

Given an ASE Atoms object we produce:
  * `edge_index` (2, E) and `edge_dist` (E,)        — bonds within `cutoff`
  * `edge_offset` (E, 3)                             — image offsets used (n)
  * `triplet_index` (M, 3) and `angles` (M,)         — bond angles (j, i, k)
  * `dist_matrix` (N, N)                             — minimum-image PBC distance

Both real-space minimum-image distance and full neighbour lists are built with
ASE's `neighbor_list`, which returns image vectors for free.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list


def _pbc_distance_matrix(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """N x N minimum-image distance matrix (Cartesian)."""
    diff = positions[:, None, :] - positions[None, :, :]
    try:
        cell_inv = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return np.linalg.norm(diff, axis=-1)
    diff_frac = diff @ cell_inv
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ cell
    return np.linalg.norm(diff_cart, axis=-1)


def build_graph(atoms: Atoms, cutoff: float = 5.0) -> Dict[str, np.ndarray]:
    """Return graph dictionary suitable for downstream tensor packing."""
    # neighbor_list returns:
    #   i, j -> indices, d -> distances, D -> displacement vectors, S -> shifts (3,)
    i, j, d, D, S = neighbor_list("ijdDS", atoms, cutoff=cutoff)

    if len(i) == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_dist = np.empty((0,), dtype=np.float32)
        edge_offset = np.empty((0, 3), dtype=np.float32)
    else:
        edge_index = np.vstack([i, j]).astype(np.int64)
        edge_dist = d.astype(np.float32)
        edge_offset = S.astype(np.float32)

    # triplet construction: for each centre i, every ordered pair (j, k) of its neighbours
    triplet_idx = []
    angles = []
    if len(i) > 0:
        order = np.argsort(i, kind="stable")
        i_sorted = i[order]
        j_sorted = j[order]
        D_sorted = D[order]
        # group neighbours per centre
        unique_centres, group_starts = np.unique(i_sorted, return_index=True)
        group_starts = np.append(group_starts, len(i_sorted))
        for k_idx, centre in enumerate(unique_centres):
            s, e = group_starts[k_idx], group_starts[k_idx + 1]
            n_local = e - s
            if n_local < 2:
                continue
            neigh_j = j_sorted[s:e]
            vecs = D_sorted[s:e]
            norms = np.linalg.norm(vecs, axis=1) + 1e-12
            # all ordered pairs (excluding self); cap to keep cost manageable
            max_pairs_per_centre = 32
            pairs = []
            for a in range(n_local):
                for b in range(n_local):
                    if a == b:
                        continue
                    pairs.append((a, b))
                    if len(pairs) >= max_pairs_per_centre:
                        break
                if len(pairs) >= max_pairs_per_centre:
                    break
            for a, b in pairs:
                cos = float(np.dot(vecs[a], vecs[b]) / (norms[a] * norms[b]))
                cos = max(-1.0, min(1.0, cos))
                triplet_idx.append((int(neigh_j[a]), int(centre), int(neigh_j[b])))
                angles.append(math.acos(cos))

    if triplet_idx:
        triplet_index = np.asarray(triplet_idx, dtype=np.int64)
        angles_arr = np.asarray(angles, dtype=np.float32)
    else:
        triplet_index = np.empty((0, 3), dtype=np.int64)
        angles_arr = np.empty((0,), dtype=np.float32)

    dist_matrix = _pbc_distance_matrix(
        atoms.get_positions().astype(np.float32),
        np.asarray(atoms.get_cell()).astype(np.float32),
    ).astype(np.float32)

    return {
        "numbers": atoms.get_atomic_numbers().astype(np.int64),
        "positions": atoms.get_positions().astype(np.float32),
        "cell": np.asarray(atoms.get_cell()).astype(np.float32),
        "edge_index": edge_index,
        "edge_dist": edge_dist,
        "edge_offset": edge_offset,
        "triplet_index": triplet_index,
        "angles": angles_arr,
        "dist_matrix": dist_matrix,
    }


def torch_pbc_distance_matrix(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    try:
        cell_inv = torch.linalg.inv(cell)
    except RuntimeError:
        return torch.linalg.norm(diff, dim=-1)
    diff_frac = diff @ cell_inv
    diff_frac = diff_frac - torch.round(diff_frac)
    diff_cart = diff_frac @ cell
    return torch.linalg.norm(diff_cart, dim=-1)
