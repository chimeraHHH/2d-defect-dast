"""Periodic graph construction utilities.

Given an ASE Atoms object we produce:
  * `edge_index` (2, E) and `edge_dist` (E,)        — bonds within `cutoff`
  * `edge_offset` (E, 3)                             — image offsets used (n)
  * `triplet_index` (M, 3) and `angles` (M,)         — bond angles (j, i, k)
  * `dist_matrix` (N, N)                             — minimum-image PBC distance

Both real-space minimum-image distance and full neighbour lists are built with
ASE.  Dense distances use :func:`ase.geometry.find_mic`, whose general branch
Minkowski-reduces oblique cells before resolving the shortest image.  The
component-wise ``frac -= round(frac)`` shortcut is not exact for a general
oblique lattice and must not be reintroduced here.

The local angle operator consumes only a triplet centre and its angle; the
first and third entries of ``triplet_index`` are retained for provenance but
are not read by any local interaction layer.  To keep the operator bounded and
permutation invariant, each centre's complete ordered-angle multiset is
represented by at most ``MAX_TRIPLETS_PER_CENTRE`` deterministic order
statistics.  This replaces the legacy "first 32" neighbour-pair truncation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list


GRAPH_BUILDER_VERSION = "exact_mic_invariant_triplets_v1"
MAX_TRIPLETS_PER_CENTRE = 32


def _pbc_distance_matrix(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Return the exact ASE minimum-image distance matrix in Cartesian units.

    ``pbc`` is mandatory.  In particular, callers rebuilding an archived
    dataset must recover it from the authoritative structure source rather
    than silently assuming three periodic axes.
    """
    positions64 = np.asarray(positions, dtype=np.float64)
    cell64 = np.asarray(cell, dtype=np.float64)
    pbc_bool = np.asarray(pbc, dtype=bool)
    if positions64.ndim != 2 or positions64.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if cell64.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    if pbc_bool.shape != (3,):
        raise ValueError("pbc must have shape (3,)")

    diff = positions64[:, None, :] - positions64[None, :, :]
    _, lengths = find_mic(diff.reshape(-1, 3), cell64, pbc=pbc_bool)
    distances = np.asarray(lengths, dtype=np.float64).reshape(
        len(positions64), len(positions64)
    )
    np.fill_diagonal(distances, 0.0)
    return distances


def _midpoint_order_statistic_ranks(n_values: int, n_keep: int) -> np.ndarray:
    """Return unique midpoint-stratum ranks for ``n_values > n_keep``.

    Rank ``q`` is ``floor((2q + 1) * n_values / (2 * n_keep))`` for
    ``q = 0, ..., n_keep - 1``.  The precondition ``n_values > n_keep`` makes
    the ranks strictly increasing.  Midpoint strata avoid privileging either
    endpoint of the sorted angular distribution.
    """
    if n_keep < 1:
        raise ValueError("n_keep must be positive")
    if n_values <= n_keep:
        raise ValueError("midpoint ranks require n_values > n_keep")
    q = np.arange(n_keep, dtype=np.int64)
    ranks = ((2 * q + 1) * int(n_values)) // (2 * int(n_keep))
    if len(np.unique(ranks)) != n_keep or ranks[0] < 0 or ranks[-1] >= n_values:
        raise RuntimeError("midpoint rank construction violated its contract")
    return ranks


def _invariant_triplets(
    centres: np.ndarray,
    neighbours: np.ndarray,
    displacements: np.ndarray,
    max_per_centre: int = MAX_TRIPLETS_PER_CENTRE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a bounded, permutation-invariant centre-angle representation.

    For a centre with degree ``d``, every unordered neighbour-image pair
    contributes its angle twice, reproducing the legacy ordered-pair
    multiplicity.  Let ``A`` be this complete multiset of ``d(d-1)`` angles.
    If ``|A| <= max_per_centre``, all values are retained.  Otherwise the
    sorted multiset is represented by the midpoint order statistics returned
    by :func:`_midpoint_order_statistic_ranks`.

    The resulting retained angle multiset is a function only of the complete
    physical angle multiset, never of atom indices or neighbour-list order.
    Equal-angle ties therefore cannot change the local operator.  The current
    local interaction layers read only ``triplet_index[:, 1]`` and ``angles``;
    first/third atom identities are provenance fields and do not enter a
    message.  We retain representative physical pairs for those fields, while
    making no claim that their identity is unique within an equal-angle tie.

    Peak temporary work is ``O(d^2)`` for one centre; persistent storage is
    ``O(max_per_centre)`` per centre.
    """
    if max_per_centre < 1:
        raise ValueError("max_per_centre must be positive")
    centres = np.asarray(centres, dtype=np.int64)
    neighbours = np.asarray(neighbours, dtype=np.int64)
    displacements = np.asarray(displacements, dtype=np.float64)
    if centres.ndim != 1 or neighbours.shape != centres.shape:
        raise ValueError("centres and neighbours must be aligned vectors")
    if displacements.shape != (len(centres), 3):
        raise ValueError("displacements must have shape (E, 3)")

    triplet_rows = []
    retained_angles = []
    if len(centres) == 0:
        return (
            np.empty((0, 3), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    order = np.argsort(centres, kind="stable")
    centres_sorted = centres[order]
    neighbours_sorted = neighbours[order]
    vectors_sorted = displacements[order]
    unique_centres, starts = np.unique(centres_sorted, return_index=True)
    stops = np.append(starts[1:], len(centres_sorted))

    for centre, start, stop in zip(unique_centres, starts, stops):
        degree = int(stop - start)
        if degree < 2:
            continue
        local_neighbours = neighbours_sorted[start:stop]
        vectors = vectors_sorted[start:stop]
        left, right = np.triu_indices(degree, k=1)
        left_vec = vectors[left]
        right_vec = vectors[right]
        denominator = np.linalg.norm(left_vec, axis=1) * np.linalg.norm(
            right_vec, axis=1
        )
        if np.any(denominator <= 0.0):
            raise ValueError("zero-length neighbour displacement in angle graph")
        cosine = np.einsum("ij,ij->i", left_vec, right_vec) / denominator
        unordered_angles = np.arccos(np.clip(cosine, -1.0, 1.0))

        # Duplicate each unordered pair in both orientations.  Sorting only by
        # angle is intentional: equal-angle ties feed identical inputs to the
        # centre-angle operator, so no atom-index tie-break is needed.
        all_angles = np.concatenate([unordered_angles, unordered_angles])
        all_left = np.concatenate([left, right])
        all_right = np.concatenate([right, left])
        angle_order = np.argsort(all_angles, kind="stable")
        all_angles = all_angles[angle_order]
        all_left = all_left[angle_order]
        all_right = all_right[angle_order]
        if len(all_angles) > max_per_centre:
            selected = _midpoint_order_statistic_ranks(
                len(all_angles), max_per_centre
            )
            all_angles = all_angles[selected]
            all_left = all_left[selected]
            all_right = all_right[selected]

        for angle, local_left, local_right in zip(
            all_angles, all_left, all_right
        ):
            triplet_rows.append(
                (
                    int(local_neighbours[int(local_left)]),
                    int(centre),
                    int(local_neighbours[int(local_right)]),
                )
            )
            retained_angles.append(float(angle))

    if not triplet_rows:
        return (
            np.empty((0, 3), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.asarray(triplet_rows, dtype=np.int64),
        np.asarray(retained_angles, dtype=np.float32),
    )


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

    triplet_index, angles_arr = _invariant_triplets(i, j, D)

    dist_matrix = _pbc_distance_matrix(
        atoms.get_positions(),
        np.asarray(atoms.get_cell()),
        atoms.get_pbc(),
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


def torch_pbc_distance_matrix(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """CPU tensor wrapper around the exact ASE minimum-image implementation.

    This preprocessing helper is intentionally non-differentiable.  It rejects
    CUDA/gradient-bearing tensors rather than silently falling back to the
    inexact component-wise fractional wrapping used by the legacy builder.
    """
    if positions.device.type != "cpu" or cell.device.type != "cpu":
        raise ValueError("exact graph preprocessing expects CPU tensors")
    if positions.requires_grad or cell.requires_grad:
        raise ValueError("exact graph preprocessing is non-differentiable")
    distances = _pbc_distance_matrix(
        positions.detach().numpy(),
        cell.detach().numpy(),
        pbc.detach().cpu().numpy(),
    )
    return torch.as_tensor(distances, dtype=positions.dtype)
