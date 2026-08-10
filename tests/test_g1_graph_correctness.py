"""G1 graph-correctness property tests.

These are scientific/numerical acceptance tests and are executed only on the
GPU server under the frozen G1 command.  Local development is limited to
syntax/static checks.
"""
from __future__ import annotations

import itertools
import unittest

import numpy as np
import torch
from ase import Atoms

from src.dataset import collate_fn
from src.graph import (
    MAX_TRIPLETS_PER_CENTRE,
    _invariant_triplets,
    _midpoint_order_statistic_ranks,
    _pbc_distance_matrix,
    build_graph,
    torch_pbc_distance_matrix,
)
from src.models.crystal_v2 import (
    ENV_ZERO_NEIGHBOR_CORRECTED,
    ENV_ZERO_NEIGHBOR_LEGACY,
    CrystalTransformerV2,
    LocalEnvEnrichment,
    LocalEnvEnrichmentV2,
    LocalInteractionLayerV2,
)


def _componentwise_distance_matrix(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    frac = diff @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return np.linalg.norm(frac @ cell, axis=-1)


def _brute_force_distance_matrix(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    radius: int = 12,
) -> np.ndarray:
    """Independent enumeration with an explicit non-boundary oracle check."""
    ranges = [range(-radius, radius + 1) if periodic else (0,) for periodic in pbc]
    integer_shifts = np.asarray(list(itertools.product(*ranges)), dtype=int)
    shifts = integer_shifts.astype(float) @ cell
    diff = positions[:, None, :] - positions[None, :, :]
    candidates = diff[..., None, :] - shifts
    norms = np.linalg.norm(candidates, axis=-1)
    argmin = norms.argmin(axis=-1)
    winning_shifts = integer_shifts[argmin]
    for axis, periodic in enumerate(pbc):
        if periodic and np.any(np.abs(winning_shifts[..., axis]) == radius):
            raise AssertionError(
                "independent MIC enumeration reached its search boundary"
            )
    return norms.min(axis=-1)


def _angles_by_old_centre(graph: dict, new_to_old: np.ndarray) -> dict[int, np.ndarray]:
    output = {}
    centres = graph["triplet_index"][:, 1]
    for centre in np.unique(centres):
        old_centre = int(new_to_old[int(centre)])
        output[old_centre] = np.sort(graph["angles"][centres == centre])
    return output


def _edge_records(graph: dict, new_to_old: np.ndarray) -> list[tuple]:
    records = []
    for edge, distance in zip(graph["edge_index"].T, graph["edge_dist"]):
        records.append(
            (
                int(new_to_old[int(edge[0])]),
                int(new_to_old[int(edge[1])]),
                round(float(distance), 6),
            )
        )
    return sorted(records)


def _graph_item(graph: dict, sample_index: int = 0) -> dict:
    numbers = torch.from_numpy(graph["numbers"]).long()
    z = numbers.float()
    features = torch.stack(
        [
            z / 100.0,
            torch.sin(z),
            torch.cos(z),
            (z % 2) / 2.0,
            (z % 3) / 3.0,
            (z % 5) / 5.0,
            (z % 7) / 7.0,
            torch.sqrt(z) / 10.0,
            torch.log1p(z) / 10.0,
        ],
        dim=-1,
    )
    defect_mask = torch.zeros(len(numbers), dtype=torch.long)
    defect_mask[0] = 1
    return {
        "sample_index": torch.tensor(sample_index, dtype=torch.long),
        "x": features,
        "atomic_numbers": numbers,
        "defect_mask": defect_mask,
        "defect_type": torch.tensor(3, dtype=torch.long),
        "edge_index": torch.from_numpy(graph["edge_index"]),
        "edge_dist": torch.from_numpy(graph["edge_dist"]),
        "edge_offset": torch.from_numpy(graph["edge_offset"]),
        "triplet_index": torch.from_numpy(graph["triplet_index"]),
        "angles": torch.from_numpy(graph["angles"]),
        "dist_matrix": torch.from_numpy(graph["dist_matrix"]),
        "positions": torch.from_numpy(graph["positions"]),
        "cell": torch.from_numpy(graph["cell"]),
        "target": torch.tensor(0.0),
        "num_atoms": len(numbers),
    }


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _assert_midpoint_rank_contract_has_no_duplicates_or_endpoint_bias() -> None:
    for n_values in range(MAX_TRIPLETS_PER_CENTRE + 1, 400):
        ranks = _midpoint_order_statistic_ranks(
            n_values, MAX_TRIPLETS_PER_CENTRE
        )
        assert len(ranks) == MAX_TRIPLETS_PER_CENTRE
        assert np.all(np.diff(ranks) > 0)
        assert 0 <= ranks[0] < ranks[-1] < n_values
        expected = (
            (2 * np.arange(MAX_TRIPLETS_PER_CENTRE) + 1) * n_values
        ) // (2 * MAX_TRIPLETS_PER_CENTRE)
        np.testing.assert_array_equal(ranks, expected)


def _assert_exact_mic_fixes_adversarial_oblique_cell() -> None:
    cell = np.asarray(
        [[4.0, 0.0, 0.0], [3.9, 0.5, 0.0], [0.0, 0.0, 20.0]]
    )
    fractional = np.asarray([[0.0, 0.0, 0.0], [0.49, 0.49, 0.0]])
    positions = fractional @ cell
    pbc = np.asarray([True, True, False])
    exact = _pbc_distance_matrix(positions, cell, pbc)
    brute = _brute_force_distance_matrix(positions, cell, pbc)
    legacy = _componentwise_distance_matrix(positions, cell)
    np.testing.assert_allclose(exact, brute, atol=1.0e-12, rtol=0.0)
    assert legacy[0, 1] - exact[0, 1] > 3.0


def _assert_exact_mic_respects_nonperiodic_axis() -> None:
    cell = np.diag([5.0, 5.0, 3.0])
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
    slab = _pbc_distance_matrix(
        positions, cell, np.asarray([True, True, False])
    )
    bulk = _pbc_distance_matrix(
        positions, cell, np.asarray([True, True, True])
    )
    assert abs(float(slab[0, 1]) - 2.5) < 1.0e-12
    assert abs(float(bulk[0, 1]) - 0.5) < 1.0e-12


def _assert_exact_mic_matches_independent_enumeration(cell: np.ndarray) -> None:
    rng = np.random.default_rng(20260810)
    positions = rng.uniform(-1.5, 1.5, size=(9, 3)) @ cell
    pbc = np.asarray([True, True, True])
    exact = _pbc_distance_matrix(positions, cell, pbc)
    brute = _brute_force_distance_matrix(positions, cell, pbc, radius=12)
    np.testing.assert_allclose(exact, brute, atol=1.0e-10, rtol=0.0)


def _assert_invariant_triplet_multiset_under_symmetric_ties() -> None:
    n = 12
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    vectors = np.stack([2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros(n)], axis=1)
    centres = np.zeros(n, dtype=np.int64)
    neighbours = np.arange(1, n + 1, dtype=np.int64)
    triplets, angles = _invariant_triplets(centres, neighbours, vectors)
    assert len(angles) == MAX_TRIPLETS_PER_CENTRE
    assert np.all(triplets[:, 1] == 0)

    permutation = np.asarray([7, 2, 10, 0, 9, 1, 5, 11, 4, 6, 8, 3])
    perm_triplets, perm_angles = _invariant_triplets(
        centres[permutation], neighbours[permutation], vectors[permutation]
    )
    assert np.all(perm_triplets[:, 1] == 0)
    np.testing.assert_allclose(angles, perm_angles, atol=1.0e-7, rtol=0.0)


def _assert_graph_is_equivariant_to_atom_reindexing() -> None:
    cell = np.asarray(
        [[6.0, 0.0, 0.0], [2.6, 5.3, 0.0], [0.3, 0.2, 18.0]]
    )
    fractional = np.asarray(
        [
            [0.05, 0.10, 0.40],
            [0.23, 0.17, 0.41],
            [0.41, 0.28, 0.39],
            [0.62, 0.43, 0.42],
            [0.78, 0.64, 0.40],
            [0.31, 0.76, 0.38],
            [0.91, 0.89, 0.41],
        ]
    )
    positions = fractional @ cell
    numbers = np.asarray([26, 16, 16, 42, 16, 34, 16])
    pbc = np.asarray([True, True, False])
    original = build_graph(
        Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc),
        cutoff=6.2,
    )
    permutation = np.asarray([4, 0, 6, 2, 5, 1, 3])
    permuted = build_graph(
        Atoms(
            numbers=numbers[permutation],
            positions=positions[permutation],
            cell=cell,
            pbc=pbc,
        ),
        cutoff=6.2,
    )
    np.testing.assert_allclose(
        permuted["dist_matrix"],
        original["dist_matrix"][np.ix_(permutation, permutation)],
        atol=1.0e-6,
        rtol=0.0,
    )
    assert _edge_records(original, np.arange(len(numbers))) == _edge_records(
        permuted, permutation
    )
    original_angles = _angles_by_old_centre(original, np.arange(len(numbers)))
    permuted_angles = _angles_by_old_centre(permuted, permutation)
    assert original_angles.keys() == permuted_angles.keys()
    for centre in original_angles:
        np.testing.assert_allclose(
            original_angles[centre], permuted_angles[centre], atol=1.0e-6, rtol=0.0
        )


def _assert_graph_is_invariant_to_translation_and_single_atom_lattice_shift() -> None:
    cell = np.asarray(
        [[5.0, 0.0, 0.0], [1.7, 4.6, 0.0], [0.0, 0.0, 17.0]]
    )
    positions = np.asarray(
        [[0.2, 0.3, 7.0], [1.8, 0.8, 7.2], [3.1, 2.4, 6.9], [1.1, 3.7, 7.1]]
    )
    numbers = np.asarray([22, 16, 16, 34])
    pbc = np.asarray([True, True, False])
    reference = build_graph(Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc))
    shifted = positions + np.asarray([1.3, -2.1, 0.7])
    shifted[2] += cell[0] - 2.0 * cell[1]
    transformed = build_graph(
        Atoms(numbers=numbers, positions=shifted, cell=cell, pbc=pbc)
    )
    np.testing.assert_allclose(
        reference["dist_matrix"], transformed["dist_matrix"], atol=1.0e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        np.sort(reference["edge_dist"]),
        np.sort(transformed["edge_dist"]),
        atol=1.0e-6,
        rtol=0.0,
    )
    for centre in range(len(numbers)):
        ref_mask = reference["triplet_index"][:, 1] == centre
        out_mask = transformed["triplet_index"][:, 1] == centre
        np.testing.assert_allclose(
            np.sort(reference["angles"][ref_mask]),
            np.sort(transformed["angles"][out_mask]),
            atol=1.0e-6,
            rtol=0.0,
        )


def _assert_triplet_endpoint_identity_does_not_enter_local_operator() -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260810)
    layer = LocalInteractionLayerV2(16).to(device).eval()
    x = torch.randn(6, 16, device=device)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], device=device
    )
    edge_attr = torch.randn(edge_index.shape[1], 32, device=device)
    angles = torch.tensor([0.4, 1.1, 2.2], device=device)
    triplets_a = torch.tensor([[1, 0, 2], [0, 1, 2], [0, 2, 1]], device=device)
    triplets_b = torch.tensor([[5, 0, 4], [3, 1, 5], [4, 2, 3]], device=device)
    with torch.no_grad():
        out_a = layer(x, edge_index, edge_attr, triplets_a, angles)
        out_b = layer(x, edge_index, edge_attr, triplets_b, angles)
    torch.testing.assert_close(out_a, out_b, rtol=0.0, atol=0.0)


def _assert_v2_prediction_is_numerically_invariant_to_atom_reindexing() -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260811)
    cell = np.asarray(
        [[6.0, 0.0, 0.0], [2.7, 5.2, 0.0], [0.2, 0.1, 18.0]]
    )
    fractional = np.asarray(
        [
            [0.05, 0.10, 0.40], [0.23, 0.17, 0.41], [0.41, 0.28, 0.39],
            [0.62, 0.43, 0.42], [0.78, 0.64, 0.40], [0.31, 0.76, 0.38],
            [0.91, 0.89, 0.41],
        ]
    )
    positions = fractional @ cell
    numbers = np.asarray([26, 16, 16, 42, 16, 34, 16])
    pbc = np.asarray([True, True, False])
    permutation = np.asarray([4, 0, 6, 2, 5, 1, 3])
    graph_a = build_graph(Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc), cutoff=6.2)
    graph_b = build_graph(
        Atoms(numbers=numbers[permutation], positions=positions[permutation], cell=cell, pbc=pbc),
        cutoff=6.2,
    )
    item_a = _graph_item(graph_a, sample_index=0)
    item_b = _graph_item(graph_b, sample_index=0)
    # Preserve the same physical impurity after reindexing.
    item_b["defect_mask"].zero_()
    item_b["defect_mask"][int(np.flatnonzero(permutation == 0)[0])] = 1
    model = CrystalTransformerV2(
        atom_fea_len=9,
        hidden_dim=32,
        n_local_layers=2,
        n_global_layers=1,
        num_heads=4,
        dropout=0.0,
        use_gated_pooling=True,
        use_env_enrichment=True,
        use_prenorm_local=True,
    ).to(device).eval()
    with torch.no_grad():
        pred_a = model(_move_batch(collate_fn([item_a]), device))
        pred_b = model(_move_batch(collate_fn([item_b]), device))
    torch.testing.assert_close(pred_a, pred_b, rtol=0.0, atol=2.0e-5)


def _assert_zero_neighbor_enrichment_is_batch_independent() -> None:
    """Freeze corrected and legacy E-module semantics on the assigned GPU."""
    device = torch.device("cuda")
    corrected = LocalEnvEnrichment(
        hidden_dim=8,
        zero_neighbor_mode=ENV_ZERO_NEIGHBOR_CORRECTED,
    ).to(device).eval()
    legacy = LocalEnvEnrichment(
        hidden_dim=8,
        zero_neighbor_mode=ENV_ZERO_NEIGHBOR_LEGACY,
    ).to(device).eval()
    with torch.no_grad():
        for parameter in corrected.parameters():
            parameter.fill_(0.125)
        legacy.load_state_dict(corrected.state_dict())

        # Flat atom order is: isolated defect A; defect B; neighbour B.  The
        # only edge belongs to B, so A has zero local neighbours in a mixed
        # batch even though the batch contains a defect-centred edge.
        mixed_h = torch.zeros(2, 2, 8, device=device)
        mixed_defect = torch.tensor([[1, 0], [1, 0]], device=device)
        mixed_edges = torch.tensor([[1], [2]], device=device)
        mixed_dist = torch.tensor([2.0], device=device)
        mixed_flat_defect = torch.tensor([1, 1, 0], device=device)
        mixed_en = torch.tensor([2.0, 3.0, 1.0], device=device)
        mixed_flat_indices = torch.tensor([0, 2, 3], device=device)
        corrected_mixed = corrected(
            mixed_h,
            mixed_defect,
            mixed_edges,
            mixed_dist,
            mixed_flat_defect,
            mixed_en,
            mixed_flat_indices,
            [1, 2],
        )

        isolated_h = torch.zeros(1, 1, 8, device=device)
        corrected_isolated = corrected(
            isolated_h,
            torch.ones(1, 1, dtype=torch.long, device=device),
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, device=device),
            torch.ones(1, dtype=torch.long, device=device),
            torch.tensor([2.0], device=device),
            torch.tensor([0], device=device),
            [1],
        )
        torch.testing.assert_close(
            corrected_mixed[0, 0], corrected_isolated[0, 0], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            corrected_mixed[0, 0], mixed_h[0, 0], rtol=0.0, atol=0.0
        )
        assert not torch.equal(corrected_mixed[1, 0], mixed_h[1, 0])

        # The compatibility branch must reproduce the historical mixed-batch
        # formula exactly, including projection bias.
        legacy_mixed = legacy(
            mixed_h,
            mixed_defect,
            mixed_edges,
            mixed_dist,
            mixed_flat_defect,
            mixed_en,
            mixed_flat_indices,
            [1, 2],
        )
        legacy_feature_a = torch.tensor(
            [0.0, 0.0, 0.0, 2.0], device=device
        )
        legacy_expected_a = mixed_h[0, 0] + legacy.proj(legacy_feature_a)
        torch.testing.assert_close(
            legacy_mixed[0, 0], legacy_expected_a, rtol=0.0, atol=0.0
        )
        assert not torch.equal(legacy_mixed[0, 0], corrected_mixed[0, 0])
        legacy_isolated = legacy(
            isolated_h,
            torch.ones(1, 1, dtype=torch.long, device=device),
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, device=device),
            torch.ones(1, dtype=torch.long, device=device),
            torch.tensor([2.0], device=device),
            torch.tensor([0], device=device),
            [1],
        )
        torch.testing.assert_close(
            legacy_isolated[0, 0], isolated_h[0, 0], rtol=0.0, atol=0.0
        )

        # Freeze the same contract for the optional 11-feature E module.  Its
        # historical empty-neighbour vector includes the clamped min-distance
        # sentinel as well as the spurious electronegativity contrast.
        corrected_v2 = LocalEnvEnrichmentV2(
            hidden_dim=8,
            zero_neighbor_mode=ENV_ZERO_NEIGHBOR_CORRECTED,
        ).to(device).eval()
        legacy_v2 = LocalEnvEnrichmentV2(
            hidden_dim=8,
            zero_neighbor_mode=ENV_ZERO_NEIGHBOR_LEGACY,
        ).to(device).eval()
        for parameter in corrected_v2.parameters():
            parameter.fill_(0.125)
        legacy_v2.load_state_dict(corrected_v2.state_dict())
        corrected_v2_mixed = corrected_v2(
            mixed_h,
            mixed_defect,
            mixed_edges,
            mixed_dist,
            mixed_flat_defect,
            mixed_en,
            mixed_flat_indices,
            [1, 2],
        )
        corrected_v2_isolated = corrected_v2(
            isolated_h,
            torch.ones(1, 1, dtype=torch.long, device=device),
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, device=device),
            torch.ones(1, dtype=torch.long, device=device),
            torch.tensor([2.0], device=device),
            torch.tensor([0], device=device),
            [1],
        )
        torch.testing.assert_close(
            corrected_v2_mixed[0, 0],
            corrected_v2_isolated[0, 0],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            corrected_v2_mixed[0, 0], mixed_h[0, 0], rtol=0.0, atol=0.0
        )
        assert not torch.equal(corrected_v2_mixed[1, 0], mixed_h[1, 0])
        legacy_v2_mixed = legacy_v2(
            mixed_h,
            mixed_defect,
            mixed_edges,
            mixed_dist,
            mixed_flat_defect,
            mixed_en,
            mixed_flat_indices,
            [1, 2],
        )
        legacy_v2_feature_a = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0],
            device=device,
        )
        legacy_v2_expected_a = mixed_h[0, 0] + legacy_v2.proj(
            legacy_v2_feature_a
        )
        torch.testing.assert_close(
            legacy_v2_mixed[0, 0], legacy_v2_expected_a, rtol=0.0, atol=0.0
        )
        assert not torch.equal(
            legacy_v2_mixed[0, 0], corrected_v2_mixed[0, 0]
        )
        legacy_v2_isolated = legacy_v2(
            isolated_h,
            torch.ones(1, 1, dtype=torch.long, device=device),
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, device=device),
            torch.ones(1, dtype=torch.long, device=device),
            torch.tensor([2.0], device=device),
            torch.tensor([0], device=device),
            [1],
        )
        torch.testing.assert_close(
            legacy_v2_isolated[0, 0], isolated_h[0, 0], rtol=0.0, atol=0.0
        )


def _assert_torch_wrapper_respects_pbc_and_rejects_cuda() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
    cell = torch.diag(torch.tensor([5.0, 5.0, 3.0]))
    slab = torch_pbc_distance_matrix(
        positions, cell, torch.tensor([True, True, False])
    )
    bulk = torch_pbc_distance_matrix(
        positions, cell, torch.tensor([True, True, True])
    )
    torch.testing.assert_close(slab[0, 1], torch.tensor(2.5), rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(bulk[0, 1], torch.tensor(0.5), rtol=0.0, atol=1.0e-6)
    if torch.cuda.is_available():
        try:
            torch_pbc_distance_matrix(
                positions.cuda(), cell.cuda(), torch.tensor([True, True, True], device="cuda")
            )
        except ValueError as exc:
            assert "CPU tensors" in str(exc)
        else:
            raise AssertionError("CUDA preprocessing tensors were not rejected")


class G1GraphCorrectnessTests(unittest.TestCase):
    """stdlib test entry point for the frozen GPU-server acceptance command."""

    def test_midpoint_rank_contract(self) -> None:
        _assert_midpoint_rank_contract_has_no_duplicates_or_endpoint_bias()

    def test_adversarial_oblique_mic(self) -> None:
        _assert_exact_mic_fixes_adversarial_oblique_cell()

    def test_nonperiodic_axis(self) -> None:
        _assert_exact_mic_respects_nonperiodic_axis()

    def test_reference_enumeration_cells(self) -> None:
        cells = [
            np.diag([5.2, 6.1, 18.0]),
            np.asarray(
                [[4.0, 0.0, 0.0], [2.0, 3.464101615, 0.0], [0.0, 0.0, 20.0]]
            ),
            np.asarray(
                [[4.0, 0.0, 0.0], [3.9, 0.5, 0.0], [0.4, 0.2, 16.0]]
            ),
        ]
        for cell in cells:
            with self.subTest(cell=cell.tolist()):
                _assert_exact_mic_matches_independent_enumeration(cell)

    def test_symmetric_ties(self) -> None:
        _assert_invariant_triplet_multiset_under_symmetric_ties()

    def test_atom_reindexing(self) -> None:
        _assert_graph_is_equivariant_to_atom_reindexing()

    def test_translation_and_lattice_shift(self) -> None:
        _assert_graph_is_invariant_to_translation_and_single_atom_lattice_shift()

    def test_torch_wrapper_contract(self) -> None:
        _assert_torch_wrapper_respects_pbc_and_rejects_cuda()

    @unittest.skipUnless(torch.cuda.is_available(), "G1 local operator property requires CUDA")
    def test_cuda_triplet_endpoint_identity(self) -> None:
        _assert_triplet_endpoint_identity_does_not_enter_local_operator()

    @unittest.skipUnless(torch.cuda.is_available(), "G1 prediction property requires CUDA")
    def test_cuda_prediction_reindexing(self) -> None:
        _assert_v2_prediction_is_numerically_invariant_to_atom_reindexing()
        _assert_zero_neighbor_enrichment_is_batch_independent()


if __name__ == "__main__":
    unittest.main(verbosity=2)
