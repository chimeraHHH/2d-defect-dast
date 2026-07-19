from __future__ import annotations

import torch

from src.models.schnet_pbc import PBCSchNet


def _single_graph(atomic_numbers, edge_index, edge_dist):
    n_atoms = len(atomic_numbers)
    return {
        "atomic_numbers": torch.tensor([atomic_numbers]),
        "atom_mask": torch.ones((1, n_atoms), dtype=torch.bool),
        "edge_index_list": [torch.tensor(edge_index)],
        "edge_dist_list": [torch.tensor(edge_dist)],
        "num_atoms_list": [n_atoms],
    }


def test_pbc_schnet_forward_shape():
    model = PBCSchNet(hidden_channels=32, num_filters=32, num_interactions=2)
    batch = {
        "atomic_numbers": torch.tensor([[6, 8, 0], [14, 1, 1]]),
        "atom_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "edge_index_list": [
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]),
        ],
        "edge_dist_list": [torch.tensor([1.2, 1.2]), torch.tensor([1.5, 1.5, 1.6, 1.6])],
        "num_atoms_list": [2, 3],
    }
    output = model(batch)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_flatten_edges_offsets_each_graph():
    edge_index, edge_dist = PBCSchNet.flatten_edges(
        [
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([[0, 2], [2, 0]]),
        ],
        [torch.tensor([1.2, 1.2]), torch.tensor([1.6, 1.6])],
        [2, 3],
        torch.device("cpu"),
    )

    assert torch.equal(
        edge_index,
        torch.tensor([[0, 1, 2, 4], [1, 0, 4, 2]]),
    )
    assert torch.equal(edge_dist, torch.tensor([1.2, 1.2, 1.6, 1.6]))


def test_batched_prediction_matches_individual_graphs():
    torch.manual_seed(7)
    model = PBCSchNet(
        hidden_channels=32, num_filters=32, num_interactions=2
    ).eval()
    first = _single_graph(
        [6, 8], [[0, 1], [1, 0]], [1.2, 1.2]
    )
    second = _single_graph(
        [14, 1, 1],
        [[0, 1, 0, 2], [1, 0, 2, 0]],
        [1.5, 1.5, 1.6, 1.6],
    )
    batch = {
        "atomic_numbers": torch.tensor([[6, 8, 0], [14, 1, 1]]),
        "atom_mask": torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        "edge_index_list": [
            first["edge_index_list"][0], second["edge_index_list"][0]
        ],
        "edge_dist_list": [
            first["edge_dist_list"][0], second["edge_dist_list"][0]
        ],
        "num_atoms_list": [2, 3],
    }

    with torch.no_grad():
        batched = model(batch)
        separate = torch.cat([model(first), model(second)])

    torch.testing.assert_close(batched, separate, rtol=1e-6, atol=1e-6)
