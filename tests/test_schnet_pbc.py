from __future__ import annotations

import torch

from src.models.schnet_pbc import PBCSchNet


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
