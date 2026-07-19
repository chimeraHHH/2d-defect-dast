"""SchNet comparator using the same precomputed periodic graph as DART."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock


class PBCSchNet(nn.Module):
    """A standard SchNet stack with explicit periodic edge distances."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        readout: str = "add",
        max_atomic_number: int = 100,
    ) -> None:
        super().__init__()
        if readout not in ("add", "mean"):
            raise ValueError("readout must be 'add' or 'mean'")
        self.readout = readout
        self.embedding = nn.Embedding(max_atomic_number + 1, hidden_channels, padding_idx=0)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
                for _ in range(num_interactions)
            ]
        )
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.activation = nn.SiLU()
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    @staticmethod
    def flatten_edges(
        edge_index_list: List[torch.Tensor],
        edge_dist_list: List[torch.Tensor],
        num_atoms_list: List[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = []
        total = 0
        for count in num_atoms_list:
            offsets.append(total)
            total += int(count)
        edge_indices = [
            edge_index.to(device=device, dtype=torch.long) + offsets[i]
            for i, edge_index in enumerate(edge_index_list)
        ]
        edge_distances = [
            edge_dist.to(device=device, dtype=torch.float32)
            for edge_dist in edge_dist_list
        ]
        return torch.cat(edge_indices, dim=1), torch.cat(edge_distances, dim=0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        atom_mask = batch["atom_mask"]
        atomic_numbers = batch["atomic_numbers"][atom_mask]
        x = self.embedding(atomic_numbers)
        edge_index, edge_weight = self.flatten_edges(
            batch["edge_index_list"], batch["edge_dist_list"],
            batch["num_atoms_list"], x.device,
        )
        edge_attr = self.distance_expansion(edge_weight)
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)
        atom_output = self.lin2(self.activation(self.lin1(x))).squeeze(-1)

        graph_index = torch.repeat_interleave(
            torch.arange(len(batch["num_atoms_list"]), device=x.device),
            torch.as_tensor(batch["num_atoms_list"], device=x.device),
        )
        output = torch.zeros(
            len(batch["num_atoms_list"]), dtype=atom_output.dtype, device=x.device
        )
        output.index_add_(0, graph_index, atom_output)
        if self.readout == "mean":
            counts = torch.as_tensor(
                batch["num_atoms_list"], dtype=output.dtype, device=x.device
            )
            output = output / counts.clamp_min(1.0)
        return output
