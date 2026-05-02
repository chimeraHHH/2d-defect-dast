"""Reproduction of the report's baseline ``CrystalTransformer``.

Architecture: Local interaction layers (bond + angle messaging on a sparse
neighbour graph) followed by Geometric self-attention layers (full transformer
with learnable distance bias).  This mirrors the design in the team's
mid-term report (Section 4.3) and the ``baseline_ref/src/model.py`` reference
implementation, restated here in cleaner form."""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFExpansion(nn.Module):
    """Gaussian radial basis expansion: maps scalar -> ``n_rbf``-vector."""

    def __init__(self, dmin: float = 0.0, dmax: float = 8.0, n_rbf: int = 32) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(dmin, dmax, n_rbf))
        self.sigma = (dmax - dmin) / max(n_rbf - 1, 1)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((d.unsqueeze(-1) - self.centers) ** 2) / (self.sigma ** 2))


class LocalInteractionLayer(nn.Module):
    """SchNet-style continuous-filter convolution + angle modulation.

    For every edge (i, j), the neighbour ``j`` sends a "filtered" version of
    its own features to centre ``i``: ``m_ij = phi(d_ij) * W_filter h_j``,
    where ``phi`` is an MLP over the RBF-expanded distance. Triplet messages
    capture three-body geometry and are aggregated at centre ``i``.
    """

    def __init__(self, hidden_dim: int, n_rbf_edge: int = 32, n_rbf_angle: int = 32) -> None:
        super().__init__()
        self.filter_mlp = nn.Sequential(
            nn.Linear(n_rbf_edge, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.angle_rbf = RBFExpansion(0.0, math.pi, n_rbf_angle)
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_dim + n_rbf_angle, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_rbf: torch.Tensor,
        triplet_index: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index  # row = centre i, col = neighbour j
        # Message: filter on distance gates the value of the neighbour.
        filt = self.filter_mlp(edge_attr_rbf)
        v_neigh = self.value_mlp(x[col])
        edge_messages = filt * v_neigh

        if triplet_index.numel() > 0:
            angle_rbf = self.angle_rbf(angles)
            centres = triplet_index[:, 1]
            triplet_messages = self.triplet_mlp(torch.cat([x[centres], angle_rbf], dim=-1))
        else:
            triplet_messages = torch.zeros((0, x.shape[-1]), dtype=x.dtype, device=x.device)
            centres = torch.empty((0,), dtype=torch.long, device=x.device)

        aggr = torch.zeros_like(x)
        aggr.index_add_(0, row, edge_messages)
        if triplet_messages.numel() > 0:
            aggr.index_add_(0, centres, triplet_messages)

        update = self.node_mlp(aggr)
        return self.norm(x + update)


class GeometricTransformerBlock(nn.Module):
    """Multi-head self-attention with learnable RBF-based distance bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        n_rbf_dist: int = 32,
        dmax: float = 12.0,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dist_rbf = RBFExpansion(0.0, dmax, n_rbf_dist)
        self.bias_mlp = nn.Sequential(
            nn.Linear(n_rbf_dist, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        dist_matrix: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, c = x.shape
        h = self.num_heads
        d = self.head_dim
        residual = x
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(b, n, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

        rbf = self.dist_rbf(dist_matrix)
        bias = self.bias_mlp(rbf).permute(0, 3, 1, 2)
        scores = scores + bias

        if mask is not None:
            # MPS softmax can produce NaN with -inf; use a large finite negative.
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.dropout(self.out_proj(out))

        x = residual + out
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CrystalTransformer(nn.Module):
    """Local + Global hybrid model from the project mid-term report."""

    def __init__(
        self,
        atom_fea_len: int = 9,
        hidden_dim: int = 128,
        n_local_layers: int = 3,
        n_global_layers: int = 2,
        num_heads: int = 4,
        n_rbf_edge: int = 32,
        n_rbf_dist: int = 32,
        rcut_local: float = 5.0,
        dmax_global: float = 12.0,
        defect_embedding: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(atom_fea_len, hidden_dim)
        self.defect_embedding = (
            nn.Embedding(2, hidden_dim) if defect_embedding else None
        )
        self.edge_rbf = RBFExpansion(0.0, rcut_local, n_rbf_edge)
        self.local_layers = nn.ModuleList(
            [LocalInteractionLayer(hidden_dim, n_rbf_edge=n_rbf_edge) for _ in range(n_local_layers)]
        )
        self.global_layers = nn.ModuleList(
            [
                GeometricTransformerBlock(
                    hidden_dim,
                    num_heads=num_heads,
                    n_rbf_dist=n_rbf_dist,
                    dmax=dmax_global,
                    dropout=dropout,
                )
                for _ in range(n_global_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _flatten_edges(
        self,
        num_atoms_list: List[int],
        edge_index_list: List[torch.Tensor],
        edge_dist_list: List[torch.Tensor],
        triplet_index_list: List[torch.Tensor],
        angles_list: List[torch.Tensor],
        device: torch.device,
    ):
        offsets = [0]
        for n in num_atoms_list[:-1]:
            offsets.append(offsets[-1] + n)
        all_edges, all_dist = [], []
        all_triplets, all_angles = [], []
        for i, off in enumerate(offsets):
            ei = edge_index_list[i].to(device)
            ed = edge_dist_list[i].to(device)
            ti = triplet_index_list[i].to(device)
            ag = angles_list[i].to(device)
            if ei.numel() > 0:
                all_edges.append(ei + off)
                all_dist.append(ed)
            if ti.numel() > 0:
                all_triplets.append(ti + off)
                all_angles.append(ag)
        edge_index = (
            torch.cat(all_edges, dim=1) if all_edges else torch.empty(2, 0, dtype=torch.long, device=device)
        )
        edge_dist = (
            torch.cat(all_dist, dim=0) if all_dist else torch.empty(0, dtype=torch.float32, device=device)
        )
        triplet_index = (
            torch.cat(all_triplets, dim=0) if all_triplets else torch.empty(0, 3, dtype=torch.long, device=device)
        )
        angles = (
            torch.cat(all_angles, dim=0) if all_angles else torch.empty(0, dtype=torch.float32, device=device)
        )
        return edge_index, edge_dist, triplet_index, angles

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        mask = batch["atom_mask"]
        dist_matrix = batch["dist_matrix"]
        defect_mask = batch.get("defect_mask")
        device = x.device

        h = self.embed(x)
        if self.defect_embedding is not None and defect_mask is not None:
            h = h + self.defect_embedding(defect_mask)

        # local path: gather valid atoms via flat indices (MPS-safe)
        b, n_max, c = h.shape
        num_atoms_list = batch["num_atoms_list"]
        flat_indices = []
        for i, n_i in enumerate(num_atoms_list):
            base = i * n_max
            flat_indices.append(torch.arange(n_i, device=device, dtype=torch.long) + base)
        flat_indices = torch.cat(flat_indices) if flat_indices else torch.empty(0, dtype=torch.long, device=device)
        h_flat_full = h.reshape(b * n_max, c)
        flat_h = h_flat_full.index_select(0, flat_indices)

        edge_index, edge_dist, triplet_index, angles = self._flatten_edges(
            num_atoms_list,
            batch["edge_index_list"],
            batch["edge_dist_list"],
            batch["triplet_index_list"],
            batch["angles_list"],
            device=device,
        )
        edge_attr_rbf = self.edge_rbf(edge_dist)
        for layer in self.local_layers:
            flat_h = layer(flat_h, edge_index, edge_attr_rbf, triplet_index, angles)
        # scatter back to padded layout
        h_local_flat = torch.zeros(b * n_max, c, dtype=h.dtype, device=device)
        h_local_flat.index_copy_(0, flat_indices, flat_h)
        h_local = h_local_flat.reshape(b, n_max, c)

        # global path
        h_global = h_local
        for layer in self.global_layers:
            h_global = layer(h_global, dist_matrix, mask)

        mask_f = mask.float().unsqueeze(-1)
        pooled = (h_global * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.readout(pooled).squeeze(-1)
