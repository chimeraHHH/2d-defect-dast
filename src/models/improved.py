"""Defect-aware Star-Sparse Transformer (DAST).

Key changes vs ``CrystalTransformer``:

1. **Defect-aware token**: a learnable virtual node is appended to every graph
   to act as the explicit "defect anchor" called for in the project mid-term
   report (section 3.1).
2. **Star-sparse attention**: real atoms attend to a sparse neighbour set
   (within ``r_global`` and/or top-k by distance) plus the virtual node, while
   the virtual node attends bi-directionally with all atoms. This realises the
   "star-sparse" topology of section 4.7.1.
3. **Lattice-aware encoding**: lattice vector lengths are featurised and
   injected as a per-atom additive shift, exposing periodic geometry that the
   baseline distance matrix does not see directly (section 2.3.2).
4. **Defect-edge bias**: edges that touch a defect atom get an extra learnable
   bias added to attention scores so the network can amplify defect-centric
   interactions even when the edge is far in real space.

Implementation choice
---------------------
We use *dense* attention with a *sparsity mask* rather than a true scatter-
based sparse implementation. This keeps the operator graph MPS-friendly
(MPS lacks robust support for ``scatter_reduce``-style amax) while still
realising the star-sparse pattern: positions where the mask is ``False`` are
``-inf`` before softmax. For supercells of ~100 atoms the dense matrix is
trivial, so this is a pragmatic choice without sacrificing the architectural
contribution.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import LocalInteractionLayer, RBFExpansion


class StarSparseAttention(nn.Module):
    """Multi-head attention over (real atoms + 1 virtual defect token).

    Receives the padded sequence ``h`` of shape ``(B, N+1, C)`` and a binary
    sparsity mask of shape ``(B, N+1, N+1)``; positions where the mask is
    ``False`` are blocked before softmax.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        n_rbf_dist: int = 32,
        dmax: float = 12.0,
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
        self.dist_bias = nn.Sequential(
            nn.Linear(n_rbf_dist, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.defect_bias = nn.Parameter(torch.zeros(num_heads))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        mask_pair: torch.Tensor,
        dist_pair: torch.Tensor,
        defect_pair: torch.Tensor,
    ) -> torch.Tensor:
        b, n, c = h.shape
        head = self.num_heads
        d = self.head_dim
        qkv = self.qkv(h).reshape(b, n, 3, head, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, d)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        rbf = self.dist_rbf(dist_pair)
        bias = self.dist_bias(rbf).permute(0, 3, 1, 2)
        scores = scores + bias
        scores = scores + defect_pair.unsqueeze(1) * self.defect_bias.view(1, head, 1, 1)

        # Use -1e9 instead of -inf: MPS softmax can produce NaN with -inf masks
        scores = scores.masked_fill(~mask_pair.unsqueeze(1), -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.out_proj(out)


class StarSparseBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        n_rbf_dist: int,
        dmax: float,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = StarSparseAttention(hidden_dim, num_heads, n_rbf_dist, dmax, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, mask_pair, dist_pair, defect_pair):
        h = h + self.dropout(self.attn(self.norm1(h), mask_pair, dist_pair, defect_pair))
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return h


class DefectAwareTransformer(nn.Module):
    """Defect-Aware Star-Sparse Transformer (DAST)."""

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
        r_global: float = 8.0,
        k_global: int = 16,
        dmax_global: float = 12.0,
        defect_embedding: bool = True,
        dropout: float = 0.0,
        use_lattice_self_loop: bool = True,
        use_virtual_node: bool = True,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(atom_fea_len, hidden_dim)
        self.defect_embedding = (
            nn.Embedding(2, hidden_dim) if defect_embedding else None
        )
        self.use_virtual_node = use_virtual_node
        if use_virtual_node:
            self.virtual_token = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.normal_(self.virtual_token, std=0.02)

        self.edge_rbf = RBFExpansion(0.0, rcut_local, n_rbf_edge)
        self.local_layers = nn.ModuleList(
            [LocalInteractionLayer(hidden_dim, n_rbf_edge=n_rbf_edge) for _ in range(n_local_layers)]
        )
        self.global_layers = nn.ModuleList(
            [
                StarSparseBlock(
                    hidden_dim,
                    num_heads=num_heads,
                    n_rbf_dist=n_rbf_dist,
                    dmax=dmax_global,
                    dropout=dropout,
                )
                for _ in range(n_global_layers)
            ]
        )
        self.r_global = r_global
        self.k_global = k_global
        self.use_lattice_self_loop = use_lattice_self_loop
        if use_lattice_self_loop:
            self.lattice_self_mlp = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    # ---------------------------------------------------------------- helpers
    def _flatten_local_edges(
        self,
        num_atoms_list,
        edge_index_list,
        edge_dist_list,
        triplet_index_list,
        angles_list,
        device,
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

    def _build_attn_mask(
        self,
        num_atoms_list: List[int],
        dist_matrix: torch.Tensor,
        defect_mask: torch.Tensor,
        device: torch.device,
    ):
        """Build (B, N+1, N+1) padded sparsity mask, dist matrix, defect flag.

        Every row gets a self-loop so softmax never sees an all-masked row
        (which would underflow to NaN on MPS).
        """
        b, n_max, _ = dist_matrix.shape
        n_pad = n_max + 1 if self.use_virtual_node else n_max
        mask_pair = torch.zeros(b, n_pad, n_pad, dtype=torch.bool, device=device)
        # default: self-loop for every row (even padding) to keep softmax stable
        eye = torch.eye(n_pad, dtype=torch.bool, device=device)
        mask_pair[:] = eye.unsqueeze(0).expand(b, -1, -1)

        dist_pair = torch.zeros(b, n_pad, n_pad, dtype=dist_matrix.dtype, device=device)
        defect_pair = torch.zeros(b, n_pad, n_pad, dtype=dist_matrix.dtype, device=device)

        for i, n_i in enumerate(num_atoms_list):
            dm = dist_matrix[i, :n_i, :n_i]
            allow = dm <= self.r_global
            allow.fill_diagonal_(False)
            if self.k_global > 0 and n_i > self.k_global + 1:
                fill = torch.eye(n_i, device=device, dtype=dm.dtype) * 1e6
                _, topk = torch.topk(dm + fill, self.k_global, largest=False, dim=-1)
                knn = torch.zeros_like(allow)
                knn.scatter_(1, topk, True)
                allow = allow & knn
            mask_pair[i, :n_i, :n_i] = allow | torch.eye(n_i, dtype=torch.bool, device=device)
            dist_pair[i, :n_i, :n_i] = dm
            df = defect_mask[i, :n_i].float()
            defect_pair[i, :n_i, :n_i] = (df.unsqueeze(0) + df.unsqueeze(1)).clamp(max=1.0)

            if self.use_virtual_node:
                v_idx = n_i  # virtual token sits at index n_i
                # virtual <-> all real atoms (and self)
                mask_pair[i, v_idx, : n_i + 1] = True
                mask_pair[i, : n_i + 1, v_idx] = True
                dist_pair[i, v_idx, :n_i] = 0.0
                dist_pair[i, :n_i, v_idx] = 0.0
                defect_pair[i, v_idx, : n_i + 1] = 1.0
                defect_pair[i, : n_i + 1, v_idx] = 1.0

        return mask_pair, dist_pair, defect_pair

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        atom_mask = batch["atom_mask"]
        dist_matrix = batch["dist_matrix"]
        defect_mask = batch.get("defect_mask")
        cell = batch.get("cell")
        device = x.device
        b, n_max, _ = x.shape
        num_atoms_list = batch["num_atoms_list"]

        # ---- node embedding ----
        h = self.embed(x)
        if self.defect_embedding is not None and defect_mask is not None:
            h = h + self.defect_embedding(defect_mask)

        # ---- local pass on flattened sparse graph (MPS-safe gather) ----
        flat_indices = []
        for i, n_i in enumerate(num_atoms_list):
            base = i * n_max
            flat_indices.append(torch.arange(n_i, device=device, dtype=torch.long) + base)
        flat_indices_t = (
            torch.cat(flat_indices) if flat_indices else torch.empty(0, dtype=torch.long, device=device)
        )
        h_flat_full = h.reshape(b * n_max, h.shape[-1])
        flat_h_real = h_flat_full.index_select(0, flat_indices_t)

        edge_index, edge_dist, triplet_index, angles = self._flatten_local_edges(
            num_atoms_list,
            batch["edge_index_list"],
            batch["edge_dist_list"],
            batch["triplet_index_list"],
            batch["angles_list"],
            device=device,
        )
        edge_attr_rbf = self.edge_rbf(edge_dist)
        for layer in self.local_layers:
            flat_h_real = layer(flat_h_real, edge_index, edge_attr_rbf, triplet_index, angles)

        # optional lattice geometry per atom
        if self.use_lattice_self_loop and cell is not None:
            lat_norms = cell.norm(dim=-1)
            mu = lat_norms.mean(dim=-1, keepdim=True)
            std = lat_norms.std(dim=-1, keepdim=True) + 1e-6
            lat_feat = (lat_norms - mu) / std
            lat_emb = self.lattice_self_mlp(lat_feat)
            atom_lat = torch.cat(
                [lat_emb[i].unsqueeze(0).expand(num_atoms_list[i], -1) for i in range(b)], dim=0
            )
            flat_h_real = flat_h_real + atom_lat

        # scatter back to padded layout
        h_pad = torch.zeros(b * n_max, h.shape[-1], dtype=h.dtype, device=device)
        h_pad.index_copy_(0, flat_indices_t, flat_h_real)
        h_pad = h_pad.reshape(b, n_max, h.shape[-1])

        # ---- assemble (B, N+1, C) sequence with virtual defect token ----
        if self.use_virtual_node:
            virt = self.virtual_token.unsqueeze(0).unsqueeze(0).expand(b, 1, -1)
            h_seq = torch.cat([h_pad, virt], dim=1)  # (B, N+1, C)
        else:
            h_seq = h_pad

        mask_pair, dist_pair, defect_pair = self._build_attn_mask(
            num_atoms_list,
            dist_matrix,
            defect_mask if defect_mask is not None else torch.zeros_like(atom_mask, dtype=torch.long),
            device=device,
        )

        for layer in self.global_layers:
            h_seq = layer(h_seq, mask_pair, dist_pair, defect_pair)

        # ---- pool ----
        if self.use_virtual_node:
            v_features = h_seq[:, n_max]  # (B, C)
            atoms_feat = h_seq[:, :n_max]
        else:
            v_features = None
            atoms_feat = h_seq

        atom_mask_f = atom_mask.float().unsqueeze(-1)
        atom_pool = (atoms_feat * atom_mask_f).sum(dim=1) / atom_mask_f.sum(dim=1).clamp(min=1.0)

        if v_features is not None:
            graph_repr = 0.5 * (v_features + atom_pool)
        else:
            graph_repr = atom_pool
        return self.readout(graph_repr).squeeze(-1)
