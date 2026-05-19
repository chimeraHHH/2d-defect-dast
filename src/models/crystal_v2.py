"""CrystalTransformerV2 — Architecture improvements for top-journal submission.

Key changes over CrystalTransformer (baseline.py):
  1. **Defect-Aware Gated Attention Pooling**: Replaces mean pooling with a
     learnable attention mechanism that naturally focuses on defect-relevant
     atoms, plus a max-pooling branch. Motivated by [Hossain et al., Chem.
     Mater. 2024] showing global max pooling reduces MAE by 55% for defect
     formation energies — defect atoms produce extreme features that mean
     pooling dilutes.

  2. **Local Environment Enrichment**: Instead of a binary defect mask, we
     compute physics-aware features for the defect site: coordination number,
     mean neighbour distance, electronegativity contrast with host atoms,
     and local strain indicator. These cost negligible FLOPs and provide
     the model with explicit chemical-environment information.

  3. **Pre-Norm Residual in Local Layers**: Matches modern Transformer
     best practices (more stable training, compatible with larger LR).

  4. **Edge Feature Gating**: Learnable gating mechanism on edge messages
     that modulates information flow based on edge distance — improves
     expressivity of message passing without adding full edge updates.

All changes preserve compatibility with existing checkpoints (new modules
are additive; the old forward path is recoverable by toggling flags).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import RBFExpansion


# ---------------------------------------------------------------------------
# Local Interaction Layer V2: Pre-norm + edge gating
# ---------------------------------------------------------------------------
class LocalInteractionLayerV2(nn.Module):
    """SchNet-style message passing with pre-norm residual and edge gating.

    Differences from V1:
      - Pre-norm (LayerNorm before the update, not after) — more stable for
        larger learning rates and deeper stacks.
      - Edge gate: a scalar sigmoid gate per edge, learned from the RBF-
        expanded distance, that modulates the message strength.  This lets
        the model learn to suppress long-range noise without hard cutoffs.
    """

    def __init__(self, hidden_dim: int, n_rbf_edge: int = 32,
                 n_rbf_angle: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.filter_mlp = nn.Sequential(
            nn.Linear(n_rbf_edge, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_mlp = nn.Linear(hidden_dim, hidden_dim)

        # Edge gate: distance → scalar gate in [0, 1]
        self.edge_gate = nn.Sequential(
            nn.Linear(n_rbf_edge, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_rbf: torch.Tensor,
        triplet_index: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.norm(x)  # Pre-norm
        row, col = edge_index

        filt = self.filter_mlp(edge_attr_rbf)
        gate = self.edge_gate(edge_attr_rbf)          # (E, 1)
        v_neigh = self.value_mlp(x_norm[col])
        edge_messages = filt * v_neigh * gate           # gated messages

        if triplet_index.numel() > 0:
            angle_rbf = self.angle_rbf(angles)
            centres = triplet_index[:, 1]
            triplet_messages = self.triplet_mlp(
                torch.cat([x_norm[centres], angle_rbf], dim=-1)
            )
        else:
            triplet_messages = torch.zeros(
                (0, x.shape[-1]), dtype=x.dtype, device=x.device
            )
            centres = torch.empty((0,), dtype=torch.long, device=x.device)

        aggr = torch.zeros_like(x)
        aggr.index_add_(0, row, edge_messages)
        if triplet_messages.numel() > 0:
            aggr.index_add_(0, centres, triplet_messages)

        update = self.node_mlp(aggr)
        return x + update  # Pre-norm residual (no LayerNorm after)


# ---------------------------------------------------------------------------
# Defect-Aware Gated Attention Pooling
# ---------------------------------------------------------------------------
class DefectAwarePooling(nn.Module):
    """Attention-weighted + max-pool readout with defect-site gating.

    Mean pooling dilutes the defect atom's extreme features among dozens of
    host atoms.  This module learns an attention weight per atom that naturally
    upweights the defect site, and fuses the attention-weighted representation
    with a max-pooled branch to capture the most extreme per-feature values.

    Architecture:
        h_attn = softmax(MLP(h)) · h          # attention-weighted sum
        h_max  = max_pool(h, mask)             # channel-wise max
        h_out  = gate · h_attn + (1 - gate) · h_max
    where gate = sigmoid(W · [h_attn || h_max]).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Gating network: decides how much to blend attn vs max
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, N, C) atom representations after global layers
            mask: (B, N) bool, True for valid atoms
        Returns:
            (B, C) graph-level representation
        """
        mask_f = mask.float().unsqueeze(-1)  # (B, N, 1)

        # --- Attention-weighted sum ---
        attn_logits = self.attn_mlp(h).squeeze(-1)           # (B, N)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)         # (B, N)
        h_attn = (attn_weights.unsqueeze(-1) * h).sum(dim=1)  # (B, C)

        # --- Max pooling ---
        h_masked = h.masked_fill(~mask.unsqueeze(-1), -1e9)
        h_max = h_masked.max(dim=1).values                    # (B, C)

        # --- Gated fusion ---
        gate = self.gate_net(torch.cat([h_attn, h_max], dim=-1))  # (B, 1)
        h_out = gate * h_attn + (1 - gate) * h_max

        return self.norm(h_out)


# ---------------------------------------------------------------------------
# Local Environment Enrichment
# ---------------------------------------------------------------------------
class LocalEnvEnrichment(nn.Module):
    """Compute physics-aware features for the defect site at runtime.

    Features per defect atom (broadcast to embedding space):
      - Coordination number (count of edges from defect atom)
      - Mean neighbour distance
      - Max neighbour distance (captures strain)
      - Electronegativity contrast: |EN_defect - mean(EN_host_neighbours)|

    These are projected to hidden_dim and added to the defect atom's embedding.
    For non-defect atoms, the contribution is zero.
    """

    def __init__(self, hidden_dim: int, n_env_features: int = 4) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_env_features, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        defect_mask: torch.Tensor,
        edge_index_list: list,
        edge_dist_list: list,
        num_atoms_list: list,
        x_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Add local environment features to defect atoms.

        Args:
            h: (B, N_max, C) current embeddings
            defect_mask: (B, N_max) int, 1 for defect atoms
            edge_index_list: per-sample edge indices
            edge_dist_list: per-sample edge distances
            num_atoms_list: atoms per sample
            x_raw: (B, N_max, atom_fea_len) raw atom features
                   (feature index 2 = electronegativity)
        Returns:
            h: (B, N_max, C) with enriched defect embeddings
        """
        device = h.device
        B, N_max, C = h.shape
        env_features = torch.zeros(B, N_max, 4, device=device, dtype=h.dtype)

        offset = 0
        for b in range(B):
            n_atoms = num_atoms_list[b]
            dm = defect_mask[b, :n_atoms]  # (n_atoms,)
            defect_indices = dm.nonzero(as_tuple=True)[0]

            if len(defect_indices) == 0 or len(edge_index_list) <= b:
                offset += n_atoms
                continue

            ei = edge_index_list[b].to(device)
            ed = edge_dist_list[b].to(device)
            en_raw = x_raw[b, :n_atoms, 2]  # electronegativity (feature idx 2)

            for di in defect_indices:
                di_val = di.item()
                # Edges where this atom is the center (row)
                edge_mask = (ei[0] == di_val)
                if edge_mask.sum() == 0:
                    continue

                neigh_dists = ed[edge_mask]
                neigh_ids = ei[1, edge_mask]

                coord_num = float(edge_mask.sum())
                mean_dist = neigh_dists.mean()
                max_dist = neigh_dists.max()

                # Electronegativity contrast
                en_defect = en_raw[di_val]
                en_neighs = en_raw[neigh_ids.long()]
                en_contrast = (en_defect - en_neighs.mean()).abs()

                env_features[b, di_val, 0] = coord_num / 20.0  # normalize
                env_features[b, di_val, 1] = mean_dist / 5.0
                env_features[b, di_val, 2] = max_dist / 8.0
                env_features[b, di_val, 3] = en_contrast

            offset += n_atoms

        # Project and add to defect atoms only
        env_emb = self.proj(env_features)  # (B, N_max, C)
        defect_mask_f = defect_mask.float().unsqueeze(-1)  # (B, N_max, 1)
        h = h + env_emb * defect_mask_f

        return h


# ---------------------------------------------------------------------------
# CrystalTransformerV2
# ---------------------------------------------------------------------------
class CrystalTransformerV2(nn.Module):
    """CrystalTransformer with architecture improvements for better defect
    formation energy prediction.

    Changes from V1:
      - Pre-norm local layers with edge gating
      - Defect-aware gated attention + max pooling readout
      - Local environment enrichment for defect sites
      - Optional knowledge distillation support via `return_hidden`
    """

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
        ct_uae_path: str = None,
        n_readout_heads: int = 1,
        # V2 additions
        use_gated_pooling: bool = True,
        use_env_enrichment: bool = True,
        use_prenorm_local: bool = True,
    ) -> None:
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.hidden_dim = hidden_dim

        # --- Input embedding ---
        if ct_uae_path is not None:
            uae_table = torch.load(ct_uae_path, map_location="cpu",
                                   weights_only=False)
            self.register_buffer("ct_uae_table", uae_table)
            uae_dim = uae_table.shape[1]
            self.embed = nn.Linear(atom_fea_len + uae_dim, hidden_dim)
        else:
            self.ct_uae_table = None
            self.embed = nn.Linear(atom_fea_len, hidden_dim)

        # --- Defect embedding ---
        self.defect_embedding = (
            nn.Embedding(2, hidden_dim) if defect_embedding else None
        )

        # --- Local environment enrichment ---
        self.use_env_enrichment = use_env_enrichment
        self.env_enrichment = (
            LocalEnvEnrichment(hidden_dim) if use_env_enrichment else None
        )

        # --- Local interaction layers ---
        self.edge_rbf = RBFExpansion(0.0, rcut_local, n_rbf_edge)
        LocalLayerClass = (
            LocalInteractionLayerV2 if use_prenorm_local
            else _import_v1_local_layer()
        )
        self.local_layers = nn.ModuleList([
            LocalLayerClass(hidden_dim, n_rbf_edge=n_rbf_edge)
            for _ in range(n_local_layers)
        ])

        # --- Global transformer layers (reuse V1 — already good) ---
        from .baseline import GeometricTransformerBlock
        self.global_layers = nn.ModuleList([
            GeometricTransformerBlock(
                hidden_dim, num_heads=num_heads,
                n_rbf_dist=n_rbf_dist, dmax=dmax_global,
                dropout=dropout,
            )
            for _ in range(n_global_layers)
        ])

        # --- Readout ---
        self.use_gated_pooling = use_gated_pooling
        if use_gated_pooling:
            self.pooling = DefectAwarePooling(hidden_dim, dropout=dropout)
        # else: fall back to mean pooling (same as V1)

        self.n_readout_heads = n_readout_heads
        if n_readout_heads <= 1:
            self.readout = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.readout = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(n_readout_heads)
            ])

    # Reuse V1's edge flattening
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
            torch.cat(all_edges, dim=1) if all_edges
            else torch.empty(2, 0, dtype=torch.long, device=device)
        )
        edge_dist = (
            torch.cat(all_dist, dim=0) if all_dist
            else torch.empty(0, dtype=torch.float32, device=device)
        )
        triplet_index = (
            torch.cat(all_triplets, dim=0) if all_triplets
            else torch.empty(0, 3, dtype=torch.long, device=device)
        )
        angles = (
            torch.cat(all_angles, dim=0) if all_angles
            else torch.empty(0, dtype=torch.float32, device=device)
        )
        return edge_index, edge_dist, triplet_index, angles

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_hidden: bool = False,
    ) -> torch.Tensor:
        x = batch["x"]
        mask = batch["atom_mask"]
        dist_matrix = batch["dist_matrix"]
        defect_mask = batch.get("defect_mask")
        device = x.device

        # --- Input embedding ---
        x_raw = x  # keep raw features for env enrichment
        if self.ct_uae_table is not None:
            z = batch.get("atomic_numbers")
            if z is not None:
                z_clamped = z.clamp(0, self.ct_uae_table.shape[0] - 1)
                uae_fea = self.ct_uae_table[z_clamped]
                x = torch.cat([x, uae_fea], dim=-1)
        h = self.embed(x)

        if self.defect_embedding is not None and defect_mask is not None:
            h = h + self.defect_embedding(defect_mask)

        # --- Local environment enrichment ---
        if (self.use_env_enrichment and self.env_enrichment is not None
                and defect_mask is not None):
            h = self.env_enrichment(
                h, defect_mask,
                batch.get("edge_index_list", []),
                batch.get("edge_dist_list", []),
                batch["num_atoms_list"],
                x_raw,
            )

        # --- Local message passing ---
        b, n_max, c = h.shape
        num_atoms_list = batch["num_atoms_list"]
        flat_indices = []
        for i, n_i in enumerate(num_atoms_list):
            base = i * n_max
            flat_indices.append(
                torch.arange(n_i, device=device, dtype=torch.long) + base
            )
        flat_indices = (
            torch.cat(flat_indices)
            if flat_indices
            else torch.empty(0, dtype=torch.long, device=device)
        )
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
            flat_h = layer(
                flat_h, edge_index, edge_attr_rbf, triplet_index, angles
            )

        # Scatter back to padded layout
        h_local_flat = torch.zeros(b * n_max, c, dtype=h.dtype, device=device)
        h_local_flat.index_copy_(0, flat_indices, flat_h)
        h_local = h_local_flat.reshape(b, n_max, c)

        # --- Global transformer ---
        h_global = h_local
        for layer in self.global_layers:
            h_global = layer(h_global, dist_matrix, mask)

        # --- Readout ---
        if self.use_gated_pooling:
            pooled = self.pooling(h_global, mask)
        else:
            mask_f = mask.float().unsqueeze(-1)
            pooled = (h_global * mask_f).sum(dim=1) / mask_f.sum(
                dim=1
            ).clamp(min=1.0)

        if return_hidden:
            # For knowledge distillation — return both prediction and hidden
            if self.n_readout_heads <= 1:
                pred = self.readout(pooled).squeeze(-1)
            else:
                preds = torch.stack(
                    [head(pooled).squeeze(-1) for head in self.readout], dim=0
                )
                pred = preds.mean(dim=0)
            return pred, pooled

        if self.n_readout_heads <= 1:
            return self.readout(pooled).squeeze(-1)
        preds = torch.stack(
            [head(pooled).squeeze(-1) for head in self.readout], dim=0
        )
        return preds.mean(dim=0)


def _import_v1_local_layer():
    """Fallback: import original LocalInteractionLayer for compatibility."""
    from .baseline import LocalInteractionLayer
    return LocalInteractionLayer
