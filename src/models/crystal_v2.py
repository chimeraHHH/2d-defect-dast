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
# Mixture-of-Experts Readout (V4)
# ---------------------------------------------------------------------------
class MoEReadout(nn.Module):
    """Mixture-of-Experts readout for crystal property prediction.

    Inspired by MoCE (ICLR 2025): multiple expert MLPs with learned gating.
    Each expert can specialise in different regions of the property space —
    e.g. one expert handles typical formation energies [0, 5] eV while
    another handles extreme values [10, 20] eV.

    The gating network operates on the pooled graph representation, so
    the specialisation is input-dependent (not hard-coded by target range).

    A load-balancing auxiliary loss encourages uniform expert utilisation
    across the batch, preventing expert collapse.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_experts: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts

        # Expert heads — each is a lightweight MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(n_experts)
        ])

        # Gating network — produces soft expert weights per sample
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_experts),
        )

        # Store balance loss for training (set by forward)
        self._balance_loss: float = 0.0

    @property
    def balance_loss(self) -> float:
        return self._balance_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, hidden_dim) pooled graph representation
        Returns:
            pred: (B,) weighted expert predictions
        """
        gate_logits = self.gate(x)                              # (B, E)
        gate_weights = F.softmax(gate_logits, dim=-1)           # (B, E)

        expert_out = torch.stack(
            [expert(x).squeeze(-1) for expert in self.experts],
            dim=-1,
        )                                                       # (B, E)

        pred = (expert_out * gate_weights).sum(dim=-1)          # (B,)

        # Load-balancing loss: KL(avg_gate || uniform)
        avg_gate = gate_weights.mean(dim=0)                     # (E,)
        uniform = torch.ones_like(avg_gate) / self.n_experts
        self._balance_loss = F.kl_div(
            (avg_gate + 1e-8).log(), uniform, reduction="sum"
        )

        return pred


# ---------------------------------------------------------------------------
# Local Environment Enrichment
# ---------------------------------------------------------------------------
class LocalEnvEnrichment(nn.Module):
    """Vectorized physics-aware features for defect sites.

    Features per defect atom (broadcast to embedding space):
      - Coordination number (count of edges from defect atom)
      - Mean neighbour distance
      - Max neighbour distance (captures strain)
      - Electronegativity contrast: |EN_defect - mean(EN_host_neighbours)|

    Uses scatter operations for GPU-efficient computation — no Python loops.
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
        edge_index_flat: torch.Tensor,
        edge_dist_flat: torch.Tensor,
        flat_defect_mask: torch.Tensor,
        flat_en: torch.Tensor,
        flat_indices: torch.Tensor,
        num_atoms_list: list,
    ) -> torch.Tensor:
        """Add local environment features to defect atoms (vectorized).

        Args:
            h: (B, N_max, C) current embeddings
            defect_mask: (B, N_max) int, 1 for defect atoms
            edge_index_flat: (2, E_total) flattened edge indices (with offsets)
            edge_dist_flat: (E_total,) flattened edge distances
            flat_defect_mask: (N_total,) defect mask in flat atom space
            flat_en: (N_total,) electronegativity in flat atom space
            flat_indices: (N_total,) mapping flat atoms back to padded layout
            num_atoms_list: atoms per sample
        Returns:
            h: (B, N_max, C) with enriched defect embeddings
        """
        device = h.device
        B, N_max, C = h.shape
        N_total = flat_defect_mask.shape[0]

        if edge_index_flat.shape[1] == 0:
            return h

        row, col = edge_index_flat  # row=center, col=neighbor

        # Filter to edges whose center is a defect atom
        is_defect_center = flat_defect_mask[row].bool()
        if not is_defect_center.any():
            return h

        d_row = row[is_defect_center]          # defect center indices
        d_dist = edge_dist_flat[is_defect_center]  # corresponding distances
        d_col = col[is_defect_center]          # neighbor indices

        # 1. Coordination number per defect atom (scatter_add count)
        ones = torch.ones_like(d_dist)
        coord_count = torch.zeros(N_total, device=device, dtype=h.dtype)
        coord_count.scatter_add_(0, d_row, ones)

        # 2. Sum of distances per defect atom → mean = sum / count
        dist_sum = torch.zeros(N_total, device=device, dtype=h.dtype)
        dist_sum.scatter_add_(0, d_row, d_dist)
        safe_count = coord_count.clamp(min=1.0)
        mean_dist = dist_sum / safe_count

        # 3. Max distance per defect atom
        max_dist = torch.zeros(N_total, device=device, dtype=h.dtype)
        max_dist.scatter_reduce_(0, d_row, d_dist, reduce="amax",
                                 include_self=False)

        # 4. EN contrast: |EN_defect - mean(EN_neighbors)|
        en_neigh = flat_en[d_col]
        en_sum = torch.zeros(N_total, device=device, dtype=h.dtype)
        en_sum.scatter_add_(0, d_row, en_neigh)
        en_mean_neigh = en_sum / safe_count
        en_contrast = (flat_en - en_mean_neigh).abs()

        # Stack features: (N_total, 4), normalized
        env_fea = torch.stack([
            coord_count / 20.0,
            mean_dist / 5.0,
            max_dist / 8.0,
            en_contrast,
        ], dim=-1)

        # Only keep features for defect atoms (zero out non-defect)
        env_fea = env_fea * flat_defect_mask.float().unsqueeze(-1)

        # Write back to padded layout (B, N_max, 4)
        env_padded = torch.zeros(B * N_max, 4, device=device, dtype=h.dtype)
        env_padded.index_copy_(0, flat_indices, env_fea)
        env_padded = env_padded.reshape(B, N_max, 4)

        # Project and add
        env_emb = self.proj(env_padded)
        defect_mask_f = defect_mask.float().unsqueeze(-1)
        h = h + env_emb * defect_mask_f

        return h


# ---------------------------------------------------------------------------
# Enhanced Local Environment Enrichment V2 (multi-shell + angular)
# ---------------------------------------------------------------------------
class LocalEnvEnrichmentV2(nn.Module):
    """Enhanced physics-aware features for defect sites.

    Extends V1 with multi-scale and topological descriptors inspired by
    persistent homology approaches (Hossain et al., Chem. Mater. 2024).

    Features per defect atom (11 total):
      Shell features (3 shells: 0-2A, 2-4A, 4-rcut):
        - Coordination number per shell (3)
        - Mean distance per shell (3)
      Aggregated features:
        - Global coordination number (1)
        - Min neighbor distance (strain indicator) (1)
        - Distance std (disorder indicator) (1)
        - Electronegativity contrast (1)
        - Electronegativity std of neighbors (1)

    All computed via vectorized scatter operations — no Python loops.
    """

    SHELL_BOUNDARIES = [0.0, 2.0, 4.0]  # Shell starts; last shell extends to rcut
    N_ENV_FEATURES = 11

    def __init__(self, hidden_dim: int, n_env_features: int = 11) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_env_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        defect_mask: torch.Tensor,
        edge_index_flat: torch.Tensor,
        edge_dist_flat: torch.Tensor,
        flat_defect_mask: torch.Tensor,
        flat_en: torch.Tensor,
        flat_indices: torch.Tensor,
        num_atoms_list: list,
    ) -> torch.Tensor:
        device = h.device
        B, N_max, C = h.shape
        N_total = flat_defect_mask.shape[0]

        if edge_index_flat.shape[1] == 0:
            return h

        row, col = edge_index_flat
        is_defect_center = flat_defect_mask[row].bool()
        if not is_defect_center.any():
            return h

        d_row = row[is_defect_center]
        d_dist = edge_dist_flat[is_defect_center]
        d_col = col[is_defect_center]

        features = []

        # --- Shell-resolved features ---
        shells = self.SHELL_BOUNDARIES + [999.0]  # last shell extends to rcut
        for s in range(len(shells) - 1):
            lo, hi = shells[s], shells[s + 1]
            shell_mask = (d_dist >= lo) & (d_dist < hi)
            s_row = d_row[shell_mask]
            s_dist = d_dist[shell_mask]

            # Coordination count per shell
            shell_count = torch.zeros(N_total, device=device, dtype=h.dtype)
            if s_row.numel() > 0:
                shell_count.scatter_add_(0, s_row, torch.ones_like(s_dist))
            features.append(shell_count / 12.0)  # normalize by typical coord

            # Mean distance per shell
            shell_dist_sum = torch.zeros(N_total, device=device, dtype=h.dtype)
            if s_row.numel() > 0:
                shell_dist_sum.scatter_add_(0, s_row, s_dist)
            safe_sc = shell_count.clamp(min=1.0)
            features.append(shell_dist_sum / safe_sc / 6.0)  # normalize

        # --- Global aggregated features ---
        # Total coordination
        ones = torch.ones_like(d_dist)
        coord_count = torch.zeros(N_total, device=device, dtype=h.dtype)
        coord_count.scatter_add_(0, d_row, ones)
        safe_count = coord_count.clamp(min=1.0)
        features.append(coord_count / 20.0)

        # Min distance (local compression indicator)
        min_dist = torch.full((N_total,), 99.0, device=device, dtype=h.dtype)
        min_dist.scatter_reduce_(0, d_row, d_dist, reduce="amin",
                                 include_self=False)
        min_dist = min_dist.clamp(max=10.0)
        features.append(min_dist / 5.0)

        # Distance std (local disorder)
        dist_sum = torch.zeros(N_total, device=device, dtype=h.dtype)
        dist_sum.scatter_add_(0, d_row, d_dist)
        mean_dist = dist_sum / safe_count
        dist_sq_sum = torch.zeros(N_total, device=device, dtype=h.dtype)
        dist_sq_sum.scatter_add_(0, d_row, d_dist ** 2)
        dist_var = (dist_sq_sum / safe_count - mean_dist ** 2).clamp(min=0)
        dist_std = dist_var.sqrt()
        features.append(dist_std / 2.0)

        # EN contrast
        en_neigh = flat_en[d_col]
        en_sum = torch.zeros(N_total, device=device, dtype=h.dtype)
        en_sum.scatter_add_(0, d_row, en_neigh)
        en_mean_neigh = en_sum / safe_count
        en_contrast = (flat_en - en_mean_neigh).abs()
        features.append(en_contrast)

        # EN std of neighbors (chemical diversity around defect)
        en_sq_sum = torch.zeros(N_total, device=device, dtype=h.dtype)
        en_sq_sum.scatter_add_(0, d_row, en_neigh ** 2)
        en_var = (en_sq_sum / safe_count - en_mean_neigh ** 2).clamp(min=0)
        en_std = en_var.sqrt()
        features.append(en_std)

        # Stack: (N_total, 11)
        env_fea = torch.stack(features, dim=-1)
        env_fea = env_fea * flat_defect_mask.float().unsqueeze(-1)

        # Write back to padded layout
        env_padded = torch.zeros(B * N_max, self.N_ENV_FEATURES,
                                 device=device, dtype=h.dtype)
        env_padded.index_copy_(0, flat_indices, env_fea)
        env_padded = env_padded.reshape(B, N_max, self.N_ENV_FEATURES)

        # Project and add (residual)
        env_emb = self.proj(env_padded)
        defect_mask_f = defect_mask.float().unsqueeze(-1)
        h = h + env_emb * defect_mask_f

        return h


# ---------------------------------------------------------------------------
# Physics-Motivated Dopant–Host Mismatch Features (V6)
# ---------------------------------------------------------------------------
class PhysicsFeatureModule(nn.Module):
    """Graph-level dopant–host mismatch descriptors based on Hume-Rothery rules.

    Computes 6 physics features that capture the fundamental drivers of defect
    formation energy from the periodic table properties of dopant vs host atoms:

      1. Size mismatch:  (r_d - ⟨r_h⟩) / ⟨r_h⟩
         → elastic strain energy; Hume-Rothery's 15% rule predicts limited
           solubility when |Δr/r| > 0.15.
      2. EN difference:  χ_d - ⟨χ_h⟩  (signed)
         → charge transfer direction & magnitude (Pauling).
      3. IE ratio:  IE_d / ⟨IE_h⟩
         → chemical hardness matching; related to Pearson's HSAB principle.
      4. EA difference:  EA_d - ⟨EA_h⟩
         → electron-accepting tendency; important for adsorbate binding.
      5. Valence mismatch:  |VE_d - mode(VE_h)| / 8
         → bonding compatibility: mismatched valence creates dangling bonds
           or requires charge compensation.
      6. Period distance:  period_d - ⟨period_h⟩  (signed)
         → orbital overlap quality; same-period → better spatial match.

    All features are computed from batch tensors via vectorised masked ops.
    The final projection is zero-initialised so the module starts as a no-op,
    preserving compatibility with pretrained V2 weights.

    References:
        Hume-Rothery W., "The Structure of Metals and Alloys" (1936).
        Bartel C. J. et al., Sci. Adv. 6, eaaz0510 (2020) — elemental
            feature importance for formation energy prediction.
        Ward L. et al., npj Comput. Mater. 2, 16028 (2016) — Magpie
            compositional descriptors.
        Goodall R. E. A. & Lee A. A., Nature Commun. 11, 6280 (2020) —
            Roost: compositional representation for property prediction.
    """

    N_FEATURES = 6

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Build raw (un-normalised) elemental property lookup tables as buffers.
        # Index by atomic number Z; Z=0 is a padding placeholder.
        from ..features import (
            PAULING_EN, IONIZATION_ENERGY, ELECTRON_AFFINITY, GROUP, PERIOD,
        )
        from ase.data import covalent_radii as _cov_r

        max_z = 100
        cov_r = torch.zeros(max_z + 1)
        en = torch.zeros(max_z + 1)
        ie = torch.zeros(max_z + 1)
        ea = torch.zeros(max_z + 1)
        ve = torch.zeros(max_z + 1)
        period = torch.zeros(max_z + 1)
        for z in range(1, max_z + 1):
            cov_r[z] = _cov_r[z]
            en[z] = PAULING_EN[z]
            ie[z] = IONIZATION_ENERGY[z]
            ea[z] = ELECTRON_AFFINITY[z]
            ve[z] = float(GROUP[z] if GROUP[z] <= 2 else (GROUP[z] - 10 if GROUP[z] >= 13 else GROUP[z]))
            period[z] = float(PERIOD[z])

        self.register_buffer("_cov_radius", cov_r)
        self.register_buffer("_pauling_en", en)
        self.register_buffer("_ionization_e", ie)
        self.register_buffer("_electron_aff", ea)
        self.register_buffer("_valence_e", ve)
        self.register_buffer("_period", period)

        # Projection: 6 raw physics features → hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        # Zero-init so the module starts as a no-op (safe for pretrained ckpts)
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        defect_mask: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute graph-level physics mismatch features.

        Args:
            atomic_numbers: (B, N_max) atomic numbers (long)
            defect_mask: (B, N_max) int, 1 for defect atoms
            atom_mask: (B, N_max) bool, True for valid atoms
        Returns:
            (B, hidden_dim) physics-conditioned representation
        """
        z = atomic_numbers.clamp(0, len(self._cov_radius) - 1)

        # Look up all properties: (B, N_max)
        r_all = self._cov_radius[z]
        en_all = self._pauling_en[z]
        ie_all = self._ionization_e[z]
        ea_all = self._electron_aff[z]
        ve_all = self._valence_e[z]
        per_all = self._period[z]

        # Masks: (B, N_max) float
        defect_f = defect_mask.float()
        host_f = ((~defect_mask.bool()) & atom_mask).float()
        d_count = defect_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        h_count = host_f.sum(dim=1, keepdim=True).clamp(min=1.0)    # (B, 1)

        # Weighted means for dopant and host
        d_r = (r_all * defect_f).sum(1) / d_count.squeeze(1)
        h_r = (r_all * host_f).sum(1) / h_count.squeeze(1)
        d_en = (en_all * defect_f).sum(1) / d_count.squeeze(1)
        h_en = (en_all * host_f).sum(1) / h_count.squeeze(1)
        d_ie = (ie_all * defect_f).sum(1) / d_count.squeeze(1)
        h_ie = (ie_all * host_f).sum(1) / h_count.squeeze(1)
        d_ea = (ea_all * defect_f).sum(1) / d_count.squeeze(1)
        h_ea = (ea_all * host_f).sum(1) / h_count.squeeze(1)
        d_ve = (ve_all * defect_f).sum(1) / d_count.squeeze(1)
        h_ve = (ve_all * host_f).sum(1) / h_count.squeeze(1)
        d_per = (per_all * defect_f).sum(1) / d_count.squeeze(1)
        h_per = (per_all * host_f).sum(1) / h_count.squeeze(1)

        # Compute 6 physics features: (B,) each
        f1 = (d_r - h_r) / h_r.clamp(min=0.1)               # size mismatch
        f2 = d_en - h_en                                      # EN difference
        f3 = d_ie / h_ie.clamp(min=0.1)                      # IE ratio
        f4 = d_ea - h_ea                                      # EA difference
        f5 = (d_ve - h_ve).abs() / 8.0                       # valence mismatch
        f6 = (d_per - h_per) / 3.0                            # period distance

        features = torch.stack([f1, f2, f3, f4, f5, f6], dim=-1)  # (B, 6)

        return self.proj(features)  # (B, hidden_dim)


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
        env_enrichment_version: int = 1,  # 1=original (4 features), 2=enhanced (11 features)
        # V3: defect-type conditioning
        use_defect_type_cond: bool = False,
        n_defect_types: int = 4,  # vacancy, substitution, interstitial, adsorbate
        # V4: Mixture of Experts readout
        use_moe_readout: bool = False,
        n_moe_experts: int = 3,
        moe_balance_weight: float = 0.01,
        # V6: Physics-motivated dopant–host mismatch
        use_physics_features: bool = False,
        # JK: Jumping Knowledge aggregation over layers
        use_jk_aggregation: bool = False,
        # Uncertainty head: predict log-variance alongside Ef
        predict_uncertainty: bool = False,
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

        # --- Defect-type conditioning (V3) ---
        # Embeds the global defect type (vacancy/sub/interstitial/adsorbate)
        # and conditions the readout head. Physically motivated: interstitials
        # and adsorbates span much wider Ef ranges than vacancies/substitutions.
        self.use_defect_type_cond = use_defect_type_cond
        if use_defect_type_cond:
            self.defect_type_embed = nn.Embedding(n_defect_types, hidden_dim)
            nn.init.zeros_(self.defect_type_embed.weight)  # start as no-op
        else:
            self.defect_type_embed = None

        # --- Physics features (V6): dopant–host mismatch ---
        self.use_physics_features = use_physics_features
        if use_physics_features:
            self.physics_module = PhysicsFeatureModule(hidden_dim)
        else:
            self.physics_module = None

        # --- Local environment enrichment ---
        self.use_env_enrichment = use_env_enrichment
        if use_env_enrichment:
            if env_enrichment_version == 2:
                self.env_enrichment = LocalEnvEnrichmentV2(hidden_dim)
            else:
                self.env_enrichment = LocalEnvEnrichment(hidden_dim)
        else:
            self.env_enrichment = None

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

        # --- JK aggregation (optional, Xu et al. ICML 2018) ---
        # Learn to weight features from [local_out, global_1, ..., global_L]
        # so the readout sees both local defect detail and global context.
        self.use_jk_aggregation = use_jk_aggregation
        if use_jk_aggregation:
            n_jk = 1 + n_global_layers  # local output + each global layer
            self.jk_weights = nn.Parameter(torch.zeros(n_jk))  # init uniform

        # --- Readout ---
        self.use_gated_pooling = use_gated_pooling
        if use_gated_pooling:
            self.pooling = DefectAwarePooling(hidden_dim, dropout=dropout)
        # else: fall back to mean pooling (same as V1)

        self.n_readout_heads = n_readout_heads
        self.use_moe_readout = use_moe_readout
        self.moe_balance_weight = moe_balance_weight

        if use_moe_readout:
            self.readout = MoEReadout(
                hidden_dim, n_experts=n_moe_experts, dropout=dropout,
            )
        elif n_readout_heads <= 1:
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

        # --- Uncertainty head (Kendall & Gal, NeurIPS 2017) ---
        # Predicts log-variance σ² alongside Ef, enabling heteroscedastic
        # loss: L = |y - ŷ| * exp(-s) + s  where s = log(σ²).
        # This naturally downweights noisy/hard samples and provides
        # calibrated uncertainty estimates for ensemble weighting.
        self.predict_uncertainty = predict_uncertainty
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            # Init bias to log(1) = 0 → initial σ = 1 (unit variance)
            nn.init.zeros_(self.uncertainty_head[-1].bias)

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

        # --- Build flat atom indices (shared by env enrichment + local layers) ---
        b, n_max, c = h.shape
        num_atoms_list = batch["num_atoms_list"]
        flat_idx_parts = []
        for i, n_i in enumerate(num_atoms_list):
            base = i * n_max
            flat_idx_parts.append(
                torch.arange(n_i, device=device, dtype=torch.long) + base
            )
        flat_indices = (
            torch.cat(flat_idx_parts)
            if flat_idx_parts
            else torch.empty(0, dtype=torch.long, device=device)
        )

        # --- Flatten edges (used by both env enrichment and local layers) ---
        edge_index, edge_dist, triplet_index, angles = self._flatten_edges(
            num_atoms_list,
            batch["edge_index_list"],
            batch["edge_dist_list"],
            batch["triplet_index_list"],
            batch["angles_list"],
            device=device,
        )

        # --- Local environment enrichment (vectorized via scatter) ---
        if (self.use_env_enrichment and self.env_enrichment is not None
                and defect_mask is not None):
            # Build flat defect mask and flat EN for the valid atoms
            n_total = flat_indices.shape[0]
            flat_defect = torch.zeros(n_total, device=device, dtype=torch.long)
            flat_en = torch.zeros(n_total, device=device, dtype=h.dtype)
            offset = 0
            for i, n_i in enumerate(num_atoms_list):
                flat_defect[offset:offset + n_i] = defect_mask[i, :n_i]
                flat_en[offset:offset + n_i] = x_raw[i, :n_i, 2]
                offset += n_i
            h = self.env_enrichment(
                h, defect_mask, edge_index, edge_dist,
                flat_defect, flat_en, flat_indices, num_atoms_list,
            )

        # --- Local message passing ---
        h_flat_full = h.reshape(b * n_max, c)
        flat_h = h_flat_full.index_select(0, flat_indices)

        edge_attr_rbf = self.edge_rbf(edge_dist)
        for layer in self.local_layers:
            flat_h = layer(
                flat_h, edge_index, edge_attr_rbf, triplet_index, angles
            )

        # Scatter back to padded layout
        h_local_flat = torch.zeros(b * n_max, c, dtype=h.dtype, device=device)
        h_local_flat.index_copy_(0, flat_indices, flat_h)
        h_local = h_local_flat.reshape(b, n_max, c)

        # --- Global transformer (with optional JK aggregation) ---
        if self.use_jk_aggregation:
            jk_layers = [h_local]  # local output as first representation
            h_global = h_local
            for layer in self.global_layers:
                h_global = layer(h_global, dist_matrix, mask)
                jk_layers.append(h_global)
            # Weighted combination: softmax over learnable layer weights
            jk_w = F.softmax(self.jk_weights, dim=0)         # (n_jk,)
            h_stack = torch.stack(jk_layers, dim=-1)          # (B, N, C, n_jk)
            h_global = (h_stack * jk_w).sum(dim=-1)           # (B, N, C)
        else:
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

        # --- Defect-type conditioning (V3): add type embedding to pooled repr ---
        if self.use_defect_type_cond and self.defect_type_embed is not None:
            dt = batch.get("defect_type")
            if dt is not None:
                pooled = pooled + self.defect_type_embed(dt)

        # --- Physics features (V6): dopant–host mismatch conditioning ---
        if self.use_physics_features and self.physics_module is not None:
            z_batch = batch.get("atomic_numbers")
            if z_batch is not None and defect_mask is not None:
                pooled = pooled + self.physics_module(z_batch, defect_mask, mask)

        # --- Compute prediction ---
        if self.use_moe_readout:
            pred = self.readout(pooled)
        elif self.n_readout_heads <= 1:
            pred = self.readout(pooled).squeeze(-1)
        else:
            preds = torch.stack(
                [head(pooled).squeeze(-1) for head in self.readout], dim=0
            )
            pred = preds.mean(dim=0)

        if return_hidden:
            return pred, pooled

        # --- Uncertainty estimation (optional) ---
        if self.predict_uncertainty:
            log_var = self.uncertainty_head(pooled).squeeze(-1)  # (B,)
            return pred, log_var

        return pred


def _import_v1_local_layer():
    """Fallback: import original LocalInteractionLayer for compatibility."""
    from .baseline import LocalInteractionLayer
    return LocalInteractionLayer
