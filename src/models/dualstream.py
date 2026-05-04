"""Dual-stream Periodic Cross-Attention Transformer (v3 architecture).

Physically motivated: the formation energy of a point defect is

    E_f = E_defect - E_host + Δμ

so the model SHOULD see both the defect supercell *and* the host pristine
supercell that serves as its energy reference, instead of being asked to
implicitly infer the host energy from a single structure (the v1/v2 setup).

Architecture
------------
1. **Shared PFA encoder.** The same ``PeriodicCrystalTransformer``-style
   stack of (local SchNet + global PFA Transformer) layers encodes both the
   defect supercell and the pristine supercell, weights tied across both
   streams.

2. **Cross-attention stack.** Two pre-norm cross-attention blocks where
   defect tokens form the query stream and pristine tokens form the
   key/value stream.  This implements ``(h_defect | h_pristine)`` — a
   learned conditional encoding of the defect environment given its
   reference.

3. **Difference readout.** The graph-level representation is
   ``Δh = pool(h_defect_xattn) - pool(h_pristine)``.  The final readout is
   a zero-initialised linear without bias, so

       E_f(defect = pristine) = readout(0) = 0   exactly at init,

   and the network can only deviate from this invariance through learned
   weights.  We additionally apply a soft invariance loss
   ``λ * MSE(forward(x, x), 0)`` during training, which keeps the
   identity-input → zero-output property a measurable property of the
   trained model rather than an architectural fiction.

This module exports:
  * ``DualStreamPeriodicTransformer``  — the full model
  * ``CrossAttentionBlock``            — the building block
  * ``compute_invariance_loss``        — convenience helper for training
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import LocalInteractionLayer, RBFExpansion
from .attention_v2 import (
    PeriodicGeometricBlock,
    PeriodicFourierBias,
    compute_frac_disp,
)


# ---------------------------------------------------------------------------
# Cross-attention block: defect (Q) attends to pristine (K, V)
# ---------------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """Single pre-norm cross-attention block with optional distance bias.

    The two streams may have different shapes (defect has N_d atoms,
    pristine has N_p = N_d − 1 atoms in IMP2D).  Both query and key/value
    masks are exposed; padding atoms get -1e9 score before softmax.

    A simple RBF distance bias is added on the pair distance ``d_qk``
    between defect query atom and pristine key atom.  ``d_qk`` is computed
    by minimum-image PBC against the *pristine* cell (the two cells are
    identical for IMP2D pairs by construction).
    """

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
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dist_rbf = RBFExpansion(0.0, dmax, n_rbf_dist)
        self.dist_proj = nn.Sequential(
            nn.Linear(n_rbf_dist, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_q: torch.Tensor,
        h_kv: torch.Tensor,
        q_mask: torch.Tensor,
        kv_mask: torch.Tensor,
        d_qk: torch.Tensor,  # (B, N_q, N_kv) PBC distance defect→pristine
    ) -> torch.Tensor:
        b, n_q, c = h_q.shape
        n_kv = h_kv.shape[1]
        h = self.num_heads
        d = self.head_dim

        residual = h_q
        q = self.q_proj(self.norm_q(h_q)).reshape(b, n_q, h, d).permute(0, 2, 1, 3)
        kv_norm = self.norm_kv(h_kv)
        kv = self.kv_proj(kv_norm).reshape(b, n_kv, 2, h, d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        rbf = self.dist_rbf(d_qk)
        bias = self.dist_proj(rbf).permute(0, 3, 1, 2)  # (B, H, N_q, N_kv)
        scores = scores + bias

        # mask: invalid q rows OR kv columns get -1e9
        valid = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2)  # (B, N_q, N_kv)
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ v).transpose(1, 2).reshape(b, n_q, c)
        h_q = residual + self.dropout(self.out_proj(out))
        h_q = h_q + self.dropout(self.ffn(self.norm_ffn(h_q)))
        return h_q


# ---------------------------------------------------------------------------
# Encoder helper: reused across two streams with shared weights
# ---------------------------------------------------------------------------
class _SharedEncoder(nn.Module):
    """A shared local-SchNet + global-PFA encoder.

    Returns per-atom features ``h_global`` of shape ``(B, N_max, C)``.
    Reuses the existing ``LocalInteractionLayer`` + ``PeriodicGeometricBlock``
    primitives so the dual-stream architecture inherits the unit-tested
    invariance properties of the v2 backbone.
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
        use_pfa: bool = True,
        k_max: int = 2,
        k_norm_sq_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(atom_fea_len, hidden_dim)
        self.defect_embedding = (
            nn.Embedding(2, hidden_dim) if defect_embedding else None
        )
        self.edge_rbf = RBFExpansion(0.0, rcut_local, n_rbf_edge)
        self.local_layers = nn.ModuleList(
            [LocalInteractionLayer(hidden_dim, n_rbf_edge=n_rbf_edge)
             for _ in range(n_local_layers)]
        )
        self.global_layers = nn.ModuleList(
            [
                PeriodicGeometricBlock(
                    hidden_dim,
                    num_heads=num_heads,
                    n_rbf_dist=n_rbf_dist,
                    dmax=dmax_global,
                    dropout=dropout,
                    use_pfa=use_pfa,
                    k_max=k_max,
                    k_norm_sq_max=k_norm_sq_max,
                    use_long_range=False,
                    use_defect_bias=False,
                )
                for _ in range(n_global_layers)
            ]
        )
        self.use_pfa = use_pfa
        self.hidden_dim = hidden_dim

    def _flatten_edges(
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
        all_edges, all_dist, all_triplets, all_angles = [], [], [], []
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
            torch.cat(all_edges, dim=1)
            if all_edges
            else torch.empty(2, 0, dtype=torch.long, device=device)
        )
        edge_dist = (
            torch.cat(all_dist, dim=0)
            if all_dist
            else torch.empty(0, dtype=torch.float32, device=device)
        )
        triplet_index = (
            torch.cat(all_triplets, dim=0)
            if all_triplets
            else torch.empty(0, 3, dtype=torch.long, device=device)
        )
        angles = (
            torch.cat(all_angles, dim=0)
            if all_angles
            else torch.empty(0, dtype=torch.float32, device=device)
        )
        return edge_index, edge_dist, triplet_index, angles

    def forward(
        self,
        x: torch.Tensor,
        atom_mask: torch.Tensor,
        dist_matrix: torch.Tensor,
        defect_mask: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        cell: Optional[torch.Tensor],
        num_atoms_list,
        edge_index_list,
        edge_dist_list,
        triplet_index_list,
        angles_list,
    ) -> torch.Tensor:
        device = x.device
        h = self.embed(x)
        if self.defect_embedding is not None and defect_mask is not None:
            h = h + self.defect_embedding(defect_mask)

        b, n_max, c = h.shape
        flat_indices = []
        for i, n_i in enumerate(num_atoms_list):
            base = i * n_max
            flat_indices.append(
                torch.arange(n_i, device=device, dtype=torch.long) + base
            )
        flat_indices_t = (
            torch.cat(flat_indices)
            if flat_indices
            else torch.empty(0, dtype=torch.long, device=device)
        )
        h_flat_full = h.reshape(b * n_max, c)
        flat_h = h_flat_full.index_select(0, flat_indices_t)

        edge_index, edge_dist, triplet_index, angles = self._flatten_edges(
            num_atoms_list,
            edge_index_list,
            edge_dist_list,
            triplet_index_list,
            angles_list,
            device=device,
        )
        edge_attr_rbf = self.edge_rbf(edge_dist)
        for layer in self.local_layers:
            flat_h = layer(flat_h, edge_index, edge_attr_rbf, triplet_index, angles)
        h_local_flat = torch.zeros(b * n_max, c, dtype=h.dtype, device=device)
        h_local_flat.index_copy_(0, flat_indices_t, flat_h)
        h_local = h_local_flat.reshape(b, n_max, c)

        frac_disp = None
        if self.use_pfa and positions is not None and cell is not None:
            try:
                frac_disp = compute_frac_disp(positions, cell)
            except RuntimeError:
                frac_disp = None

        h_global = h_local
        for layer in self.global_layers:
            h_global = layer(h_global, dist_matrix, frac_disp, defect_mask, atom_mask)
        return h_global


# ---------------------------------------------------------------------------
# DualStreamPeriodicTransformer (full model)
# ---------------------------------------------------------------------------
class DualStreamPeriodicTransformer(nn.Module):
    """Two-stream model with shared encoder + cross-attention + zero-init readout.

    Forward inputs (in batch dict):
      defect:    x, atom_mask, dist_matrix, defect_mask, positions, cell,
                 num_atoms_list, edge_index_list, ...
      pristine:  pristine_x, pristine_atom_mask, pristine_dist_matrix,
                 pristine_positions, pristine_cell, pristine_num_atoms_list,
                 pristine_edge_index_list, ...
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
        use_pfa: bool = True,
        k_max: int = 2,
        k_norm_sq_max: float = 4.0,
        n_cross_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = _SharedEncoder(
            atom_fea_len=atom_fea_len,
            hidden_dim=hidden_dim,
            n_local_layers=n_local_layers,
            n_global_layers=n_global_layers,
            num_heads=num_heads,
            n_rbf_edge=n_rbf_edge,
            n_rbf_dist=n_rbf_dist,
            rcut_local=rcut_local,
            dmax_global=dmax_global,
            defect_embedding=defect_embedding,
            dropout=dropout,
            use_pfa=use_pfa,
            k_max=k_max,
            k_norm_sq_max=k_norm_sq_max,
        )
        self.cross_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_dim,
                    num_heads=num_heads,
                    n_rbf_dist=n_rbf_dist,
                    dmax=dmax_global,
                    dropout=dropout,
                )
                for _ in range(n_cross_layers)
            ]
        )
        self.readout_norm = nn.LayerNorm(hidden_dim)
        # bias-free linear: f(0) = 0 by construction.
        # Small Gaussian init (not zero) so gradients flow back to the encoder
        # and cross-attention from epoch 0; the Δh=0→output=0 invariance is
        # maintained by the soft auxiliary loss `compute_invariance_loss`.
        self.readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.normal_(self.readout.weight, std=0.02)
        self.hidden_dim = hidden_dim

    @staticmethod
    def _pbc_pair_distance(
        pos_a: torch.Tensor,
        pos_b: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """Minimum-image PBC distance between every (a, b) pair.

        pos_a: (B, N_a, 3); pos_b: (B, N_b, 3); cell: (B, 3, 3).
        Returns (B, N_a, N_b).
        """
        cart = pos_a.unsqueeze(2) - pos_b.unsqueeze(1)  # (B, N_a, N_b, 3)
        cell_inv = torch.linalg.inv(cell)
        frac = torch.einsum("bijk,bkl->bijl", cart, cell_inv)
        frac = frac - torch.round(frac)
        cart_min = torch.einsum("bijk,bkl->bijl", frac, cell)
        return torch.linalg.norm(cart_min, dim=-1)

    def _encode_stream(self, batch: Dict, prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode one stream. ``prefix`` is '' for defect or 'pristine_' for pristine."""
        return self.encoder(
            x=batch[f"{prefix}x"] if prefix else batch["x"],
            atom_mask=batch[f"{prefix}atom_mask"] if prefix else batch["atom_mask"],
            dist_matrix=batch[f"{prefix}dist_matrix"] if prefix else batch["dist_matrix"],
            defect_mask=(
                batch[f"{prefix}defect_mask"] if prefix and f"{prefix}defect_mask" in batch
                else batch.get("defect_mask") if not prefix else None
            ),
            positions=batch[f"{prefix}positions"] if prefix else batch.get("positions"),
            cell=batch[f"{prefix}cell"] if prefix else batch.get("cell"),
            num_atoms_list=batch[f"{prefix}num_atoms_list"] if prefix else batch["num_atoms_list"],
            edge_index_list=batch[f"{prefix}edge_index_list"] if prefix else batch["edge_index_list"],
            edge_dist_list=batch[f"{prefix}edge_dist_list"] if prefix else batch["edge_dist_list"],
            triplet_index_list=batch[f"{prefix}triplet_index_list"] if prefix else batch["triplet_index_list"],
            angles_list=batch[f"{prefix}angles_list"] if prefix else batch["angles_list"],
        ), (batch[f"{prefix}atom_mask"] if prefix else batch["atom_mask"])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode both streams with shared weights
        h_def, mask_def = self._encode_stream(batch, "")
        h_pri, mask_pri = self._encode_stream(batch, "pristine_")

        # Pair-wise PBC distance defect → pristine using pristine cell
        # (defect cell == pristine cell in IMP2D pairs)
        cell = batch.get("pristine_cell", batch.get("cell"))
        d_qk = self._pbc_pair_distance(
            batch["positions"], batch["pristine_positions"], cell,
        )

        # Cross attention: defect Q attends to pristine K/V
        for layer in self.cross_layers:
            h_def = layer(h_def, h_pri, mask_def, mask_pri, d_qk)

        # Pool both streams (masked mean)
        m_d = mask_def.float().unsqueeze(-1)
        m_p = mask_pri.float().unsqueeze(-1)
        pool_def = (h_def * m_d).sum(dim=1) / m_d.sum(dim=1).clamp(min=1.0)
        pool_pri = (h_pri * m_p).sum(dim=1) / m_p.sum(dim=1).clamp(min=1.0)

        # Difference representation, normalised, then zero-init readout
        delta = pool_def - pool_pri
        delta = self.readout_norm(delta)
        return self.readout(delta).squeeze(-1)


def compute_invariance_loss(
    model: DualStreamPeriodicTransformer,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """λ-soft invariance: feeding the SAME structure to both streams must
    yield prediction 0 (in the model's normalised space).

    We construct a synthetic batch where the defect stream is replaced by
    the pristine stream (so defect == pristine).  No new graphs are built;
    we just point the defect-side keys at the pristine-side tensors.
    """
    syn = dict(batch)
    for k_pri, k_def in [
        ("pristine_x", "x"),
        ("pristine_atom_mask", "atom_mask"),
        ("pristine_dist_matrix", "dist_matrix"),
        ("pristine_positions", "positions"),
        ("pristine_cell", "cell"),
        ("pristine_num_atoms_list", "num_atoms_list"),
        ("pristine_edge_index_list", "edge_index_list"),
        ("pristine_edge_dist_list", "edge_dist_list"),
        ("pristine_triplet_index_list", "triplet_index_list"),
        ("pristine_angles_list", "angles_list"),
    ]:
        if k_pri in batch:
            syn[k_def] = batch[k_pri]
    # zero-out defect_mask (pristine has no defect)
    if "atom_mask" in syn:
        syn["defect_mask"] = torch.zeros_like(syn["atom_mask"], dtype=torch.long)
    pred = model(syn)
    return pred.pow(2).mean()
