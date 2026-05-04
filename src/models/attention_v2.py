"""Periodic-aware self-attention for 2D defect supercells (v2 architecture).

Three additive bias channels inject physics-grounded long-range structure into
the global self-attention layer used by ``CrystalTransformer``:

1. ``PeriodicFourierBias`` — direction- and anisotropy-aware bias built from a
   truncated Fourier basis on the (i, j) minimum-image fractional displacement
   ``f_ij = (r_i - r_j) C^{-1} mod 1``.  By construction every value is invariant
   under translation of the whole crystal AND under shifting any atom by a
   lattice vector, so periodicity is enforced exactly (not only via the scalar
   distance).
2. ``MultiScaleDistanceBias`` — replaces the single Gaussian-RBF distance bias
   with two physically distinguishable channels: a short-range RBF channel
   (chemistry, r ≲ 5 Å) and a long-range analytical channel built on
   {1/(r+ε), 1/(r+ε)^2, 1/(r+ε)^3, exp(-r/λ)} that decays smoothly to 0 at
   ``r_max``.  This breaks the implicit "one bandwidth fits both" assumption
   of the baseline RBF and gives the network a defensible inductive bias for
   defect-induced elastic / Coulombic tails.
3. ``DefectAwareBias`` — a 4-entry categorical bias keyed on
   (defect_i, defect_j) ∈ {0, 1}².  Cheap, parameter-light, and (unlike the
   failed DAST virtual token) only acts as a *bias* on existing real atoms,
   so it can never split the representation space.

The new global block ``PeriodicGeometricBlock`` is a drop-in replacement for
``GeometricTransformerBlock`` that consumes ``positions``, ``cell`` and
``defect_mask`` in addition to the regular ``dist_matrix`` and ``mask`` and
emits identical output shape — this means we can swap baseline → v2 layer-
by-layer for ablations.

The composing ``PeriodicCrystalTransformer`` re-uses the local SchNet path
unchanged and only changes the global stack.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import LocalInteractionLayer, RBFExpansion


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def compute_frac_disp(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Per-pair minimum-image fractional displacement.

    Args:
        positions: ``(B, N, 3)`` Cartesian positions, padded.
        cell: ``(B, 3, 3)`` lattice vectors with row i = a_i.

    Returns:
        ``(B, N, N, 3)`` fractional displacement in ``[-0.5, 0.5)`` along each
        lattice direction.  ``frac_disp[b, i, j, :] = (r_i - r_j) C^{-1}`` after
        wrapping.
    """
    cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
    cart_disp = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 3)
    frac = torch.einsum("bijk,bkl->bijl", cart_disp, cell_inv)
    frac = frac - torch.round(frac)
    return frac


def compute_pbc_dist(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Minimum-image PBC distance ``(B, N, N)`` from positions+cell."""
    frac = compute_frac_disp(positions, cell)
    cart = torch.einsum("bijk,bkl->bijl", frac, cell)
    return torch.linalg.norm(cart, dim=-1)


# ---------------------------------------------------------------------------
# Bias modules
# ---------------------------------------------------------------------------
class PeriodicFourierBias(nn.Module):
    """Learnable Fourier-basis attention bias on the minimum-image fractional
    displacement.

    For each pair we evaluate ``cos(2π k·f)`` and ``sin(2π k·f)`` for every
    integer ``k`` in a truncated set ``|k|^2 ≤ k_norm_sq_max`` (canonicalised so
    that ``k > 0`` lexicographically — this avoids storing both ``k`` and
    ``-k`` whose cos/sin features are linearly dependent).  A small MLP
    projects the resulting feature vector to ``num_heads`` head biases.

    Why this is principled:
        * Translation in real space cancels in pairwise differences.
        * Lattice shift ``r_i → r_i + L`` leaves ``f_ij`` invariant mod 1, and
          since ``cos`` and ``sin`` are 2π-periodic, the bias is *exactly*
          unchanged.  No tolerance, no "approximately periodic" claim.
        * The basis natively encodes direction and anisotropy that the scalar
          ``|r_ij|`` discards — important for layered 2D materials where the
          a vs b direction has very different chemistry.
    """

    def __init__(
        self,
        num_heads: int,
        k_max: int = 2,
        k_norm_sq_max: float = 4.0,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        ks: List[Tuple[int, int, int]] = []
        for kx in range(-k_max, k_max + 1):
            for ky in range(-k_max, k_max + 1):
                for kz in range(-k_max, k_max + 1):
                    norm_sq = kx * kx + ky * ky + kz * kz
                    if norm_sq == 0 or norm_sq > k_norm_sq_max:
                        continue
                    if (kx, ky, kz) <= (-kx, -ky, -kz):
                        continue  # keep canonical half (cos symmetry)
                    ks.append((kx, ky, kz))
        if not ks:
            raise ValueError("No k-vectors selected — relax k_max or k_norm_sq_max.")
        self.register_buffer("k_vecs", torch.tensor(ks, dtype=torch.float32))
        n_basis = 2 * len(ks) + 1  # cos, sin, plus DC term for graceful fall-through
        self.proj = nn.Sequential(
            nn.Linear(n_basis, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_heads),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, frac_disp: torch.Tensor) -> torch.Tensor:
        """frac_disp: ``(B, N, N, 3)`` → bias ``(B, num_heads, N, N)``."""
        phase = 2.0 * math.pi * (frac_disp @ self.k_vecs.t())  # (B, N, N, M)
        feats = torch.cat(
            [torch.ones_like(phase[..., :1]), torch.cos(phase), torch.sin(phase)],
            dim=-1,
        )
        bias = self.proj(feats)  # (B, N, N, H)
        return bias.permute(0, 3, 1, 2).contiguous()


class MultiScaleDistanceBias(nn.Module):
    """Two-channel distance bias: short-range RBF + long-range analytical.

    The short-range channel is a Gaussian RBF active for ``r ≤ r_short`` and
    smoothly suppressed beyond.  The long-range channel mixes
    ``[1/(r+δ), 1/(r+δ)^2, 1/(r+δ)^3, exp(-r/λ_k)]_k`` with learnable head
    weights, smoothly suppressed at ``r_max``.  Both channels are added.
    Returning a (B, H, N, N) tensor makes this drop-in compatible with the
    baseline ``GeometricTransformerBlock``.
    """

    def __init__(
        self,
        num_heads: int,
        n_rbf: int = 32,
        r_short: float = 5.0,
        r_max: float = 12.0,
        n_lambda: int = 4,
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.r_short = float(r_short)
        self.r_max = float(r_max)
        self.delta = float(delta)
        self.register_buffer("short_centers", torch.linspace(0.0, r_short, n_rbf))
        self.short_sigma = float(r_short / max(n_rbf - 1, 1))
        self.short_proj = nn.Sequential(
            nn.Linear(n_rbf, max(num_heads * 2, 16)),
            nn.SiLU(),
            nn.Linear(max(num_heads * 2, 16), num_heads),
        )
        self.long_lambda = nn.Parameter(torch.linspace(2.0, max(r_max - 2.0, 4.0), n_lambda))
        self.long_proj = nn.Sequential(
            nn.Linear(3 + n_lambda, max(num_heads * 2, 16)),
            nn.SiLU(),
            nn.Linear(max(num_heads * 2, 16), num_heads),
        )
        # init zero so the new bias starts inert
        nn.init.zeros_(self.short_proj[-1].weight)
        nn.init.zeros_(self.short_proj[-1].bias)
        nn.init.zeros_(self.long_proj[-1].weight)
        nn.init.zeros_(self.long_proj[-1].bias)

    @staticmethod
    def _smooth_window(r: torch.Tensor, r0: float, r1: float) -> torch.Tensor:
        """Smooth-step window: 1 inside ``[0, r0]``, 0 beyond ``r1``."""
        x = ((r - r0) / max(r1 - r0, 1e-6)).clamp(0.0, 1.0)
        s = 6.0 * x.pow(5) - 15.0 * x.pow(4) + 10.0 * x.pow(3)
        return 1.0 - s

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: ``(B, N, N)`` → ``(B, num_heads, N, N)``."""
        # Short range
        rbf = torch.exp(
            -((dist.unsqueeze(-1) - self.short_centers) ** 2) / (self.short_sigma ** 2)
        )
        short_w = self._smooth_window(dist, self.r_short - 1.0, self.r_short + 0.5)
        short_bias = self.short_proj(rbf) * short_w.unsqueeze(-1)
        # Long range: bounded analytical features (uses (r+δ) to avoid r=0 blow-up)
        r_safe = dist + self.delta
        inv_feats = torch.stack([1.0 / r_safe, 1.0 / r_safe ** 2, 1.0 / r_safe ** 3], dim=-1)
        lam = self.long_lambda.clamp(min=0.5)
        exp_feats = torch.exp(-dist.unsqueeze(-1) / lam)
        long_feats = torch.cat([inv_feats, exp_feats], dim=-1)
        long_w = self._smooth_window(dist, self.r_max - 2.0, self.r_max)
        long_bias = self.long_proj(long_feats) * long_w.unsqueeze(-1)
        return (short_bias + long_bias).permute(0, 3, 1, 2).contiguous()


class DefectAwareBias(nn.Module):
    """Per-pair categorical bias keyed on (defect_i, defect_j) ∈ {0,1}²."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_heads, 4))

    def forward(self, defect_mask: torch.Tensor) -> torch.Tensor:
        """defect_mask: ``(B, N)`` long. Returns ``(B, num_heads, N, N)``."""
        # idx[b, i, j] = defect_mask[b, i] * 2 + defect_mask[b, j]
        idx = (defect_mask.unsqueeze(2) * 2 + defect_mask.unsqueeze(1)).long()
        # gather: bias is (H, 4) → per-pair (B, N, N, H)
        flat = idx.reshape(-1)
        gathered = self.bias[:, flat]  # (H, B*N*N)
        bias = gathered.reshape(self.bias.shape[0], *idx.shape).permute(1, 0, 2, 3)
        return bias.contiguous()


# ---------------------------------------------------------------------------
# Periodic Geometric Block (drop-in for GeometricTransformerBlock)
# ---------------------------------------------------------------------------
class PeriodicGeometricBlock(nn.Module):
    """Self-attention block with PFA + multi-scale + defect biases.

    Drop-in replacement for ``GeometricTransformerBlock``.  Same input
    convention (``x``, padded ``mask``) but additionally consumes
    ``frac_disp``, ``dist_matrix`` and ``defect_mask`` already computed at
    model level once per forward pass.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        n_rbf_dist: int = 32,
        dmax: float = 12.0,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_pfa: bool = True,
        k_max: int = 2,
        k_norm_sq_max: float = 4.0,
        use_long_range: bool = True,
        use_defect_bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        if use_long_range:
            self.dist_bias = MultiScaleDistanceBias(
                num_heads, n_rbf=n_rbf_dist, r_short=5.0, r_max=dmax
            )
        else:
            # baseline parity: single RBF channel
            self.dist_rbf = RBFExpansion(0.0, dmax, n_rbf_dist)
            self.dist_proj = nn.Sequential(
                nn.Linear(n_rbf_dist, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_heads),
            )
            self.dist_bias = None  # signal: use baseline path
        self.pfa = PeriodicFourierBias(num_heads, k_max=k_max, k_norm_sq_max=k_norm_sq_max) if use_pfa else None
        self.defect_bias = DefectAwareBias(num_heads) if use_defect_bias else None

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
        frac_disp: Optional[torch.Tensor],
        defect_mask: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, c = x.shape
        h = self.num_heads
        d = self.head_dim
        residual = x
        xn = self.norm1(x)
        qkv = self.qkv(xn).reshape(b, n, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

        if self.dist_bias is not None:
            scores = scores + self.dist_bias(dist_matrix)
        else:
            rbf = self.dist_rbf(dist_matrix)
            bias = self.dist_proj(rbf).permute(0, 3, 1, 2)
            scores = scores + bias

        if self.pfa is not None and frac_disp is not None:
            scores = scores + self.pfa(frac_disp)
        if self.defect_bias is not None and defect_mask is not None:
            scores = scores + self.defect_bias(defect_mask)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.dropout(self.out_proj(out))
        x = residual + out
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Periodic Crystal Transformer (model)
# ---------------------------------------------------------------------------
class PeriodicCrystalTransformer(nn.Module):
    """Local SchNet + Global Periodic-aware Transformer."""

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
        # v2-specific knobs
        use_pfa: bool = True,
        k_max: int = 2,
        k_norm_sq_max: float = 4.0,
        use_long_range: bool = True,
        use_defect_bias: bool = True,
        recompute_dist_from_positions: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(atom_fea_len, hidden_dim)
        self.defect_embedding = (
            nn.Embedding(2, hidden_dim) if defect_embedding else None
        )
        self.edge_rbf = RBFExpansion(0.0, rcut_local, n_rbf_edge)
        self.local_layers = nn.ModuleList(
            [
                LocalInteractionLayer(hidden_dim, n_rbf_edge=n_rbf_edge)
                for _ in range(n_local_layers)
            ]
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
                    use_long_range=use_long_range,
                    use_defect_bias=use_defect_bias,
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
        self.use_pfa = use_pfa
        self.recompute_dist_from_positions = recompute_dist_from_positions

    # -- local edge flattening borrowed from baseline --
    def _flatten_edges(
        self,
        num_atoms_list: List[int],
        edge_index_list,
        edge_dist_list,
        triplet_index_list,
        angles_list,
        device: torch.device,
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        mask = batch["atom_mask"]
        dist_matrix = batch["dist_matrix"]
        defect_mask = batch.get("defect_mask")
        positions = batch.get("positions")
        cell = batch.get("cell")
        device = x.device

        h = self.embed(x)
        if self.defect_embedding is not None and defect_mask is not None:
            h = h + self.defect_embedding(defect_mask)

        # Local pass on flattened sparse graph (MPS-safe gather)
        b, n_max, c = h.shape
        num_atoms_list = batch["num_atoms_list"]
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
            batch["edge_index_list"],
            batch["edge_dist_list"],
            batch["triplet_index_list"],
            batch["angles_list"],
            device=device,
        )
        edge_attr_rbf = self.edge_rbf(edge_dist)
        for layer in self.local_layers:
            flat_h = layer(flat_h, edge_index, edge_attr_rbf, triplet_index, angles)

        h_local_flat = torch.zeros(b * n_max, c, dtype=h.dtype, device=device)
        h_local_flat.index_copy_(0, flat_indices_t, flat_h)
        h_local = h_local_flat.reshape(b, n_max, c)

        # Compute geometry tensors used by every global layer
        frac_disp: Optional[torch.Tensor] = None
        dist_for_attn = dist_matrix
        if self.use_pfa and positions is not None and cell is not None:
            try:
                frac_disp = compute_frac_disp(positions, cell)
                if self.recompute_dist_from_positions:
                    cart = torch.einsum("bijk,bkl->bijl", frac_disp, cell)
                    dist_for_attn = torch.linalg.norm(cart, dim=-1)
            except RuntimeError:
                # singular cell — silently fall back to dist-only path
                frac_disp = None

        h_global = h_local
        for layer in self.global_layers:
            h_global = layer(h_global, dist_for_attn, frac_disp, defect_mask, mask)

        mask_f = mask.float().unsqueeze(-1)
        pooled = (h_global * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.readout(pooled).squeeze(-1)
