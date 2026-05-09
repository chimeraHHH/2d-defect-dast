"""Online (on-the-fly) augmentation and coordinate-space adversarial training.

Online augmentation applies *fresh random transforms every epoch*, so the model
never sees the same perturbed structure twice.

Transforms applied per sample:
  1. Random SO(2) rotation — FREE: preserves all distances and angles exactly.
  2. Random Gaussian perturbation — O(E + N²): recomputes edge_dist and
     dist_matrix from perturbed positions + stored edge_offset.
  3. Random in-plane biaxial strain — same cost as perturbation.

Adversarial training uses FGSM on atomic coordinates (not features):
  δpos = ε · sign(∂L/∂pos)
This finds the worst-case geometric perturbation — physically meaningful as it
corresponds to thermal vibrations / DFT relaxation uncertainty. Distance
computation is fully differentiable through PyTorch autograd.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class OnlineAugTransform:
    """Applies random geometric augmentations to a crystal graph sample."""

    def __init__(
        self,
        sigma_range: Tuple[float, float] = (0.01, 0.05),
        strain_range: float = 2.0,
        rotate_prob: float = 1.0,
        perturb_prob: float = 0.8,
        strain_prob: float = 0.3,
    ) -> None:
        self.sigma_range = sigma_range
        self.strain_range = strain_range
        self.rotate_prob = rotate_prob
        self.perturb_prob = perturb_prob
        self.strain_prob = strain_prob

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in sample.items()}

        pos = sample["positions"]
        cell = sample["cell"]
        changed = False

        # 1. Random SO(2) rotation (exact — distances and angles invariant)
        if random.random() < self.rotate_prob:
            angle = random.uniform(0, 2 * math.pi)
            c, s = math.cos(angle), math.sin(angle)
            R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]],
                             dtype=pos.dtype)
            pos = pos @ R.T
            cell = cell @ R.T
            sample["positions"] = pos
            sample["cell"] = cell
            # edge_dist, dist_matrix, angles all unchanged

        # 2. Random Gaussian perturbation
        if random.random() < self.perturb_prob:
            sigma = random.uniform(*self.sigma_range)
            noise = torch.randn_like(pos) * sigma
            pos = pos + noise
            sample["positions"] = pos
            changed = True

        # 3. Random in-plane biaxial strain
        if random.random() < self.strain_prob:
            strain_pct = random.uniform(-self.strain_range, self.strain_range)
            factor = 1.0 + strain_pct / 100.0
            pos = pos.clone()
            cell = cell.clone()
            pos[:, :2] *= factor
            cell[0, :2] *= factor
            cell[1, :2] *= factor
            sample["positions"] = pos
            sample["cell"] = cell
            changed = True

        # Recompute distance features if positions changed
        if changed:
            sample["edge_dist"] = _recompute_edge_dist(
                pos, sample["edge_index"], sample["edge_offset"], cell
            )
            sample["dist_matrix"] = _recompute_dist_matrix(pos, cell)

        return sample


def _recompute_edge_dist(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_offset: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Recompute edge distances from positions and stored periodic offsets."""
    i, j = edge_index[0], edge_index[1]
    offset_cart = edge_offset @ cell  # (E, 3)
    d_ij = positions[j] - positions[i] + offset_cart
    return d_ij.norm(dim=-1)


def _recompute_dist_matrix(
    positions: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Recompute N×N minimum-image distance matrix."""
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
    try:
        cell_inv = torch.linalg.inv(cell)
        diff_frac = diff @ cell_inv
        diff_frac = diff_frac - diff_frac.round()
        diff = diff_frac @ cell
    except torch.linalg.LinAlgError:
        pass
    return diff.norm(dim=-1)


class OnlineAugDataset(Dataset):
    """Wraps a dataset (or Subset) to apply online augmentation."""

    def __init__(self, base_dataset, transform: Optional[OnlineAugTransform] = None):
        self.base = base_dataset
        self.transform = transform
        # Expose attributes needed by HostBalancedSampler and Normalizer
        if hasattr(base_dataset, 'dataset'):
            self.data = base_dataset.dataset.data
            self.indices = base_dataset.indices
        elif hasattr(base_dataset, 'data'):
            self.data = base_dataset.data
            self.indices = list(range(len(base_dataset)))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def compute_dist_matrix_differentiable(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable N×N minimum-image distance matrix.

    Args:
        positions: (B, N_max, 3) atomic coordinates
        cell: (B, 3, 3) unit cell matrices
        atom_mask: (B, N_max) bool mask for valid atoms

    Returns:
        (B, N_max, N_max) distance matrix with grad through positions
    """
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 3)
    cell_inv = torch.linalg.inv(cell)  # (B, 3, 3)
    diff_frac = torch.einsum("bnmd,bdk->bnmk", diff, cell_inv)
    # round() has zero grad but acts as constant offset for small perturbations
    diff_frac = diff_frac - diff_frac.round()
    diff_mic = torch.einsum("bnmd,bdk->bnmk", diff_frac, cell)
    dist = diff_mic.norm(dim=-1)  # (B, N, N)
    mask_2d = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    return dist * mask_2d.float()


def compute_edge_dist_differentiable(
    positions: torch.Tensor,
    edge_index_list,
    edge_offset_list,
    cell: torch.Tensor,
    num_atoms_list,
) -> list:
    """Differentiable per-sample edge distances.

    Args:
        positions: (B, N_max, 3) atomic coordinates (requires_grad OK)
        edge_index_list: list of (2, E_b) tensors
        edge_offset_list: list of (E_b, 3) integer offset tensors
        cell: (B, 3, 3) unit cell matrices
        num_atoms_list: list of ints

    Returns:
        list of (E_b,) distance tensors with grad through positions
    """
    device = positions.device
    result = []
    for b in range(len(num_atoms_list)):
        n = num_atoms_list[b]
        pos_b = positions[b, :n]  # (n, 3)
        ei = edge_index_list[b].to(device)
        cell_b = cell[b]
        i, j = ei[0].long(), ei[1].long()
        if edge_offset_list:
            eo = edge_offset_list[b].to(device).float()
            offset_cart = eo @ cell_b  # (E, 3)
        else:
            offset_cart = 0.0
        d_ij = pos_b[j] - pos_b[i] + offset_cart
        result.append(d_ij.norm(dim=-1))
    return result


def adversarial_perturbation(
    model,
    batch: Dict[str, torch.Tensor],
    criterion,
    target_norm: torch.Tensor,
    eps: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate-space FGSM adversarial training (Madry-style robustness).

    Finds the atomic displacement δpos = ε·sign(∂L/∂pos) that maximally
    increases the loss, then trains the model to predict accurately even at
    that worst-case perturbation. Unlike consistency loss (which forces
    invariance and conflicts with the regression objective), this preserves
    the model's sensitivity to geometry while improving worst-case accuracy.

    Returns (preds_clean, task_loss, adv_robustness_loss).
    """
    positions = batch["positions"].detach().clone().requires_grad_(True)
    cell = batch["cell"]
    atom_mask = batch["atom_mask"]

    dist_matrix = compute_dist_matrix_differentiable(positions, cell, atom_mask)
    edge_dist_list = compute_edge_dist_differentiable(
        positions,
        batch["edge_index_list"],
        batch.get("edge_offset_list", []),
        cell,
        batch["num_atoms_list"],
    )

    batch_grad = {k: v for k, v in batch.items()}
    batch_grad["positions"] = positions
    batch_grad["dist_matrix"] = dist_matrix
    batch_grad["edge_dist_list"] = edge_dist_list

    preds_clean = model(batch_grad)
    loss = criterion(preds_clean, target_norm)
    grad_pos = torch.autograd.grad(loss, positions, retain_graph=True)[0]

    mask_3d = atom_mask.float().unsqueeze(-1)
    pos_adv = (positions + eps * grad_pos.sign() * mask_3d).detach()

    with torch.no_grad():
        dist_matrix_adv = compute_dist_matrix_differentiable(pos_adv, cell, atom_mask)
        edge_dist_list_adv = compute_edge_dist_differentiable(
            pos_adv,
            batch["edge_index_list"],
            batch.get("edge_offset_list", []),
            cell,
            batch["num_atoms_list"],
        )

    batch_adv = {k: v for k, v in batch.items()}
    batch_adv["positions"] = pos_adv
    batch_adv["dist_matrix"] = dist_matrix_adv
    batch_adv["edge_dist_list"] = edge_dist_list_adv

    preds_adv = model(batch_adv)
    adv_loss = criterion(preds_adv, target_norm)
    return preds_clean, loss, adv_loss
