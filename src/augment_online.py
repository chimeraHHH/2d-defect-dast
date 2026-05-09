"""Online (on-the-fly) augmentation for crystal graph data.

Unlike static augmentation (pre-compute N× copies to disk), online augmentation
applies *fresh random transforms every epoch*, so the model never sees the same
perturbed structure twice. This eliminates the fixed-noise memorization problem
of static augmentation while giving effectively infinite data diversity.

Transforms applied per sample:
  1. Random SO(2) rotation — FREE: preserves all distances and angles exactly.
  2. Random Gaussian perturbation — O(E + N²): recomputes edge_dist and
     dist_matrix from perturbed positions + stored edge_offset. Angles are
     approximately preserved for σ ≤ 0.05 Å (δθ < 0.5°).
  3. Random in-plane biaxial strain — same cost as perturbation.

Also provides an adversarial training helper that perturbs input features
in the direction that maximises loss (FGSM on x-space), forcing the model
to learn robust representations.
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


def adversarial_perturbation(
    model,
    batch: Dict[str, torch.Tensor],
    criterion,
    target_norm: torch.Tensor,
    eps: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FGSM-style adversarial perturbation on input features.

    Returns (preds_clean, adv_loss) where adv_loss is the consistency loss
    between clean and adversarial predictions.
    """
    x = batch["x"].detach().clone().requires_grad_(True)
    batch_for_grad = {k: v for k, v in batch.items()}
    batch_for_grad["x"] = x

    preds_clean = model(batch_for_grad)
    loss = criterion(preds_clean, target_norm)
    grad_x = torch.autograd.grad(loss, x, retain_graph=True)[0]

    x_adv = (x + eps * grad_x.sign()).detach()
    batch_adv = {k: v for k, v in batch.items()}
    batch_adv["x"] = x_adv
    preds_adv = model(batch_adv)

    adv_loss = F.mse_loss(preds_clean, preds_adv)
    return preds_clean, loss, adv_loss
