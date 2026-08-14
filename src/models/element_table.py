"""Atomic-number lookup helpers for externally derived elemental tables."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def prepare_ct_uae_table(table: torch.Tensor) -> torch.Tensor:
    """Add the zero-padding row required for one-based atomic numbers."""
    if not isinstance(table, torch.Tensor) or table.ndim != 2:
        raise ValueError("ct-UAE asset must be a two-dimensional tensor")
    if table.shape[0] != 100:
        raise ValueError(
            f"ct-UAE asset must contain 100 element rows, observed {table.shape[0]}"
        )
    padding = table.new_zeros((1, table.shape[1]))
    return torch.cat((padding, table), dim=0)


def lookup_ct_uae(
    padded_table: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    """Map padding 0 to zero and atomic numbers 1--100 to rows 0--99."""
    if padded_table.ndim != 2 or padded_table.shape[0] != 101:
        raise ValueError("prepared ct-UAE table must have shape (101, features)")
    return F.embedding(
        atomic_numbers.to(dtype=torch.long),
        padded_table,
        padding_idx=0,
    )
