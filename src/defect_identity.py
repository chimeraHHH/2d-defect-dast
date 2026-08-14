"""Permutation-safe impurity identity helpers for IMP2D structures."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from ase.data import atomic_numbers


def dopant_candidate_indices(sample: Dict[str, Any]) -> np.ndarray:
    """Return atom indices whose element matches the sample dopant label."""
    dopant = str(sample.get("metadata", {}).get("dopant", ""))
    atomic_number = atomic_numbers.get(dopant)
    if atomic_number is None:
        return np.empty(0, dtype=np.int64)
    numbers = np.asarray(sample["numbers"], dtype=int)
    return np.flatnonzero(numbers == atomic_number).astype(np.int64, copy=False)


def unique_defect_index(sample: Dict[str, Any]) -> int:
    """Return the impurity index only when final composition identifies it uniquely.

    Identical nuclei cannot retain a physically meaningful identity tag through a
    relaxed structure. Samples whose dopant element also occurs in the host must
    therefore not be assigned an atom by array-order convention.
    """
    candidates = dopant_candidate_indices(sample)
    if len(candidates) != 1:
        sample_id = sample.get("id", sample.get("unique_id", "unknown"))
        raise ValueError(
            f"sample {sample_id} has {len(candidates)} atoms matching the dopant; "
            "a unique impurity identity cannot be inferred from the final structure"
        )
    return int(candidates[0])


def permutation_safe_defect_mask(sample: Dict[str, Any]) -> np.ndarray:
    """Mark a unique impurity, or return an all-zero mask when identity is ambiguous."""
    mask = np.zeros(len(sample["numbers"]), dtype=np.int64)
    candidates = dopant_candidate_indices(sample)
    if len(candidates) == 1:
        mask[int(candidates[0])] = 1
    return mask
