"""Element-wise physical/chemical descriptors used as initial node features.

Nine descriptors are extracted for atomic numbers Z = 1..100:
  group, period, electronegativity (Pauling), covalent radius, van-der-Waals
  radius, valence electrons, first ionisation energy, electron affinity, atomic
  mass.

Values come from the standard tables shipped with ASE; missing entries are
imputed with the column median so that the lookup table always returns a
well-defined 9-vector. Each descriptor is z-score normalised across elements
to keep magnitudes comparable.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ase.data import (
    atomic_masses,
    covalent_radii,
    vdw_radii,
)

# Pauling electronegativity (Z=0 placeholder + 1..100). Source: pymatgen / Pauling.
PAULING_EN = [
    0.00,
    2.20, 0.00, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.00,
    0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0.00, 0.82, 1.00,
    1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65,
    1.81, 2.01, 2.18, 2.55, 2.96, 3.00, 0.82, 0.95, 1.22, 1.33,
    1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96,
    2.05, 2.10, 2.66, 2.60, 0.79, 0.89, 1.10, 1.12, 1.13, 1.14,
    1.13, 1.17, 1.20, 1.20, 1.10, 1.22, 1.23, 1.24, 1.25, 1.10,
    1.27, 1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00,
    1.62, 2.33, 2.02, 2.00, 2.20, 0.00, 0.70, 0.90, 1.10, 1.30,
    1.50, 1.38, 1.36, 1.28, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30,
]

# First ionisation energy (eV). 0..100 with 0 placeholder.
IONIZATION_ENERGY = [
    0.00,
    13.60, 24.59, 5.39, 9.32, 8.30, 11.26, 14.53, 13.62, 17.42, 21.56,
    5.14, 7.65, 5.99, 8.15, 10.49, 10.36, 12.97, 15.76, 4.34, 6.11,
    6.56, 6.83, 6.75, 6.77, 7.43, 7.90, 7.88, 7.64, 7.73, 9.39,
    6.00, 7.90, 9.79, 9.75, 11.81, 14.00, 4.18, 5.69, 6.22, 6.63,
    6.76, 7.09, 7.28, 7.36, 7.46, 8.34, 7.58, 8.99, 5.79, 7.34,
    8.61, 9.01, 10.45, 12.13, 3.89, 5.21, 5.58, 5.54, 5.46, 5.53,
    5.55, 5.64, 5.67, 6.15, 5.86, 5.94, 6.02, 6.11, 6.18, 6.25,
    5.43, 6.83, 7.55, 7.86, 7.83, 8.44, 8.97, 8.96, 9.23, 10.44,
    6.11, 7.42, 7.29, 8.41, 9.32, 10.75, 4.07, 5.28, 5.17, 6.31,
    5.89, 6.19, 6.27, 6.03, 5.97, 6.20, 6.27, 6.06, 6.50, 6.58,
]

# Group (1..18 main + lanthanide/actinide assigned to group 3, 0 if unknown).
GROUP = [
    0,
    1, 18, 1, 2, 13, 14, 15, 16, 17, 18,
    1, 2, 13, 14, 15, 16, 17, 18, 1, 2,
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 1, 2, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 1, 2, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
]

# Period number (1..7).
PERIOD = [
    0,
    1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
]

# Number of valence electrons (rough, by group).
def _valence_for_group(g: int) -> int:
    if g == 0:
        return 0
    if g <= 2:
        return g
    if g >= 13:
        return g - 10
    return g  # transition metals: use group number


# Electron affinity (eV). Many missing values - using common-table approximations.
ELECTRON_AFFINITY = [
    0.00,
    0.75, 0.00, 0.62, 0.00, 0.28, 1.26, -0.07, 1.46, 3.40, 0.00,
    0.55, 0.00, 0.43, 1.39, 0.75, 2.08, 3.61, 0.00, 0.50, 0.02,
    0.19, 0.08, 0.53, 0.67, 0.00, 0.16, 0.66, 1.16, 1.24, 0.00,
    0.41, 1.23, 0.81, 2.02, 3.36, 0.00, 0.49, 0.05, 0.31, 0.43,
    0.92, 0.75, 0.55, 1.05, 1.14, 0.56, 1.30, 0.00, 0.40, 1.11,
    1.05, 1.97, 3.06, 0.00, 0.47, 0.14, 0.50, 0.50, 0.50, 0.50,
    0.50, 0.50, 0.50, 0.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
    0.50, 0.00, 0.32, 0.81, 0.15, 1.10, 1.56, 2.13, 2.31, 0.00,
    0.38, 0.36, 0.94, 1.91, 2.80, 0.00, 0.50, 0.00, 0.50, 0.00,
    0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
]


def _build_feature_table(max_z: int = 100) -> np.ndarray:
    """Return a (max_z+1, 9) array indexed by atomic number."""
    feat = np.zeros((max_z + 1, 9), dtype=np.float32)
    for z in range(1, max_z + 1):
        feat[z, 0] = GROUP[z]
        feat[z, 1] = PERIOD[z]
        feat[z, 2] = PAULING_EN[z]
        feat[z, 3] = covalent_radii[z]
        feat[z, 4] = vdw_radii[z] if not math.isnan(vdw_radii[z]) else covalent_radii[z]
        feat[z, 5] = _valence_for_group(GROUP[z])
        feat[z, 6] = IONIZATION_ENERGY[z]
        feat[z, 7] = ELECTRON_AFFINITY[z]
        feat[z, 8] = atomic_masses[z]
    # column-wise z-score normalisation over Z=1..max_z (skip 0 padding row)
    valid = feat[1:]
    mu = valid.mean(axis=0, keepdims=True)
    sigma = valid.std(axis=0, keepdims=True) + 1e-6
    feat[1:] = (valid - mu) / sigma
    return feat


_TABLE: Optional[torch.Tensor] = None


def get_atom_feature_table(path: Optional[Path] = None) -> torch.Tensor:
    """Lazy-load (or build) the (101, 9) feature lookup table as a torch tensor.

    Calling this with the same ``path`` that already exists returns the cached
    file; otherwise the table is built from ASE / Pauling tables and saved.
    """
    global _TABLE
    if path is not None and Path(path).exists():
        _TABLE = torch.load(path, map_location="cpu", weights_only=False)
        return _TABLE
    if _TABLE is not None:
        return _TABLE
    arr = _build_feature_table(max_z=100)
    _TABLE = torch.from_numpy(arr).float()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(_TABLE, path)
    return _TABLE


if __name__ == "__main__":  # quick smoke test
    table = get_atom_feature_table()
    print("Feature table shape:", tuple(table.shape))
    print("Mean (excluding pad row):", table[1:].mean(0))
    print("Std (excluding pad row):", table[1:].std(0))
