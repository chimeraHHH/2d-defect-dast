"""Phase A1 — per-atom physical descriptors from defect & pristine pairs.

For every IMP2D test sample (the ordered 1065 of cleaned_dataset_with_pristine.pkl)
we compute, per atom:

  * distance_to_defect      Å (minimum-image PBC)
  * bond_strain_max         max |Δd_ij / d_ref| over its first-shell bonds
  * bond_strain_mean        mean over first-shell bonds
  * angle_distortion_max    max |Δθ_jik| (radians) at this atom as centre
  * coord_change            Δ coordination number (defect minus pristine)
  * shell_index             0..5 for radial decay shells (0-3, 3-5, 5-7, 7-9, >9)

Per-sample (graph-level) we also compute:

  * delta_chi               |χ_dopant − ⟨χ⟩_host|  (electronegativity)
  * delta_rcov              |r_cov_dopant − ⟨r_cov⟩_host|
  * delta_valence           |valence_dopant − ⟨valence⟩_host|
  * defect_type_int         0 adsorbate, 1 interstitial
  * dopant_block            0 s, 1 p, 2 d, 3 f, 4 other
  * host_natoms             pristine atom count
  * mean_strain_first_shell mean |Δd_ij/d_ref| within 3 Å of defect
  * pct_atoms_strained      fraction of atoms with strain > 0.02

Output: results/phase_a_descriptors.npz + results/phase_a_descriptors_summary.json

Defect / pristine alignment: the pristine supercell is the defect supercell
with the dopant atom (defect_atom_index from build_pristine_pairs.py)
removed; the remaining N-1 atoms are aligned 1:1 with the corresponding
N-1 atoms of the defect supercell once the dopant is removed.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# atomic-property dictionaries: small, hard-coded so we don't depend on
# pubchempy / mendeleev over a network (the test set has 65 elements).
PAULING_CHI = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
    26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01,
    33: 2.18, 34: 2.55, 35: 2.96, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33,
    41: 1.6, 42: 2.16, 43: 1.9, 44: 2.2, 45: 2.28, 46: 2.20, 47: 1.93,
    48: 1.69, 49: 1.78, 50: 1.96, 51: 2.05, 52: 2.1, 53: 2.66, 55: 0.79,
    56: 0.89, 57: 1.10, 58: 1.12, 59: 1.13, 60: 1.14, 62: 1.17, 63: 1.20,
    64: 1.20, 65: 1.10, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.10,
    71: 1.27, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.20,
    78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02,
}
COVALENT_R = {  # Å, from CRC Handbook
    1: 0.31, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.50,
    26: 1.42, 27: 1.38, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20,
    33: 1.19, 34: 1.20, 35: 1.20, 37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75,
    41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45,
    48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39, 55: 2.44,
    56: 2.15, 57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01, 62: 1.98, 63: 1.98,
    64: 1.96, 65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87,
    71: 1.87, 72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41,
    78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48,
}
VALENCE_E = {
    1: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7,
    19: 1, 20: 2,
    21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10, 29: 11, 30: 12,
    31: 3, 32: 4, 33: 5, 34: 6, 35: 7,
    37: 1, 38: 2,
    39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10, 47: 11, 48: 12,
    49: 3, 50: 4, 51: 5, 52: 6, 53: 7,
    55: 1, 56: 2,
    57: 3, 58: 3, 59: 3, 60: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3,
    68: 3, 69: 3, 70: 3, 71: 3,
    72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10, 79: 11, 80: 12,
    81: 3, 82: 4, 83: 5,
}


def block_of(z: int) -> int:
    """0=s, 1=p, 2=d, 3=f, 4=other."""
    if z in (1, 2):
        return 0
    if 3 <= z <= 4 or 11 <= z <= 12 or 19 <= z <= 20 or 37 <= z <= 38 or 55 <= z <= 56:
        return 0
    if 5 <= z <= 10 or 13 <= z <= 18 or 31 <= z <= 36 or 49 <= z <= 54 or 81 <= z <= 86:
        return 1
    if 21 <= z <= 30 or 39 <= z <= 48 or 72 <= z <= 80:
        return 2
    if 57 <= z <= 71 or 89 <= z <= 103:
        return 3
    return 4


def pbc_distance_matrix(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """N x N minimum-image PBC distance, vectorised."""
    diff = positions[:, None, :] - positions[None, :, :]
    cell_inv = np.linalg.inv(cell)
    diff_frac = diff @ cell_inv
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ cell
    return np.linalg.norm(diff_cart, axis=-1)


def pbc_distance_def_to_pri(
    pos_def: np.ndarray, pos_pri: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """N_def × N_pri minimum-image distance."""
    diff = pos_def[:, None, :] - pos_pri[None, :, :]
    cell_inv = np.linalg.inv(cell)
    diff_frac = diff @ cell_inv
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ cell
    return np.linalg.norm(diff_cart, axis=-1)


def first_shell_bonds(dist_def: np.ndarray, atom_idx: int,
                      cutoff: float = 3.0) -> List[int]:
    """Indices of atoms within `cutoff` of ``atom_idx`` (excluding self)."""
    d = dist_def[atom_idx]
    return [j for j, dj in enumerate(d) if 1e-3 < dj < cutoff]


def shell_index(d: float) -> int:
    if d < 3.0:
        return 0
    if d < 5.0:
        return 1
    if d < 7.0:
        return 2
    if d < 9.0:
        return 3
    return 4


def build_pair_reference_table(data, max_samples: int = 2000) -> Dict:
    """Build a data-driven equilibrium-bond-length lookup keyed on
    (Z_smaller, Z_larger): the median distance over all first-shell
    (within 3 Å) bonds observed across the dataset.

    Falls back to (covalent_radius(Z_a) + covalent_radius(Z_b)) for any
    pair not seen in the lookup.
    """
    from collections import defaultdict
    pair_dists = defaultdict(list)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(data), size=min(max_samples, len(data)), replace=False)
    for idx in sample_idx:
        s = data[int(idx)]
        nums = np.asarray(s["numbers"])
        pos = np.asarray(s["positions"], dtype=np.float64)
        cell = np.asarray(s["cell"], dtype=np.float64)
        d = pbc_distance_matrix(pos, cell)
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if 1.0 < d[i, j] < 3.0:
                    a, b = sorted((int(nums[i]), int(nums[j])))
                    pair_dists[(a, b)].append(float(d[i, j]))
    table = {}
    for k, v in pair_dists.items():
        if len(v) >= 5:
            table[k] = float(np.median(v))
    print(f"  bond-pair reference table: {len(table)} unique (Z_a, Z_b) pairs "
          f"from {len(sample_idx)} samples")
    return table


def reference_bond_length(z_a: int, z_b: int, table: Dict) -> float:
    """Look up data-driven equilibrium bond length, fallback to covalent."""
    key = (min(z_a, z_b), max(z_a, z_b))
    if key in table:
        return table[key]
    return COVALENT_R.get(z_a, 1.5) + COVALENT_R.get(z_b, 1.5)


def descriptors_per_sample(s: dict, ref_table: Dict = None) -> Dict:
    """Compute per-atom + per-sample descriptors. Returns a dict of np arrays
    plus per-sample scalars."""
    pos_def = s["positions"].astype(np.float64)
    cell = s["cell"].astype(np.float64)
    nums_def = np.asarray(s["numbers"], dtype=int)
    n_def = pos_def.shape[0]

    pristine = s.get("pristine")
    has_pristine = pristine is not None
    if has_pristine:
        pos_pri = pristine["positions"].astype(np.float64)
        nums_pri = np.asarray(pristine["numbers"], dtype=int)
    else:
        pos_pri = pos_def.copy()
        nums_pri = nums_def.copy()
    didx = int(s.get("defect_atom_index", -1))

    dist_def = pbc_distance_matrix(pos_def, cell)
    dist_pri = pbc_distance_matrix(pos_pri, cell) if has_pristine else dist_def

    # Defect → pristine atom alignment: pristine has N-1 atoms (dopant removed
    # at index didx). For non-defect atoms i in defect, the corresponding
    # pristine atom is i if i<didx, else i-1.
    if has_pristine and didx >= 0 and pos_pri.shape[0] == n_def - 1:
        def_to_pri = np.array(
            [-1 if i == didx else (i if i < didx else i - 1) for i in range(n_def)],
            dtype=int,
        )
    else:
        def_to_pri = -np.ones(n_def, dtype=int)

    distance_to_defect = (
        dist_def[didx] if 0 <= didx < n_def else np.zeros(n_def)
    )

    bond_strain_max = np.zeros(n_def)
    bond_strain_mean = np.zeros(n_def)
    coord_change = np.zeros(n_def)
    angle_distortion_max = np.zeros(n_def)

    for i in range(n_def):
        # First shell bonds in defect (within 3 Å)
        nbr = first_shell_bonds(dist_def, i, cutoff=3.0)
        coord_change[i] = len(nbr)
        if not nbr:
            continue

        # Bond strain relative to data-driven equilibrium (median pair distance)
        # — works without a relaxed pristine reference.
        if ref_table is not None:
            d_def_i = np.array([dist_def[i, j] for j in nbr])
            d_eq_i = np.array([
                reference_bond_length(int(nums_def[i]), int(nums_def[j]), ref_table)
                for j in nbr
            ])
            rel = (d_def_i - d_eq_i) / np.maximum(d_eq_i, 1e-3)
            bond_strain_max[i] = float(np.max(np.abs(rel)))
            bond_strain_mean[i] = float(np.mean(np.abs(rel)))

        # Pristine first-shell coordination at the same site (within 3 Å)
        # — coord change is INTEGER and unaffected by the unrelaxed pristine
        # reference (depends only on which atoms are present).
        i_pri = def_to_pri[i]
        if has_pristine and i_pri >= 0:
            nbr_pri_pri = [k for k in range(pos_pri.shape[0])
                           if k != i_pri and 1e-3 < dist_pri[i_pri, k] < 3.0]
            coord_change[i] = len(nbr) - len(nbr_pri_pri)
        # Angle distortion is omitted: with unrelaxed pristine
        # (= defect minus dopant), angles among kept atoms are identical
        # to defect angles by construction. A meaningful angle metric
        # would require DFT-relaxed pristine (Phase C work).

    shell_idx = np.array([shell_index(d) for d in distance_to_defect], dtype=int)

    # Per-sample scalars
    dopant = s.get("metadata", {}).get("dopant", "")
    host_metadata = s.get("metadata", {}).get("host", "")
    defect_type_str = s.get("metadata", {}).get("defecttype", "")
    defect_type_int = 1 if defect_type_str == "interstitial" else 0
    if 0 <= didx < n_def:
        z_dop = int(nums_def[didx])
        host_zs = np.delete(nums_def, didx) if has_pristine else nums_def
    else:
        z_dop = 0
        host_zs = nums_def

    chi_dop = PAULING_CHI.get(z_dop, np.nan)
    rcov_dop = COVALENT_R.get(z_dop, np.nan)
    val_dop = VALENCE_E.get(z_dop, np.nan)
    chi_host = np.nanmean([PAULING_CHI.get(int(z), np.nan) for z in host_zs])
    rcov_host = np.nanmean([COVALENT_R.get(int(z), np.nan) for z in host_zs])
    val_host = np.nanmean([VALENCE_E.get(int(z), np.nan) for z in host_zs])
    delta_chi = abs((chi_dop - chi_host)) if np.isfinite(chi_dop + chi_host) else 0.0
    delta_rcov = abs((rcov_dop - rcov_host)) if np.isfinite(rcov_dop + rcov_host) else 0.0
    delta_val = abs((val_dop - val_host)) if np.isfinite(val_dop + val_host) else 0.0
    dopant_block = block_of(z_dop)
    near = np.where(distance_to_defect < 3.0)[0]
    mean_strain_first_shell = (
        float(np.nanmean(bond_strain_mean[near])) if len(near) else 0.0
    )
    pct_atoms_strained = float((bond_strain_max > 0.02).mean())

    return {
        # per-atom arrays (N_def,)
        "distance_to_defect": distance_to_defect.astype(np.float32),
        "bond_strain_max": bond_strain_max.astype(np.float32),
        "bond_strain_mean": bond_strain_mean.astype(np.float32),
        "angle_distortion_max": angle_distortion_max.astype(np.float32),
        "coord_change": coord_change.astype(np.float32),
        "shell_index": shell_idx.astype(np.int32),
        # per-sample scalars
        "n_atoms": int(n_def),
        "defect_atom_index": didx,
        "target": float(s["target"]),
        "host": host_metadata,
        "dopant": dopant,
        "defect_type": defect_type_str,
        "defect_type_int": defect_type_int,
        "z_dopant": z_dop,
        "delta_chi": float(delta_chi),
        "delta_rcov": float(delta_rcov),
        "delta_valence": float(delta_val),
        "dopant_block": dopant_block,
        "host_natoms": int(pos_pri.shape[0]),
        "mean_strain_first_shell": mean_strain_first_shell,
        "pct_atoms_strained": pct_atoms_strained,
    }


def main():
    src = ROOT / "data" / "processed" / "cleaned_dataset_with_pristine.pkl"
    with open(src, "rb") as f:
        blob = pickle.load(f)
    data = blob["data"]
    print(f"loaded {len(data)} samples")

    # Build data-driven equilibrium bond-length lookup over the train fold
    # (NOT touching test, to keep this leak-free).
    import random
    rng = random.Random(42)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_train = int(0.8 * len(data))
    n_val = int(0.1 * len(data))
    train_idx = indices[:n_train]
    test_idx = indices[n_train + n_val:]

    print(f"building pair-bond reference from train fold ({len(train_idx)} samples) ...")
    ref_table = build_pair_reference_table(
        [data[i] for i in train_idx], max_samples=2000
    )
    print(f"test fold: {len(test_idx)} samples")

    rows = []
    flat_atom = []  # list of dicts, one per atom
    sample_offsets = [0]
    t0 = time.time()
    for k, idx in enumerate(test_idx):
        s = data[idx]
        d = descriptors_per_sample(s, ref_table=ref_table)
        rows.append({
            "sample_id": idx,
            "n_atoms": d["n_atoms"],
            "defect_atom_index": d["defect_atom_index"],
            "target": d["target"],
            "host": d["host"],
            "dopant": d["dopant"],
            "defect_type": d["defect_type"],
            "defect_type_int": d["defect_type_int"],
            "z_dopant": d["z_dopant"],
            "delta_chi": d["delta_chi"],
            "delta_rcov": d["delta_rcov"],
            "delta_valence": d["delta_valence"],
            "dopant_block": d["dopant_block"],
            "host_natoms": d["host_natoms"],
            "mean_strain_first_shell": d["mean_strain_first_shell"],
            "pct_atoms_strained": d["pct_atoms_strained"],
        })
        for j in range(d["n_atoms"]):
            flat_atom.append({
                "sample_id": idx,
                "atom_idx": j,
                "is_defect": int(j == d["defect_atom_index"]),
                "distance_to_defect": float(d["distance_to_defect"][j]),
                "bond_strain_max": float(d["bond_strain_max"][j]),
                "bond_strain_mean": float(d["bond_strain_mean"][j]),
                "angle_distortion_max": float(d["angle_distortion_max"][j]),
                "coord_change": float(d["coord_change"][j]),
                "shell_index": int(d["shell_index"][j]),
            })
        sample_offsets.append(sample_offsets[-1] + d["n_atoms"])
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(test_idx)}  ({time.time()-t0:.0f}s)", flush=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "phase_a_descriptors.npz"
    keys = list(flat_atom[0].keys())
    arrs = {k: np.array([a[k] for a in flat_atom]) for k in keys}
    arrs["sample_offsets"] = np.array(sample_offsets, dtype=np.int64)
    np.savez(out_path, **arrs)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB) "
          f"with {len(flat_atom)} atoms across {len(rows)} samples")

    sum_path = out_dir / "phase_a_descriptors_summary.json"
    with open(sum_path, "w") as f:
        json.dump({"n_test_samples": len(rows),
                   "n_total_atoms": len(flat_atom),
                   "per_sample": rows[:50],  # first 50 for inspection
                   "wall_min": (time.time() - t0) / 60.0}, f, indent=2)
    print(f"wrote {sum_path}")


if __name__ == "__main__":
    main()
