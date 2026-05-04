"""Fetch JARVIS dft_3d (~75k pristine 3D materials) for backbone enrichment.

This provides massive structural diversity that the IMP2D-trained model
has never seen. Even if the prediction target (pristine formation energy
per atom) differs from defect formation energy, the SHARED backbone
should benefit from exposure to ~10× more chemistry/structure space.

Outputs
-------
- data/processed/dft_3d_lite.pkl  (subsampled to ~20k for tractable training)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ase import Atoms  # noqa: E402
from ase.data import atomic_numbers  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

OUT_PATH = ROOT / "data" / "processed" / "dft_3d_lite.pkl"

CUTOFF = 5.0
MAX_NEIGHBORS = 32
MAX_NATOMS = 80    # skip huge supercells to keep tractable
TARGET_N = 20000   # subsample to ~20k for fast training
SEED = 42


def convert(entry, idx):
    """Convert one JARVIS dft_3d entry → internal sample dict."""
    if entry.get("formation_energy_peratom") is None:
        return None

    atoms = entry["atoms"]
    elements = atoms["elements"]
    coords = np.asarray(atoms["coords"], dtype=np.float32)
    cell = np.asarray(atoms["lattice_mat"], dtype=np.float32)
    if not atoms.get("cartesian", False):
        coords = coords @ cell

    Z = np.array([atomic_numbers[e] for e in elements], dtype=np.int64)
    n = len(Z)
    if n < 2 or n > MAX_NATOMS:
        return None
    if (Z < 1).any() or (Z > 100).any():
        return None

    ase_atoms = Atoms(numbers=Z, positions=coords, cell=cell, pbc=True)
    try:
        ii, jj, dd, oo = neighbor_list("ijdS", ase_atoms, cutoff=CUTOFF,
                                        self_interaction=False)
    except Exception:
        return None
    if len(ii) == 0:
        return None

    if MAX_NEIGHBORS > 0:
        keep = np.ones(len(ii), dtype=bool)
        for src in np.unique(ii):
            mask = ii == src
            if mask.sum() > MAX_NEIGHBORS:
                idxs = np.flatnonzero(mask)
                top = np.argsort(dd[idxs])[:MAX_NEIGHBORS]
                drop = np.ones(len(idxs), dtype=bool)
                drop[top] = False
                keep[idxs[drop]] = False
        ii = ii[keep]; jj = jj[keep]; dd = dd[keep]; oo = oo[keep]

    edge_index = np.stack([ii, jj], axis=0).astype(np.int64)
    edge_dist = dd.astype(np.float32)
    edge_offset = oo.astype(np.float32)

    inc = defaultdict(list)
    for k, (u, v, d_, o_) in enumerate(zip(ii, jj, dd, oo)):
        inc[int(u)].append((int(v), int(k)))
    triplet_index = []
    angles = []
    for u, neigh_list in inc.items():
        if len(neigh_list) < 2:
            continue
        if len(neigh_list) > 6:
            chosen = list(np.random.default_rng(0).choice(
                len(neigh_list), 6, replace=False))
            neigh_list = [neigh_list[i] for i in chosen]
        for a in range(len(neigh_list)):
            for b in range(a + 1, len(neigh_list)):
                v_a_atom, ka = neigh_list[a]
                v_b_atom, kb = neigh_list[b]
                triplet_index.append([u, v_a_atom, v_b_atom])
                v_a = (coords[v_a_atom] + oo[ka] @ cell - coords[u])
                v_b = (coords[v_b_atom] + oo[kb] @ cell - coords[u])
                cos = float(np.dot(v_a, v_b)
                            / (np.linalg.norm(v_a) * np.linalg.norm(v_b)
                               + 1e-9))
                cos = max(-1.0, min(1.0, cos))
                angles.append(np.arccos(cos))
    if not triplet_index:
        triplet_index = np.zeros((0, 3), dtype=np.int64)
        angles = np.zeros((0,), dtype=np.float32)
    else:
        triplet_index = np.asarray(triplet_index, dtype=np.int64)
        angles = np.asarray(angles, dtype=np.float32)

    dist_matrix = cdist(coords, coords).astype(np.float32)

    sample = {
        "id": idx,
        "unique_id": entry.get("jid", str(idx)),
        "numbers": Z,
        "positions": coords.astype(np.float32),
        "cell": cell,
        "edge_index": edge_index,
        "edge_dist": edge_dist,
        "edge_offset": edge_offset,
        "triplet_index": triplet_index,
        "angles": angles,
        "dist_matrix": dist_matrix,
        # mark NO defect (pristine bulk)
        "defect_mask": np.zeros(n, dtype=np.int64),
        # target: formation energy PER ATOM
        "target": float(entry["formation_energy_peratom"]),
        "metadata": {
            "host": entry.get("formula", ""),
            "dopant": "",
            "site": "",
            "defecttype": "pristine",
            "natoms": int(n),
            "spacegroup": str(entry.get("spg_number", "1")),
            "supercell": "111",
            "source": "jarvis_dft_3d",
            "jid": entry.get("jid"),
        },
    }
    return sample


def main():
    t0 = time.time()
    print("Downloading JARVIS dft_3d (~75k pristine 3D materials)...")
    from jarvis.db.figshare import data
    raw = data("dft_3d")
    print(f"  got {len(raw)} entries; converting...")

    rng = np.random.default_rng(SEED)
    if len(raw) > TARGET_N:
        idx = rng.choice(len(raw), size=TARGET_N, replace=False)
        raw = [raw[i] for i in idx]
        print(f"  subsampled to {len(raw)} (TARGET_N={TARGET_N})")

    converted = []
    skip = 0
    for i, entry in enumerate(raw):
        s = convert(entry, idx=len(converted))
        if s is None:
            skip += 1
            continue
        converted.append(s)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(raw)}  kept {len(converted)}  skip {skip}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDone: {len(converted)} samples ({skip} skipped, {time.time()-t0:.0f}s)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(converted, f)
    print(f"saved -> {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
