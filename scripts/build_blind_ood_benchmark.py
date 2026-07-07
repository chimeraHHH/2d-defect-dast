"""Build the blind cross-code OOD benchmark dataset.

Merges the GPAW-relaxed defect structures (backup_data 2/sample_*_defect_*.traj,
batches 40-89: graphene C50 + VS2/CrS2 TMD hosts) with their DFT formation
energies (backup_data/dft_formation_energies.csv) into a single sample list
compatible with the PeriDefT / CrystalTransformer inference pipeline
(same format as data/processed/candidates_c17.pkl).

These 3 hosts (graphene, VS2, CrS2) are NOT among the 44 IMP2D hosts, and the
structures were generated independently of the model, making this a blind
host-OOD benchmark (cf. paper Sec. 5.7/5.8).

Inclusion rules
---------------
- usable == "yes"          -> included, flag "usable"
- usable == "needs_review" -> included, flag "needs_review" (#81, suspected
                              SCF high-energy state; excluded from headline
                              stats downstream)
- usable == "no"           -> excluded (missing host energy / mu / defect run)

Graph construction mirrors scripts/generate_candidates.py exactly:
cutoff 5.0 A, max 32 neighbours per source, <=6 neighbours per triplet centre,
dense distance matrix, defect_mask marks the dopant atom.

Output
------
- data/processed/blind_ood_benchmark.pkl
"""
from __future__ import annotations

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "backup_data" / "dft_formation_energies.csv"
TRAJ_DIR = ROOT / "backup_data 2"
OUT_PATH = ROOT / "data" / "processed" / "blind_ood_benchmark.pkl"

CUTOFF = 5.0
MAX_NEIGHBORS = 32


def find_traj(sample_id: int) -> tuple[Path, str] | None:
    """Prefer stage2 (final precision) trajectory, fall back to stage1."""
    for stage in ("s2", "s1"):
        p = TRAJ_DIR / f"sample_{sample_id}_defect_{stage}.traj"
        if p.exists():
            return p, stage
    return None


def atoms_to_sample(atoms: Atoms, idx: int, meta: dict, target: float) -> dict:
    """Convert ASE Atoms -> pipeline sample dict.

    Identical graph-construction logic to
    scripts/generate_candidates.py::candidate_to_sample.
    """
    Z = atoms.get_atomic_numbers().astype(np.int64)
    pos = atoms.get_positions().astype(np.float32)
    cell = np.asarray(atoms.get_cell()).astype(np.float32)
    n = len(Z)

    graph_atoms = Atoms(numbers=Z, positions=pos, cell=cell, pbc=True)
    ii, jj, dd, oo = neighbor_list(
        "ijdS", graph_atoms, cutoff=CUTOFF, self_interaction=False
    )
    if len(ii) == 0:
        raise RuntimeError(f"sample {meta['sample_id']}: empty neighbour list")

    # cap neighbours per source atom (keep nearest MAX_NEIGHBORS)
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
        ii, jj, dd, oo = ii[keep], jj[keep], dd[keep], oo[keep]

    edge_index = np.stack([ii, jj], axis=0).astype(np.int64)
    edge_dist = dd.astype(np.float32)
    edge_offset = oo.astype(np.float32)

    # triplets (center, n1, n2), <=6 neighbours per centre (rng seed 0)
    inc = defaultdict(list)
    for k, (u, v) in enumerate(zip(ii, jj)):
        inc[int(u)].append((int(v), int(k)))
    triplet_index, angles = [], []
    for u, neigh in inc.items():
        if len(neigh) < 2:
            continue
        if len(neigh) > 6:
            chosen = list(
                np.random.default_rng(0).choice(len(neigh), 6, replace=False)
            )
            neigh = [neigh[i] for i in chosen]
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                va_atom, ka = neigh[a]
                vb_atom, kb = neigh[b]
                triplet_index.append([u, va_atom, vb_atom])
                v_a = pos[va_atom] + oo[ka] @ cell - pos[u]
                v_b = pos[vb_atom] + oo[kb] @ cell - pos[u]
                cos = float(
                    np.dot(v_a, v_b)
                    / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-9)
                )
                angles.append(np.arccos(max(-1.0, min(1.0, cos))))
    if triplet_index:
        triplet_index = np.asarray(triplet_index, dtype=np.int64)
        angles = np.asarray(angles, dtype=np.float32)
    else:
        triplet_index = np.zeros((0, 3), dtype=np.int64)
        angles = np.zeros((0,), dtype=np.float32)

    dist_matrix = cdist(pos, pos).astype(np.float32)

    # dopant atom is the appended (last) atom in every GPAW build script;
    # verified against chemical symbols below in main().
    defect_mask = np.zeros(n, dtype=np.int64)
    defect_mask[-1] = 1

    return {
        "id": idx,
        "unique_id": (
            f"blind_{meta['sample_id']:03d}_{meta['host']}_"
            f"{meta['dopant']}_adsorbate"
        ),
        "numbers": Z,
        "positions": pos,
        "cell": cell,
        "edge_index": edge_index,
        "edge_dist": edge_dist,
        "edge_offset": edge_offset,
        "triplet_index": triplet_index,
        "angles": angles,
        "dist_matrix": dist_matrix,
        "defect_mask": defect_mask,
        "target": float(target),
        "metadata": meta,
    }


def main() -> None:
    from ase.data import chemical_symbols

    rows = list(csv.DictReader(open(CSV_PATH)))
    print(f"CSV rows: {len(rows)}")

    samples, skipped = [], []
    for row in rows:
        sid = int(row["sample_id"])
        usable = row["usable"].strip()
        if usable == "no":
            skipped.append((sid, f"unusable ({row['notes'] or row['status']})"))
            continue

        found = find_traj(sid)
        if found is None:
            skipped.append((sid, "no trajectory file"))
            continue
        traj_path, stage = found

        atoms = read(traj_path, -1)  # final relaxed frame
        syms = atoms.get_chemical_symbols()
        dopant = row["dopant"].strip()

        # --- integrity checks -------------------------------------------
        dop_idx = [i for i, s in enumerate(syms) if s == dopant]
        assert len(dop_idx) == 1, (
            f"sample {sid}: expected exactly 1 {dopant}, found {len(dop_idx)}"
        )
        assert dop_idx[0] == len(atoms) - 1, (
            f"sample {sid}: dopant not last atom (idx {dop_idx[0]})"
        )
        e_form = float(row["E_form_eV"])

        meta = {
            "sample_id": sid,
            "host": row["host"].strip(),
            "dopant": dopant,
            "formula": row["formula"].strip(),
            "defecttype": "adsorbate/interstitial",
            "natoms": len(atoms),
            "batch": row["batch"].strip(),
            "convergence": row["convergence"].strip(),
            "traj_stage": stage,
            "usable_flag": usable,          # "yes" | "needs_review"
            "notes": row["notes"].strip(),
            "E_form_dft_eV": e_form,
            "dft_code": "GPAW/PAW (PBE)",
            "source": "gpaw_blind_ood",
        }
        samples.append(atoms_to_sample(atoms, len(samples), meta, e_form))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(samples, f)

    # --- report ----------------------------------------------------------
    print(f"\nIncluded: {len(samples)}  |  skipped: {len(skipped)}")
    print(f"saved -> {OUT_PATH}\n")
    print(f"{'sid':>4} {'host':<9} {'dop':<3} {'formula':<11} "
          f"{'n':>3} {'stage':<5} {'flag':<12} {'E_dft':>8}")
    for s in samples:
        m = s["metadata"]
        print(f"{m['sample_id']:>4} {m['host']:<9} {m['dopant']:<3} "
              f"{m['formula']:<11} {m['natoms']:>3} {m['traj_stage']:<5} "
              f"{m['usable_flag']:<12} {m['E_form_dft_eV']:>8.4f}")
    print("\nSkipped:")
    for sid, why in skipped:
        print(f"  #{sid}: {why}")

    by_host = defaultdict(int)
    for s in samples:
        by_host[s["metadata"]["host"]] += 1
    print(f"\nPer host: {dict(by_host)}")
    n_review = sum(
        1 for s in samples if s["metadata"]["usable_flag"] == "needs_review"
    )
    print(f"usable: {len(samples) - n_review}, needs_review: {n_review}")


if __name__ == "__main__":
    sys.exit(main())
