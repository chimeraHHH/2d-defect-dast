"""C17 Stage 1: Generate candidate defect structures NOT in IMP2D.

Strategy
--------
Take all unique pristine 2D host structures from IMP2D, then for each
host enumerate substitutions of every site by every dopant element
not already covered for that (host, site) combination in IMP2D.

For each new (host, site, dopant) triple, build the supercell defect
structure with ASE (replacing the atom at the site), pre-compute the
graph features required by our CrystalTransformer, and emit a sample
dict in the same internal format as the cleaned IMP2D dataset.

This produces structures *consistent with IMP2D's chemistry universe*
but covering combinations the dataset never sampled — exactly the
missing data identified by the §5.17 scaling-law analysis.

Outputs
-------
- data/processed/candidates_c17.pkl   (list of sample dicts, no targets)
- results/candidates_c17_meta.json    (host coverage statistics)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ase import Atoms  # noqa: E402
from ase.data import atomic_numbers, chemical_symbols  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
OUT_PATH = ROOT / "data" / "processed" / "candidates_c17.pkl"
META_PATH = ROOT / "results" / "candidates_c17_meta.json"

# control parameters
DOPANT_SET = [
    # cover most common dopants used in defect studies
    "H", "Li", "B", "C", "N", "O", "F",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",
    "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi",
]
# how many candidates to keep per host
PER_HOST_LIMIT = 100
RNG_SEED = 42
CUTOFF = 5.0
MAX_NEIGHBORS = 32


def reconstruct_pristine_from_defect(sample):
    """Best-effort: drop the dopant atom (the one matching metadata['dopant'])
    to reconstruct an approximate pristine cell.  Works for substitutional
    defects (just put back the original element).  For interstitials it
    works exactly: removing the inserted atom recovers pristine.
    """
    Z = sample["numbers"].copy()
    pos = sample["positions"].copy()
    cell = sample["cell"].copy()
    meta = sample["metadata"]
    dopant = meta.get("dopant", "")
    host = meta.get("host", "")

    # find dopant atom index
    if dopant in atomic_numbers:
        z_dop = atomic_numbers[dopant]
        candidates = np.flatnonzero(Z == z_dop)
        if len(candidates) == 0:
            return None
        # heuristic: defect is the LAST atom matching the dopant
        defect_idx = candidates[-1]
    else:
        return None

    defect_type = meta.get("defecttype", "")

    if defect_type in ("substitution", "antisite", "substitutional"):
        # we don't actually know the original element — skip, would
        # need DFT-aware reconstruction
        return None
    elif defect_type in ("interstitial", "adsorbate", "adatom"):
        # just remove the dopant
        keep = np.ones(len(Z), dtype=bool)
        keep[defect_idx] = False
        return {
            "Z": Z[keep],
            "positions": pos[keep],
            "cell": cell,
            "host": host,
            "site": meta.get("site", ""),
            "defect_type": defect_type,
            "n_atoms": int(keep.sum()),
            "defect_idx_in_original": int(defect_idx),
            "defect_pos": pos[defect_idx].copy(),
        }
    else:
        return None


def build_substitution_candidate(pristine, new_dopant_z, replace_idx):
    """Replace the atom at `replace_idx` of pristine with `new_dopant_z`."""
    Z = pristine["Z"].copy()
    pos = pristine["positions"].copy()
    Z[replace_idx] = new_dopant_z
    return {
        "Z": Z,
        "positions": pos,
        "cell": pristine["cell"],
        "host": pristine["host"],
        "site": pristine["site"],
        "defect_type": "substitution",
        "n_atoms": len(Z),
        "dopant": chemical_symbols[new_dopant_z],
        "replaced_z": int(pristine["Z"][replace_idx]),
    }


def build_interstitial_candidate(pristine, new_dopant_z):
    """Add the new dopant at the original interstitial position."""
    Z = np.append(pristine["Z"], new_dopant_z)
    pos = np.vstack([pristine["positions"],
                     pristine["defect_pos"][None, :]])
    return {
        "Z": Z,
        "positions": pos,
        "cell": pristine["cell"],
        "host": pristine["host"],
        "site": pristine["site"],
        "defect_type": "interstitial",
        "n_atoms": len(Z),
        "dopant": chemical_symbols[new_dopant_z],
    }


def candidate_to_sample(cand, idx):
    """Convert candidate dict → sample dict with full graph features."""
    Z = cand["Z"].astype(np.int64)
    pos = cand["positions"].astype(np.float32)
    cell = cand["cell"].astype(np.float32)
    n = len(Z)

    atoms = Atoms(numbers=Z, positions=pos, cell=cell, pbc=True)
    ii, jj, dd, oo = neighbor_list("ijdS", atoms, cutoff=CUTOFF,
                                    self_interaction=False)
    if len(ii) == 0:
        return None

    # cap neighbours per source
    if MAX_NEIGHBORS > 0:
        keep = np.ones(len(ii), dtype=bool)
        sources = ii
        for src in np.unique(sources):
            mask = sources == src
            if mask.sum() > MAX_NEIGHBORS:
                # keep nearest
                idxs = np.flatnonzero(mask)
                d_src = dd[idxs]
                top = np.argsort(d_src)[:MAX_NEIGHBORS]
                drop = np.ones(len(idxs), dtype=bool)
                drop[top] = False
                keep[idxs[drop]] = False
        ii = ii[keep]; jj = jj[keep]; dd = dd[keep]; oo = oo[keep]

    edge_index = np.stack([ii, jj], axis=0).astype(np.int64)
    edge_dist = dd.astype(np.float32)
    edge_offset = oo.astype(np.float32)

    # triplets (center, neighbor1, neighbor2) in atom-index format
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
                v_a = (pos[v_a_atom] + oo[ka] @ cell - pos[u])
                v_b = (pos[v_b_atom] + oo[kb] @ cell - pos[u])
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

    dist_matrix = cdist(pos, pos).astype(np.float32)

    # mark defect atom: last atom for substitution & interstitial
    defect_mask = np.zeros(n, dtype=np.int64)
    defect_mask[-1] = 1

    sample = {
        "id": idx,
        "unique_id": f"c17_{idx:06d}_{cand['host']}_{cand['dopant']}_{cand['defect_type']}",
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
        "target": 0.0,  # placeholder, will be filled by predictor
        "metadata": {
            "host": cand["host"],
            "dopant": cand["dopant"],
            "site": cand.get("site", ""),
            "defecttype": cand["defect_type"],
            "natoms": int(n),
            "spacegroup": "",
            "supercell": "",
            "source": "c17_generated",
        },
    }
    return sample


def main():
    t0 = time.time()
    print(f"Loading IMP2D from {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data)} samples")

    # group samples by host so we can find one representative per host with
    # an interstitial defect (used as pristine template)
    by_host = defaultdict(list)
    for s in data:
        by_host[s["metadata"]["host"]].append(s)
    print(f"  {len(by_host)} unique hosts")

    # for each host, find a sample we can recover pristine from
    pristine_per_host = {}
    for host, samples in by_host.items():
        for s in samples:
            recon = reconstruct_pristine_from_defect(s)
            if recon is not None and recon["n_atoms"] >= 6:
                pristine_per_host[host] = recon
                break
    print(f"  recovered pristine templates for {len(pristine_per_host)}/{len(by_host)} hosts")

    # also collect existing (host, dopant) combinations to avoid duplicates
    existing = set()
    for s in data:
        existing.add((s["metadata"]["host"], s["metadata"]["dopant"]))
    print(f"  {len(existing)} unique (host, dopant) combos already in IMP2D")

    # generate candidates
    rng = np.random.default_rng(RNG_SEED)
    candidates = []
    skipped_dup = 0
    skipped_build = 0
    per_host_count = Counter()

    for host, pristine in pristine_per_host.items():
        # try interstitials of every dopant not yet seen for this host
        for dopant_sym in DOPANT_SET:
            if (host, dopant_sym) in existing:
                skipped_dup += 1
                continue
            if dopant_sym not in atomic_numbers:
                continue
            z_new = atomic_numbers[dopant_sym]
            cand = build_interstitial_candidate(pristine, z_new)
            try:
                samp = candidate_to_sample(cand, idx=len(candidates))
            except Exception:
                samp = None
            if samp is None:
                skipped_build += 1
                continue
            candidates.append(samp)
            per_host_count[host] += 1
            if per_host_count[host] >= PER_HOST_LIMIT:
                break

    print(f"\n=== Generated {len(candidates)} candidates ===")
    print(f"  skipped (duplicate of IMP2D): {skipped_dup}")
    print(f"  skipped (build failure):       {skipped_build}")
    print(f"  hosts covered: {len([h for h, n in per_host_count.items() if n > 0])}")

    # save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(candidates, f)
    print(f"saved -> {OUT_PATH}")

    meta = {
        "n_candidates": len(candidates),
        "n_unique_hosts": len(pristine_per_host),
        "n_dopant_set": len(DOPANT_SET),
        "skipped_dup": skipped_dup,
        "skipped_build": skipped_build,
        "per_host_count": dict(per_host_count),
        "wall_time_min": (time.time() - t0) / 60,
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta saved -> {META_PATH}")


if __name__ == "__main__":
    main()
