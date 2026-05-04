"""Build (defect, pristine) graph pairs for the dual-stream architecture.

For every IMP2D defect sample we construct an *unrelaxed* pristine reference
by removing the dopant atom from the defect supercell. The IMP2D defects are
either ``adsorbate`` (dopant added on top of pristine host) or ``interstitial``
(dopant added inside the host); in both cases removing the dopant atom from
the supercell positions yields the host pristine supercell at the host's
relaxed coordinates (modulo small local relaxation perturbations from the
defect, which we accept as the cost of the unrelaxed approximation).

The dopant atom is identified by the same heuristic used in
``CrystalGraphDataset._compute_defect_mask``: the LAST atom in
``numbers`` whose atomic number matches ``metadata["dopant"]``. For
IMP2D defect types (adsorbate / interstitial), that last-matching index
is consistently the inserted atom.

Output: ``data/processed/cleaned_dataset_with_pristine.pkl`` containing a
``data`` list where each entry has the original defect fields plus a
``pristine`` sub-dict carrying the freshly-built host-supercell graph
(numbers, positions, cell, edge_index, edge_dist, edge_offset,
triplet_index, angles, dist_matrix).
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as _AZ

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph  # noqa: E402


def find_defect_index(sample) -> int | None:
    """Index of the dopant atom in sample['numbers'], or None if undetermined.

    Mirrors ``CrystalGraphDataset._compute_defect_mask`` so the pristine we
    construct is exactly the host structure that the existing defect_mask
    flags.
    """
    dopant = sample["metadata"].get("dopant", "")
    if not dopant:
        return None
    z = _AZ.get(dopant, None)
    if z is None:
        return None
    candidates = np.flatnonzero(sample["numbers"] == z)
    if candidates.size == 0:
        return None
    return int(candidates[-1])


def build_pristine_pair(sample) -> dict | None:
    """Return the pristine sub-graph dict to pair with this defect sample."""
    didx = find_defect_index(sample)
    if didx is None:
        return None
    keep = np.ones(len(sample["numbers"]), dtype=bool)
    keep[didx] = False
    pristine_numbers = sample["numbers"][keep]
    pristine_positions = sample["positions"][keep]
    cell = sample["cell"]
    if pristine_numbers.size < 2:
        return None  # not enough atoms left to form a graph
    atoms = Atoms(
        numbers=pristine_numbers,
        positions=pristine_positions,
        cell=cell,
        pbc=True,
    )
    g = build_graph(atoms, cutoff=5.0)
    return {
        "numbers": g["numbers"],
        "positions": g["positions"],
        "cell": g["cell"],
        "edge_index": g["edge_index"],
        "edge_dist": g["edge_dist"],
        "edge_offset": g["edge_offset"],
        "triplet_index": g["triplet_index"],
        "angles": g["angles"],
        "dist_matrix": g["dist_matrix"],
    }


def main():
    src = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
    dst = ROOT / "data" / "processed" / "cleaned_dataset_with_pristine.pkl"

    print(f"Loading {src} ...")
    with open(src, "rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and "data" in blob:
        data = blob["data"]
        meta = blob.get("meta", {})
    else:
        data = blob
        meta = {}
    print(f"  total samples: {len(data)}")

    t0 = time.time()
    out_data = []
    skipped = 0
    self_substitution = 0
    for i, s in enumerate(data):
        pristine = build_pristine_pair(s)
        if pristine is None:
            skipped += 1
            continue
        # detect self-substitution: dopant Z == any host atom Z
        dop = s["metadata"].get("dopant", "")
        host_zs = set(s["numbers"][np.flatnonzero(s["numbers"] != _AZ.get(dop, 0))])
        if _AZ.get(dop, 0) in {int(z) for z in s["numbers"][:-1]}:
            self_substitution += 1
        new_sample = dict(s)
        new_sample["pristine"] = pristine
        new_sample["defect_atom_index"] = int(find_defect_index(s))
        out_data.append(new_sample)
        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(data)}  ({time.time()-t0:.1f}s, skipped={skipped})")

    print(f"\nDone. Built pairs for {len(out_data)} / {len(data)} samples "
          f"(skipped {skipped}, self-substitution flagged {self_substitution})")

    out_meta = {**meta, "version": "with_pristine_v1",
                "n_total": len(out_data),
                "skipped_no_dopant": skipped,
                "self_substitution_count": self_substitution,
                "build_time_min": (time.time() - t0) / 60.0}
    with open(dst, "wb") as f:
        pickle.dump({"data": out_data, "meta": out_meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
