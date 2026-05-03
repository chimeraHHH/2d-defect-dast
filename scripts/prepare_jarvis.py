"""Download JARVIS-DFT vacancydb and convert to CrystalGraphDataset format.

Produces data/processed/jarvis_2d.pkl and jarvis_3d.pkl with the same sample
dict schema as cleaned_dataset.pkl so they can be loaded directly by
CrystalGraphDataset.

JARVIS vacancies differ from IMP2D impurities in three key ways:
  1. Defect type: vacancies (atom removal) vs interstitials/adsorbates (atom addition)
  2. DFT code: VASP + OptB88vdW vs GPAW + PBE
  3. Defect identification: JARVIS stores the removed atom symbol; we mark
     the vacancy site heuristically via the atom closest to the missing position.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph


def jarvis_atoms_to_ase(jatoms: dict) -> Atoms:
    """Convert JARVIS atoms dict to ASE Atoms."""
    lattice = np.array(jatoms["lattice_mat"])
    coords = np.array(jatoms["coords"])
    elements = jatoms["elements"]
    cartesian = jatoms.get("cartesian", False)

    if cartesian:
        return Atoms(symbols=elements, positions=coords, cell=lattice, pbc=True)
    else:
        return Atoms(symbols=elements, scaled_positions=coords, cell=lattice, pbc=True)


def find_vacancy_site(bulk_atoms: Atoms, defective_atoms: Atoms) -> int:
    """Find the atom index in defective_atoms closest to the missing vacancy site.

    Compare bulk vs defective positions to find the removed atom's position,
    then return the index of the nearest remaining atom as the "defect site".
    This is the best proxy for vacancy location in a structure where the atom
    is absent.
    """
    bulk_pos = bulk_atoms.get_positions()
    def_pos = defective_atoms.get_positions()
    cell = np.array(bulk_atoms.get_cell())

    used = set()
    for i in range(len(def_pos)):
        diff = bulk_pos - def_pos[i]
        try:
            cell_inv = np.linalg.inv(cell)
            diff_frac = diff @ cell_inv
            diff_frac -= np.round(diff_frac)
            diff_cart = diff_frac @ cell
        except np.linalg.LinAlgError:
            diff_cart = diff
        dists = np.linalg.norm(diff_cart, axis=1)
        for _ in range(len(bulk_pos)):
            idx = int(np.argmin(dists))
            if idx not in used:
                used.add(idx)
                break
            dists[idx] = 1e9

    missing_indices = set(range(len(bulk_pos))) - used
    if not missing_indices:
        return 0

    missing_idx = list(missing_indices)[0]
    missing_pos = bulk_pos[missing_idx]

    diff = def_pos - missing_pos
    try:
        cell_inv = np.linalg.inv(cell)
        diff_frac = diff @ cell_inv
        diff_frac -= np.round(diff_frac)
        diff_cart = diff_frac @ cell
    except np.linalg.LinAlgError:
        diff_cart = diff
    dists = np.linalg.norm(diff_cart, axis=1)
    return int(np.argmin(dists))


def main():
    from jarvis.db.figshare import data

    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading JARVIS vacancydb...")
    entries = data("vacancydb")
    print(f"Total entries: {len(entries)}")

    for mat_type, out_name in [("2D", "jarvis_2d.pkl"), ("3D", "jarvis_3d.pkl")]:
        subset = [e for e in entries if e["material_type"] == mat_type]
        print(f"\n{'='*60}")
        print(f"Processing {mat_type}: {len(subset)} entries -> {out_name}")

        samples = []
        skipped = {"graph_error": 0, "ef_missing": 0}
        t0 = time.time()

        for i, entry in enumerate(subset):
            ef = entry.get("ef")
            if ef is None or np.isnan(ef):
                skipped["ef_missing"] += 1
                continue

            try:
                bulk_ase = jarvis_atoms_to_ase(entry["bulk_atoms"])
                def_ase = jarvis_atoms_to_ase(entry["defective_atoms"])
                g = build_graph(def_ase, cutoff=5.0)
            except Exception as exc:
                skipped["graph_error"] += 1
                print(f"  graph error on {entry['id']}: {exc}")
                continue

            n_atoms = len(def_ase)
            defect_idx = find_vacancy_site(bulk_ase, def_ase)
            defect_mask = np.zeros(n_atoms, dtype=np.int64)
            defect_mask[defect_idx] = 1

            sample = {
                "id": i,
                "unique_id": entry["id"],
                "numbers": g["numbers"],
                "positions": g["positions"],
                "cell": g["cell"],
                "edge_index": g["edge_index"],
                "edge_dist": g["edge_dist"],
                "edge_offset": g["edge_offset"],
                "triplet_index": g["triplet_index"],
                "angles": g["angles"],
                "dist_matrix": g["dist_matrix"],
                "defect_mask": defect_mask,
                "target": float(ef),
                "metadata": {
                    "host": entry["bulk_formula"],
                    "dopant": entry["symbol"],
                    "site": entry.get("wycoff", ""),
                    "defecttype": "vacancy",
                    "natoms": n_atoms,
                    "spacegroup": "",
                    "supercell": "",
                    "source": "JARVIS-DFT",
                    "jid": entry["jid"],
                },
            }
            samples.append(sample)

            if (i + 1) % 20 == 0:
                print(f"  processed {i+1}/{len(subset)}, kept {len(samples)}")

        dt = time.time() - t0
        print(f"Finished {mat_type} in {dt:.1f}s. Kept {len(samples)}, skipped {skipped}")

        out_path = out_dir / out_name
        with open(out_path, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

        ef_vals = [s["target"] for s in samples]
        print(f"  Ef range: [{min(ef_vals):.3f}, {max(ef_vals):.3f}] eV")
        print(f"  Ef mean: {np.mean(ef_vals):.3f} ± {np.std(ef_vals):.3f} eV")
        n_atoms_list = [s["metadata"]["natoms"] for s in samples]
        print(f"  Atom count range: [{min(n_atoms_list)}, {max(n_atoms_list)}]")
        hosts = set(s["metadata"]["host"] for s in samples)
        print(f"  Unique hosts: {len(hosts)}")


if __name__ == "__main__":
    main()
