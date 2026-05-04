"""C18b: Pristine-host DFT references for the C18 top-10 candidates.

To turn raw DFT total energies into formation energies Ef we need the
pristine-host total energy (same supercell, no dopant). This script
identifies the 5 unique hosts among the top-10 priority candidates,
strips the interstitial dopant, and runs single-point GPAW PW PBE
at the same settings used in dft_validation.py.

Outputs:
- results/dft_pristine.json  (E_total per host, with provenance)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ase import Atoms  # noqa: E402

CANDIDATES_PATH = ROOT / "data" / "processed" / "candidates_c17.pkl"
DFT_RESULTS = ROOT / "results" / "dft_validation.json"
RESULTS = ROOT / "results"
LOG_DIR = ROOT / "logs" / "dft_pristine"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PW_CUTOFF = 300
KPTS = (2, 2, 1)
XC = "PBE"


def strip_dopant(s, dopant_Z):
    """Remove the interstitial dopant atom from an internal sample dict.

    The interstitial dopant is the atom with the matching Z that is
    not part of the host stoichiometry. We assume there is exactly one
    such atom in each candidate (true for IMP2D interstitials).
    """
    Z = np.array(s["numbers"])
    pos = np.array(s["positions"])
    cell = np.array(s["cell"])
    matches = np.where(Z == dopant_Z)[0]
    assert len(matches) >= 1, f"no atom with Z={dopant_Z}"
    # remove the LAST one (interstitials are typically appended)
    keep = np.ones(len(Z), dtype=bool)
    keep[matches[-1]] = False
    return Z[keep], pos[keep], cell


def to_atoms(Z, pos, cell):
    cell = cell.copy()
    if cell[2, 2] < 12:
        cell[2, 2] = max(15.0, cell[2, 2] + 10.0)
    return Atoms(numbers=Z, positions=pos, cell=cell, pbc=True)


def run_dft(atoms, txt_path):
    from gpaw import GPAW, PW
    atoms.calc = GPAW(
        mode=PW(PW_CUTOFF), kpts=KPTS, xc=XC, txt=str(txt_path),
        symmetry={"point_group": False},
        convergence={"energy": 1e-3, "density": 1e-3, "eigenstates": 1e-4},
        maxiter=80,
    )
    t0 = time.time()
    try:
        e = float(atoms.get_potential_energy())
        return {"ok": True, "e_total_eV": e, "n_atoms": len(atoms),
                "wall_sec": time.time() - t0,
                "e_per_atom_eV": e / len(atoms)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "n_atoms": len(atoms), "wall_sec": time.time() - t0}


def main():
    from ase.data import atomic_numbers as Z_OF

    with open(CANDIDATES_PATH, "rb") as f:
        candidates = pickle.load(f)
    with open(DFT_RESULTS) as f:
        dft = json.load(f)["results"]

    # group top-10 by host: pick first occurrence per host
    seen_hosts = {}
    for r in dft:
        if r["host"] not in seen_hosts:
            seen_hosts[r["host"]] = r  # keep one representative

    print(f"Unique hosts to compute pristine for: {list(seen_hosts.keys())}")
    print(f"  ({len(seen_hosts)} pristine calcs)\n")

    results = []
    t_start = time.time()
    for host, r in seen_hosts.items():
        c_idx = r["candidate_idx"]
        s = candidates[c_idx]
        dopant_Z = Z_OF[r["dopant"]]
        n_full = len(s["numbers"])
        Z, pos, cell = strip_dopant(s, dopant_Z)
        n_pristine = len(Z)
        atoms = to_atoms(Z, pos, cell)

        print(f"\nhost={host}  doped→pristine: {n_full}→{n_pristine} atoms",
              flush=True)
        print(f"  candidate={c_idx} (used to derive supercell)")
        print(f"  starting DFT...", flush=True)

        ret = run_dft(atoms, LOG_DIR / f"pristine_{host}.log")
        print(f"  -> {ret}", flush=True)

        results.append({
            "host": host,
            "from_candidate_idx": c_idx,
            "from_dopant": r["dopant"],
            "n_atoms_doped": n_full,
            "n_atoms_pristine": n_pristine,
            **ret,
        })
        # incremental save
        out = {
            "config": {"pw_cutoff_eV": PW_CUTOFF, "kpts": list(KPTS),
                       "xc": XC},
            "results": results,
            "wall_min": (time.time() - t_start) / 60,
        }
        with open(RESULTS / "dft_pristine.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved partial to {RESULTS / 'dft_pristine.json'}",
              flush=True)

    print(f"\n=== Done. {(time.time()-t_start)/60:.1f} min ===")
    print(f"  successful: {sum(1 for r in results if r.get('ok'))}/{len(results)}")


if __name__ == "__main__":
    main()
