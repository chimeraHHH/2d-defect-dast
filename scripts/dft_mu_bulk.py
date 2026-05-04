"""C18e: Bulk-elemental GPAW PBE energies for the 8 dopant elements.

Replaces the coarse isolated-atom mu values with the standard bulk
elemental reference used by most defect-formation databases (incl.
IMP2D). For each dopant element, build the standard bulk crystal via
ase.build.bulk() and compute single-point GPAW PW PBE.

Outputs:
- results/dft_mu_bulk.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ase.build import bulk  # noqa: E402

DFT_VAL = ROOT / "results" / "dft_validation.json"
RESULTS = ROOT / "results"
LOG_DIR = ROOT / "logs" / "dft_mu_bulk"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Standard bulk crystal structures (matching ASE bulk() defaults
# but explicit to be defensible).
# (crystal type, lattice constant in A from experiment)
BULK_SPEC = {
    "Pb": ("fcc", 4.95),
    "Bi": ("rhombohedral", 4.75),  # ASE uses rhombohedral for Bi by default
    "In": ("tetragonal", 3.25),
    "Ir": ("fcc", 3.84),
    "La": ("hcp", 3.77),
    "Rh": ("fcc", 3.80),
    "Ru": ("hcp", 2.71),
    "S": ("orthorhombic", 10.46),  # alpha-S (S8 ring); too complex, skip and use atom
}
# For S, the S8 ring is hard with default bulk(); fall back to isolated atom + warning.


def make_bulk(element):
    spec = BULK_SPEC.get(element)
    if spec is None:
        return None
    crys, a = spec
    if crys == "fcc":
        return bulk(element, "fcc", a=a, cubic=True)
    if crys == "hcp":
        return bulk(element, "hcp", a=a)
    if crys == "rhombohedral":
        return bulk(element, "rhombohedral", a=a, alpha=57.35)
    if crys == "tetragonal":
        return bulk(element, "tetragonal", a=a, c=4.95)
    if crys == "orthorhombic":
        return None  # complex; skip
    return None


def run_bulk(element, atoms):
    from gpaw import GPAW, PW
    n = len(atoms)
    atoms.calc = GPAW(
        mode=PW(300),
        kpts=(8, 8, 8),  # dense for bulk
        xc="PBE",
        txt=str(LOG_DIR / f"bulk_{element}.log"),
        symmetry={"point_group": False},
        convergence={"energy": 1e-3, "density": 1e-3, "eigenstates": 1e-4},
        occupations={"name": "fermi-dirac", "width": 0.05},
        maxiter=200,
    )
    t0 = time.time()
    try:
        e = float(atoms.get_potential_energy())
        return {"ok": True, "element": element, "n_atoms": n,
                "e_total_eV": e, "e_per_atom_eV": e / n,
                "wall_sec": time.time() - t0}
    except Exception as exc:
        return {"ok": False, "element": element, "n_atoms": n,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_sec": time.time() - t0}


def main():
    with open(DFT_VAL) as f:
        dv = json.load(f)["results"]
    elements = sorted({r["dopant"] for r in dv})
    print(f"Elements: {elements}\n")

    results = []
    t_start = time.time()
    for el in elements:
        atoms = make_bulk(el)
        if atoms is None:
            print(f"\n  {el}: SKIP (no bulk spec available)")
            results.append({"ok": False, "element": el,
                            "skipped": True,
                            "note": "no bulk spec available; will fall back to atomic mu"})
            continue
        print(f"\n  bulk {el} ({len(atoms)} atoms)...", flush=True)
        r = run_bulk(el, atoms)
        print(f"    -> {r}", flush=True)
        results.append(r)
        out = {"results": results, "wall_min": (time.time() - t_start) / 60,
               "bulk_spec": BULK_SPEC}
        (RESULTS / "dft_mu_bulk.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== Done. {(time.time()-t_start)/60:.1f} min ===")
    print(f"  successful: {sum(1 for r in results if r.get('ok'))}/{len(results)}")


if __name__ == "__main__":
    main()
