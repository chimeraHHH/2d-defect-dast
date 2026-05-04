"""C18d: Per-atom GPAW PBE energy for the dopant elements.

Computes E_atom for each unique dopant element appearing in the C18
top-10 candidates, used as a reference μ in
  Ef_DFT = E_doped - E_pristine - μ_dopant

Isolated atom in a large box, single-point PBE. This is a coarse
chemical potential (overestimates by 1-3 eV vs bulk elemental phase)
but provides a *consistent* offset across candidates so that ranking
and σ-vs-error correlation are preserved.

Outputs:
- results/dft_mu_atoms.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ase import Atoms  # noqa: E402

DFT_VAL = ROOT / "results" / "dft_validation.json"
RESULTS = ROOT / "results"
LOG_DIR = ROOT / "logs" / "dft_mu"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_atom(symbol, box=12.0, smear=False):
    """Single isolated atom in a `box`-A^3 cell, gamma point, PBE.

    `smear=True` enables Fermi-Dirac smearing — needed for heavy d/f
    atoms (Ir, La, Pb, etc.) where degenerate frontier orbitals
    prevent SCF convergence under integer occupations.
    """
    from gpaw import GPAW, PW
    a = Atoms(symbol, positions=[(box / 2, box / 2, box / 2)],
              cell=[box, box, box], pbc=False)
    kw = dict(
        mode=PW(300),
        kpts=(1, 1, 1),
        xc="PBE",
        txt=str(LOG_DIR / f"atom_{symbol}{'_smear' if smear else ''}.log"),
        symmetry={"point_group": False},
        convergence={"energy": 1e-3, "density": 1e-3, "eigenstates": 1e-4},
        spinpol=True,
        maxiter=300,
    )
    if smear:
        kw["occupations"] = {"name": "fermi-dirac", "width": 0.1}
        # do NOT use hund with smearing — let SCF find occupations freely
    else:
        kw["hund"] = True
    a.calc = GPAW(**kw)
    t0 = time.time()
    try:
        e = float(a.get_potential_energy())
        return {"ok": True, "element": symbol, "e_per_atom_eV": e,
                "smear": smear, "wall_sec": time.time() - t0}
    except Exception as exc:
        return {"ok": False, "element": symbol, "smear": smear,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_sec": time.time() - t0}


def main():
    with open(DFT_VAL) as f:
        dv = json.load(f)["results"]
    elements = sorted({r["dopant"] for r in dv})
    print(f"Elements to compute: {elements}")

    results = []
    t_start = time.time()
    for e in elements:
        print(f"\n  atom {e}...", flush=True)
        r = run_atom(e, smear=False)
        if not r.get("ok"):
            print(f"    integer-occ failed; retrying with Fermi smearing...",
                  flush=True)
            r = run_atom(e, smear=True)
        print(f"    -> {r}", flush=True)
        results.append(r)
        out = {"results": results, "wall_min": (time.time() - t_start) / 60}
        with open(RESULTS / "dft_mu_atoms.json", "w") as f:
            json.dump(out, f, indent=2)

    print(f"\n=== Done. {(time.time()-t_start)/60:.1f} min ===")
    print(f"  successful: {sum(1 for r in results if r.get('ok'))}/{len(results)}")


if __name__ == "__main__":
    main()
