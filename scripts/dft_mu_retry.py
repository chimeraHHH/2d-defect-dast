"""C18d-retry: Retry the 3 atoms that failed in dft_mu_atoms.py.

Ir, La, Pb did not converge under integer occupations. Retry with
Fermi-Dirac smearing (width=0.1) which lifts the degeneracy at the
Fermi level for heavy d/f-electron atoms.

Merges the new energies into results/dft_mu_atoms.json.
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

RESULTS = ROOT / "results"
LOG_DIR = ROOT / "logs" / "dft_mu"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RETRY = ["Ir", "La", "Pb"]


def run_atom_smear(symbol, box=12.0, width=0.1):
    from gpaw import GPAW, PW
    a = Atoms(symbol, positions=[(box / 2, box / 2, box / 2)],
              cell=[box, box, box], pbc=False)
    a.calc = GPAW(
        mode=PW(300),
        kpts=(1, 1, 1),
        xc="PBE",
        txt=str(LOG_DIR / f"atom_{symbol}_smear.log"),
        symmetry={"point_group": False},
        convergence={"energy": 1e-3, "density": 1e-3, "eigenstates": 1e-4},
        spinpol=True,
        occupations={"name": "fermi-dirac", "width": width},
        maxiter=300,
    )
    t0 = time.time()
    try:
        e = float(a.get_potential_energy())
        return {"ok": True, "element": symbol, "e_per_atom_eV": e,
                "smear": True, "wall_sec": time.time() - t0}
    except Exception as exc:
        return {"ok": False, "element": symbol, "smear": True,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_sec": time.time() - t0}


def main():
    src = json.loads((RESULTS / "dft_mu_atoms.json").read_text())
    by_el = {r["element"]: r for r in src["results"]}

    t_start = time.time()
    for el in RETRY:
        print(f"\nretrying {el} with Fermi smearing...", flush=True)
        r = run_atom_smear(el)
        print(f"  -> {r}", flush=True)
        by_el[el] = r
        out = {"results": list(by_el.values()),
               "wall_min": (time.time() - t_start) / 60}
        (RESULTS / "dft_mu_atoms.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== Done. {(time.time()-t_start)/60:.1f} min ===")
    print(f"  successful: {sum(1 for r in by_el.values() if r.get('ok'))}/{len(by_el)}")


if __name__ == "__main__":
    main()
