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

from scripts.dft_mu_atoms import run_atom  # noqa: E402

RESULTS = ROOT / "results"
RETRY = ["Ir", "La", "Pb"]


def main():
    src = json.loads((RESULTS / "dft_mu_atoms.json").read_text())
    by_el = {r["element"]: r for r in src["results"]}

    t_start = time.time()
    for el in RETRY:
        print(f"\nretrying {el} with Fermi smearing...", flush=True)
        r = run_atom(el, smear=True)
        print(f"  -> {r}", flush=True)
        by_el[el] = r

    out = {"results": list(by_el.values()),
           "wall_min": (time.time() - t_start) / 60}
    (RESULTS / "dft_mu_atoms.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== Done. {(time.time()-t_start)/60:.1f} min ===")
    print(f"  successful: {sum(1 for r in by_el.values() if r.get('ok'))}/{len(by_el)}")


if __name__ == "__main__":
    main()
