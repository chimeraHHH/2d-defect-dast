"""Fetch QPOD defect data from the DTU 2DHub web interface.

The QPOD database doesn't offer a direct .db download, so we scrape the
JSON API behind the web table at qpod.fysik.dtu.dk to collect:
  - host material formula
  - defect type (vacancy / antisite)
  - charge state
  - formation energy
  - atomic structure (Atoms object)

We filter for charge-neutral (q=0) entries and those with valid formation
energies, then build crystal graphs and save as jarvis-compatible pkl.

Output: data/processed/qpod_neutral.pkl
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import requests
from ase import Atoms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph


def fetch_qpod_page(offset=0, limit=100):
    """Fetch a page of QPOD entries from the web API."""
    url = "https://qpod.fysik.dtu.dk/row"
    params = {
        "offset": offset,
        "limit": limit,
    }
    headers = {"Accept": "application/json"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  fetch error at offset={offset}: {e}")
        return None


def fetch_qpod_detail(row_id):
    """Fetch detailed structure for a specific QPOD row."""
    url = f"https://qpod.fysik.dtu.dk/row/{row_id}"
    headers = {"Accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  detail error for id={row_id}: {e}")
        return None


def main():
    print("=" * 60)
    print("Fetching QPOD database via web API")
    print("=" * 60)

    # First, try to get the total count
    page = fetch_qpod_page(0, 1)
    if page is None:
        print("Failed to reach QPOD API. Trying alternative approach...")
        return

    # Check what structure the API returns
    print(f"API response keys: {list(page.keys()) if isinstance(page, dict) else type(page)}")
    if isinstance(page, list):
        print(f"First entry keys: {list(page[0].keys()) if page else 'empty'}")
    elif isinstance(page, dict):
        if "rows" in page:
            print(f"Total rows: {page.get('total', '?')}")
            if page["rows"]:
                print(f"First row keys: {list(page['rows'][0].keys())}")
                print(f"First row sample: {json.dumps(page['rows'][0], indent=2, default=str)[:500]}")
        else:
            print(f"Response: {json.dumps(page, indent=2, default=str)[:500]}")

    # Collect all entries
    all_entries = []
    offset = 0
    batch_size = 100
    while True:
        print(f"  fetching offset={offset}...", end=" ", flush=True)
        page = fetch_qpod_page(offset, batch_size)
        if page is None:
            break

        if isinstance(page, dict) and "rows" in page:
            rows = page["rows"]
            total = page.get("total", 0)
        elif isinstance(page, list):
            rows = page
            total = None
        else:
            break

        if not rows:
            break

        all_entries.extend(rows)
        print(f"got {len(rows)}, total collected: {len(all_entries)}")

        if total and len(all_entries) >= total:
            break
        if len(rows) < batch_size:
            break
        offset += batch_size
        time.sleep(0.5)

    print(f"\nTotal entries collected: {len(all_entries)}")

    if not all_entries:
        print("No entries collected. API may have changed format.")
        return

    # Filter for charge-neutral entries with formation energy
    neutral = []
    for entry in all_entries:
        charge = entry.get("charge", entry.get("q", None))
        ef = entry.get("ef", entry.get("formation_energy", None))
        if charge is not None and int(charge) == 0 and ef is not None:
            neutral.append(entry)

    print(f"Charge-neutral entries: {len(neutral)}")

    if not neutral:
        print("No neutral entries found. Checking available keys...")
        if all_entries:
            print(f"Sample entry: {json.dumps(all_entries[0], indent=2, default=str)[:1000]}")
        return

    # Build crystal graphs
    samples = []
    skipped = 0
    for i, entry in enumerate(neutral):
        try:
            # Try to reconstruct ASE Atoms from the entry
            # The format depends on what the API returns
            atoms_data = entry.get("atoms", entry.get("structure", None))
            if atoms_data is None:
                # Try fetching detail
                detail = fetch_qpod_detail(entry.get("id", i))
                if detail:
                    atoms_data = detail.get("atoms", detail.get("structure", None))

            if atoms_data is None:
                skipped += 1
                continue

            # Build ASE Atoms
            if isinstance(atoms_data, dict):
                atoms = Atoms(
                    symbols=atoms_data.get("symbols", atoms_data.get("elements", [])),
                    positions=np.array(atoms_data.get("positions", [])),
                    cell=np.array(atoms_data.get("cell", np.eye(3) * 10)),
                    pbc=True,
                )
            else:
                skipped += 1
                continue

            g = build_graph(atoms, cutoff=5.0)
            n_atoms = len(atoms)

            # For vacancies/antisites, mark the defect site
            defect_mask = np.zeros(n_atoms, dtype=np.int64)
            defect_site = entry.get("defect_site", None)
            if defect_site is not None and 0 <= defect_site < n_atoms:
                defect_mask[defect_site] = 1

            ef = float(entry.get("ef", entry.get("formation_energy", 0)))
            sample = {
                "id": i,
                "unique_id": f"qpod_{entry.get('id', i)}",
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
                "target": ef,
                "metadata": {
                    "host": str(entry.get("host", entry.get("formula", ""))),
                    "dopant": str(entry.get("defect_type", "")),
                    "site": str(entry.get("site", "")),
                    "defecttype": str(entry.get("defect_type", "vacancy")),
                    "natoms": n_atoms,
                    "spacegroup": "",
                    "supercell": "",
                    "source": "QPOD",
                },
            }
            samples.append(sample)
        except Exception as exc:
            skipped += 1
            if i < 5:
                print(f"  error on entry {i}: {exc}")

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(neutral)}, kept {len(samples)}")

    print(f"\nKept {len(samples)}, skipped {skipped}")

    if samples:
        out_path = ROOT / "data/processed/qpod_neutral.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

        ef_vals = [s["target"] for s in samples]
        print(f"Ef range: [{min(ef_vals):.3f}, {max(ef_vals):.3f}] eV")
        print(f"Ef mean: {np.mean(ef_vals):.3f} ± {np.std(ef_vals):.3f} eV")


if __name__ == "__main__":
    main()
