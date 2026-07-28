"""Audit chemical overlap between JARVIS pretraining and canonical IMP2D."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from ase.data import chemical_symbols
from ase.formula import Formula

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prm_provenance import require_clean_git_snapshot


def strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def reduced_formula(formula: str) -> str:
    value = str(formula).strip()
    if not value:
        raise ValueError("chemical formula is empty")
    return str(Formula(value).reduce()[0])


def load_canonical_imp2d_rows(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"host", "dopant", "canonical_retained"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"IMP2D sample table is incomplete: {path}")
    retained = [
        row for row in rows
        if str(row["canonical_retained"]).strip().lower() == "true"
    ]
    if not retained:
        raise ValueError(f"IMP2D sample table has no canonical rows: {path}")
    return retained


def load_pretraining_samples(path: Path) -> Sequence[Mapping[str, Any]]:
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"pretraining dataset is not a nonempty list: {path}")
    return samples


def element_symbols(numbers: Iterable[Any]) -> set[str]:
    symbols = set()
    for raw_number in numbers:
        number = int(raw_number)
        if number <= 0 or number >= len(chemical_symbols):
            raise ValueError(f"invalid atomic number in pretraining data: {number}")
        symbols.add(chemical_symbols[number])
    return symbols


def summarize_overlap(
    imp2d_rows: Sequence[Mapping[str, Any]],
    pretraining_samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    imp2d_hosts = sorted({str(row["host"]) for row in imp2d_rows})
    imp2d_impurities = sorted({str(row["dopant"]) for row in imp2d_rows})
    if not imp2d_hosts or not imp2d_impurities:
        raise ValueError("canonical IMP2D chemistry is empty")

    pretraining_formula_counts: Counter[str] = Counter()
    pretraining_elements: set[str] = set()
    source_names: set[str] = set()
    for sample in pretraining_samples:
        metadata = sample.get("metadata")
        if not isinstance(metadata, Mapping) or not metadata.get("host"):
            raise ValueError("pretraining sample lacks host metadata")
        pretraining_formula_counts[reduced_formula(str(metadata["host"]))] += 1
        pretraining_elements.update(element_symbols(sample.get("numbers", [])))
        if metadata.get("source"):
            source_names.add(str(metadata["source"]))

    imp2d_reduced_formulas = {reduced_formula(host) for host in imp2d_hosts}
    matching_reduced_formulas = {
        formula for formula in imp2d_reduced_formulas
        if pretraining_formula_counts[formula]
    }
    host_details = []
    for host in imp2d_hosts:
        formula = reduced_formula(host)
        count = int(pretraining_formula_counts[formula])
        if count:
            host_details.append(
                {
                    "imp2d_host": host,
                    "reduced_formula": formula,
                    "pretraining_record_count": count,
                }
            )

    present_impurities = sorted(set(imp2d_impurities) & pretraining_elements)
    absent_impurities = sorted(set(imp2d_impurities) - pretraining_elements)
    return {
        "imp2d": {
            "n_canonical_rows": len(imp2d_rows),
            "n_hosts": len(imp2d_hosts),
            "n_impurities": len(imp2d_impurities),
        },
        "pretraining": {
            "n_records": len(pretraining_samples),
            "n_elements": len(pretraining_elements),
            "source_names": sorted(source_names),
        },
        "overlap": {
            "host_reduced_formula": {
                "n_imp2d_hosts_present": len(host_details),
                "fraction_imp2d_hosts_present": len(host_details) / len(imp2d_hosts),
                "n_imp2d_reduced_formulas": len(imp2d_reduced_formulas),
                "n_imp2d_reduced_formulas_present": len(
                    matching_reduced_formulas
                ),
                "n_pretraining_records": sum(
                    pretraining_formula_counts[formula]
                    for formula in matching_reduced_formulas
                ),
                "details": host_details,
            },
            "impurity_element": {
                "n_imp2d_impurities_present": len(present_impurities),
                "fraction_imp2d_impurities_present": (
                    len(present_impurities) / len(imp2d_impurities)
                ),
                "present": present_impurities,
                "absent": absent_impurities,
            },
        },
        "interpretation_boundary": (
            "Reduced-formula and element overlap is not structural identity or "
            "IMP2D-target leakage. Grouped IMP2D tests hold chemistry out from "
            "IMP2D target supervision, but DART may have encountered the same "
            "host formula or impurity element during JARVIS source-task "
            "pretraining."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-csv", type=Path,
        default=ROOT / "artifacts/prm_protocol_v2/samples.csv",
    )
    parser.add_argument("--jarvis-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_path = args.samples_csv.resolve()
    jarvis_path = args.jarvis_data.resolve()
    git = git_snapshot()
    require_clean_git_snapshot(git, context="pretraining-overlap audit")
    imp2d_rows = load_canonical_imp2d_rows(samples_path)
    pretraining_samples = load_pretraining_samples(jarvis_path)
    summary = summarize_overlap(imp2d_rows, pretraining_samples)
    payload = {
        "schema_version": "prm_pretraining_overlap_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "inputs": {
            "imp2d_samples": {
                "file_name": samples_path.name,
                "sha256": file_sha256(samples_path),
                "size_bytes": samples_path.stat().st_size,
            },
            "jarvis_pretraining": {
                "file_name": jarvis_path.name,
                "sha256": file_sha256(jarvis_path),
                "size_bytes": jarvis_path.stat().st_size,
            },
        },
        **summary,
    }
    output_path = args.out.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(strict_json(payload) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
