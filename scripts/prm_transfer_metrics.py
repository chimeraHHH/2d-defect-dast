"""Assemble repaired-transfer comparison metrics into one JSON.

Computes per-regime, per-fold test MAEs for three aligned models:
- DART (repaired G2 predictions; seed-averaged inside each fold),
- SchNet (archived campaign predictions; unaffected by the DART graph
  defects per the committed pipeline-independence audit),
- the validation-selected descriptor model (archived per-regime summary).

No training or inference is performed; every value is recomputed from
hash-verifiable prediction archives or read from the audited descriptor
summary.  The output feeds the transfer figure and benchmark table.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

REGIMES = {
    "id": ("id_cv5_f{fold}", (242,), "random cross-validation"),
    "pair": ("pair_cv5_f{fold}", (242,), "pair held out"),
    "chemistry_block": ("chemistry_block_g6x3d", (242,), "chemistry block"),
    "dopant": ("dopant_cv5_f{fold}", (242, 243, 244), "impurity held out"),
    "host": ("host_cv5_f{fold}", (242, 243, 244), "host held out"),
}
SCHNET_SEEDS = (342, 343, 344)


def fold_mae(run_dir: Path, seeds: tuple[int, ...]) -> float:
    per_seed = []
    reference = None
    for seed in seeds:
        with np.load(run_dir / f"seed{seed}" / "test_predictions.npz",
                     allow_pickle=False) as archive:
            indices = np.asarray(archive["indices"])
            order = np.argsort(indices)
            preds = np.asarray(archive["preds"], dtype=float)[order]
            targets = np.asarray(archive["targets"], dtype=float)[order]
        if reference is None:
            reference = (indices[order], targets)
        else:
            if not np.array_equal(reference[0], indices[order]):
                raise ValueError(f"seed index mismatch in {run_dir}")
        per_seed.append(preds)
    mean_preds = np.mean(np.column_stack(per_seed), axis=1)
    return float(np.mean(np.abs(mean_preds - reference[1])))


def model_metrics(root: Path, seeds_map, seed_override=None) -> dict:
    out = {}
    for regime, (pattern, dart_seeds, label) in REGIMES.items():
        seeds = seed_override if seed_override else dart_seeds
        if "{fold}" in pattern:
            folds = [
                fold_mae(root / pattern.format(fold=fold), seeds)
                for fold in range(5)
            ]
        else:
            folds = [fold_mae(root / pattern, seeds)]
        out[regime] = {
            "label": label,
            "fold_maes": [round(v, 4) for v in folds],
            "pooled_mean_of_folds": round(float(np.mean(folds)), 4),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dart-root", required=True,
                        help="repaired G2 run root (server)")
    parser.add_argument("--schnet-root", default=str(
        ROOT / "artifacts/prm_results/comparison/runs/baselines/schnet"
    ))
    parser.add_argument("--descriptor-summary", default=str(
        ROOT / "artifacts/prm_results/descriptors/selected_summary.csv"
    ))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    dart = model_metrics(Path(args.dart_root), REGIMES)
    schnet = model_metrics(Path(args.schnet_root), REGIMES,
                           seed_override=SCHNET_SEEDS)
    descriptor = {}
    with open(args.descriptor_summary, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "descriptor:selected":
                continue
            regime = {
                "id_cv": "id", "pair_cv": "pair", "host_cv": "host",
                "dopant_cv": "dopant", "chemistry_block": "chemistry_block",
                "id": "id", "pair": "pair", "host": "host", "dopant": "dopant",
            }.get(row["regime"], row["regime"])
            descriptor[regime] = {
                "test_mae_mean": round(float(row["test_mae_mean"]), 4),
                "n_folds": int(row["n_folds"]),
            }

    payload = {
        "schema_version": "prm_repaired_transfer_metrics_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "dart": "repaired G2 predictions, seed-averaged per fold",
            "schnet": (
                "archived campaign predictions; independence from the DART "
                "graph defects established by the committed audit"
            ),
            "descriptor": "audited selected-descriptor summary (legacy archive)",
        },
        "dart": dart,
        "schnet": schnet,
        "descriptor": descriptor,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
