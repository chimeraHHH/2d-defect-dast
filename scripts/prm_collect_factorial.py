"""Collect the paired 2^3 DART factorial without test-set model selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from src.prm_provenance import (
    ExpectedConfig,
    archive_training_artifacts,
    load_expected_configs,
    validate_dart_assets,
    validate_manifest_config,
    validate_training_completion,
)


ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = (
    ("G", "use_gated_pooling"),
    ("E", "use_env_enrichment"),
    ("P", "use_prenorm_local"),
)
TERMS = (
    ("G", (0,)), ("E", (1,)), ("P", (2,)),
    ("G:E", (0, 1)), ("G:P", (0, 2)), ("E:P", (1, 2)),
    ("G:E:P", (0, 1, 2)),
)
METRICS = ("mae", "rmse", "bias", "spearman", "r2")


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
    return {"commit": commit or None, "dirty": bool(status), "status_porcelain": status.splitlines()}


def variant_from_config(config: Mapping[str, Any]) -> Tuple[str, Tuple[int, int, int]]:
    kwargs = config["model_kwargs"]
    bits = tuple(int(bool(kwargs[name])) for _, name in COMPONENTS)
    return "g" + "".join(str(bit) for bit in bits), bits


def contract_hash(config: Mapping[str, Any]) -> str:
    contract = deepcopy(dict(config))
    for key in ("output_dir", "split_path", "seed"):
        contract.pop(key, None)
    kwargs = contract.get("model_kwargs", {})
    for _, name in COMPONENTS:
        kwargs.pop(name, None)
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def bootstrap_mean_ci(
    values: Sequence[float], samples: int = 50_000, seed: int = 20260719,
) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return {
            "mean": float(array.mean()), "std": 0.0,
            "ci_low": float("nan"), "ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def factorial_contrast(
    repeat_rows: Sequence[Mapping[str, Any]], field: str, term: Sequence[int],
) -> float:
    positive = []
    negative = []
    for row in repeat_rows:
        sign = int(np.prod([1 if row["bits"][index] else -1 for index in term]))
        (positive if sign > 0 else negative).append(float(row[field]))
    if len(positive) != 4 or len(negative) != 4:
        raise ValueError(f"factorial term {term} is not balanced")
    return float(np.mean(positive) - np.mean(negative))


def summarize_factorial(
    rows: Sequence[Mapping[str, Any]], bootstrap_samples: int = 50_000,
) -> Dict[str, Any]:
    variants = sorted({str(row["variant"]) for row in rows})
    repeats = sorted({int(row["repeat"]) for row in rows})
    if variants != [f"g{mask:03b}" for mask in range(8)]:
        raise ValueError(f"expected all eight variants, found {variants}")
    if repeats != list(range(42, 47)):
        raise ValueError(f"expected paired repeats 42--46, found {repeats}")
    expected_pairs = {(variant, repeat) for variant in variants for repeat in repeats}
    observed_pairs = {(str(row["variant"]), int(row["repeat"])) for row in rows}
    if observed_pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise ValueError("factorial runs are missing or duplicated")

    variant_summary = []
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        validation = bootstrap_mean_ci(
            [row["validation_mae"] for row in subset], bootstrap_samples,
            seed=20260719 + int(variant[1:], 2),
        )
        test = bootstrap_mean_ci(
            [row["test_mae"] for row in subset], bootstrap_samples,
            seed=20260819 + int(variant[1:], 2),
        )
        variant_summary.append(
            {
                "variant": variant,
                "enabled_components": int(sum(subset[0]["bits"])),
                "validation_mae": validation,
                "test_mae": test,
            }
        )

    # Test metrics are deliberately absent from the ordering key.
    ranked = sorted(
        variant_summary,
        key=lambda item: (
            item["validation_mae"]["mean"],
            item["enabled_components"], item["variant"],
        ),
    )
    selected = ranked[0]
    selected_rows = [row for row in rows if row["variant"] == selected["variant"]]

    effects = []
    for split_name in ("validation", "test"):
        for metric in METRICS:
            field = f"{split_name}_{metric}"
            for term_name, term_indices in TERMS:
                paired = []
                for repeat in repeats:
                    repeat_rows = [row for row in rows if int(row["repeat"]) == repeat]
                    paired.append(factorial_contrast(repeat_rows, field, term_indices))
                summary = bootstrap_mean_ci(
                    paired, bootstrap_samples,
                    seed=20261000 + 100 * len(term_indices) + sum(term_indices)
                    + (0 if split_name == "validation" else 10),
                )
                effects.append(
                    {
                        "split": split_name,
                        "metric": metric,
                        "term": term_name,
                        "definition": "mean(metric | contrast=+1) - mean(metric | contrast=-1)",
                        "repeat_effects": paired,
                        **summary,
                        "n_paired_repeats": len(paired),
                    }
                )

    locked_test = {
        metric: bootstrap_mean_ci(
            [row[f"test_{metric}"] for row in selected_rows],
            bootstrap_samples,
            seed=20262000 + index,
        )
        for index, metric in enumerate(METRICS)
    }
    return {
        "selection": {
            "rule": "minimum mean validation MAE; exact ties use fewer enabled components then variant code",
            "selection_data": "validation only",
            "selected_variant": selected["variant"],
            "validation_mae": selected["validation_mae"],
            "locked_test": locked_test,
        },
        "variant_summary": variant_summary,
        "effects": effects,
        "repeats": repeats,
    }


def load_runs(
    result_root: Path, expected_data_sha256: str,
    expected_split_hashes: Mapping[str, str],
    expected_configs: Mapping[str, ExpectedConfig],
    allow_dirty: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    manifests = sorted((result_root / "factorial").glob("g*/split*_seed*/run_manifest.json"))
    rows: List[Dict[str, Any]] = []
    sources = []
    contracts = set()
    commits = set()
    for path in manifests:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        if payload.get("schema_version") != "prm_run_manifest_v1":
            raise ValueError(f"unsupported manifest schema: {path}")
        if payload["data"]["data_sha256"] != expected_data_sha256:
            raise ValueError(f"dataset hash mismatch: {path}")
        if payload.get("git", {}).get("dirty") and not allow_dirty:
            raise ValueError(f"dirty training run is not admissible: {path}")
        config = payload["config"]
        expected_config = validate_manifest_config(payload, expected_configs, path)
        validate_training_completion(payload, path)
        validate_dart_assets(payload, path)
        variant, bits = variant_from_config(config)
        if path.parent.parent.name != variant:
            raise ValueError(f"variant path/config mismatch: {path}")
        split_id = str(payload["split"]["split_id"])
        if not split_id.startswith("id_repeat_s"):
            raise ValueError(f"unexpected factorial split: {split_id}")
        repeat = int(split_id.rsplit("s", 1)[1])
        if payload["split"].get("sha256") != expected_split_hashes.get(split_id):
            raise ValueError(f"split hash mismatch: {path}")
        contracts.add(contract_hash(config))
        commits.add(payload["git"]["commit"])
        row: Dict[str, Any] = {
            "variant": variant, "bits": bits, "repeat": repeat,
            "seed": int(payload["seed"]), "n_params": int(payload["metrics"]["n_params"]),
            "manifest_path": str(path), "git_commit": payload["git"]["commit"],
        }
        for split_name in ("validation", "test"):
            for metric in METRICS:
                row[f"{split_name}_{metric}"] = float(payload["metrics"][split_name][metric])
        rows.append(row)
        sources.append(
            {
                "path": str(path), "sha256": file_sha256(path),
                "git": payload["git"], "config_sha256": payload["config_sha256"],
                "expected_config": str(expected_config.path),
            }
        )
    if len(contracts) > 1:
        raise ValueError("factorial training settings differ beyond component flags, split and seed")
    if len(commits) > 1:
        raise ValueError("factorial runs were produced by more than one code commit")
    return rows, sources


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-manifest", type=Path,
        default=ROOT / "artifacts/prm_protocol_v2/manifest.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts/prm_results/factorial",
    )
    parser.add_argument(
        "--config-dir", type=Path,
        default=ROOT / "configs/prm/generated/factorial",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol_manifest.read_text())
    protocol_dir = args.protocol_manifest.resolve().parent
    expected_split_hashes = {
        f"id_repeat_s{repeat}": file_sha256(
            protocol_dir / "splits" / f"id_repeat_s{repeat}.json"
        )
        for repeat in range(42, 47)
    }
    expected_configs = load_expected_configs(
        sorted(args.config_dir.resolve().glob("*.yaml"))
    )
    rows, sources = load_runs(
        args.result_root.resolve(), protocol["data_sha256"],
        expected_split_hashes, expected_configs, args.allow_dirty,
    )
    summary = summarize_factorial(rows, args.bootstrap_samples)
    collector_git = git_snapshot()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    archived_runs = archive_training_artifacts(
        [Path(row["manifest_path"]) for row in rows],
        out_dir / "runs",
        repository_root=ROOT,
        strip_output_prefix="factorial",
    )

    run_fields = [
        "variant", "repeat", "seed", "n_params", "git_commit",
        *[f"{split}_{metric}" for split in ("validation", "test") for metric in METRICS],
        "manifest_path",
    ]
    write_csv(out_dir / "runs.csv", rows, run_fields)
    variant_rows = []
    for item in summary["variant_summary"]:
        variant_rows.append(
            {
                "variant": item["variant"],
                "enabled_components": item["enabled_components"],
                **{f"validation_mae_{key}": value for key, value in item["validation_mae"].items()},
                **{f"test_mae_{key}": value for key, value in item["test_mae"].items()},
            }
        )
    write_csv(
        out_dir / "variant_summary.csv", variant_rows,
        list(variant_rows[0]) if variant_rows else [],
    )
    effect_rows = [
        {**item, "repeat_effects": json.dumps(item["repeat_effects"])}
        for item in summary["effects"]
    ]
    write_csv(
        out_dir / "effects.csv", effect_rows,
        list(effect_rows[0]) if effect_rows else [],
    )
    bundle = {
        "schema_version": "prm_factorial_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector_git": collector_git,
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "data_sha256": protocol["data_sha256"],
        "bootstrap": {
            "method": "paired nonparametric bootstrap over the five repeat-level contrasts",
            "samples": args.bootstrap_samples,
            "confidence": 0.95,
        },
        "n_runs": len(rows),
        "sources": sources,
        "archive_policy": {
            "included": [
                "run_manifest.json", "metrics.json", "split_indices.npz",
                "val_predictions.npz", "test_predictions.npz",
            ],
            "checkpoint": "SHA-256 recorded; binary retained outside Git",
        },
        "n_archived_runs": len(archived_runs),
        "archived_runs": archived_runs,
        **summary,
    }
    bundle_path = out_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    selection_document = {
        "schema_version": "prm_factorial_selection_v1",
        **summary["selection"],
        "factorial_bundle": bundle_path.name,
        "factorial_bundle_sha256": file_sha256(bundle_path),
        "data_sha256": protocol["data_sha256"],
        "n_runs": len(rows),
        "training_commits": sorted({str(row["git_commit"]) for row in rows}),
        "collector_git": collector_git,
    }
    (out_dir / "selection.json").write_text(
        json.dumps(selection_document, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(selection_document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
