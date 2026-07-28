"""Collect the post-hoc SchNet sum-versus-mean host-CV sensitivity."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prm_collect_results import (  # noqa: E402
    file_sha256,
    load_prediction_array,
    paired_sample_comparison,
    regression_metrics,
    write_csv,
)
from src.prm_metrics import finite_spearman  # noqa: E402
from src.prm_provenance import (  # noqa: E402
    ExpectedConfig,
    archive_training_artifacts,
    load_expected_configs,
    load_protocol_targets,
    require_clean_git_snapshot,
    validate_manifest_config,
    validate_protocol_targets,
    validate_training_completion,
)


def git_snapshot() -> dict[str, Any]:
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


def fixed_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must remain equal across the readout intervention."""
    controlled = deepcopy(dict(config))
    controlled.pop("output_dir", None)
    model_kwargs = controlled.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise ValueError("SchNet configuration lacks model_kwargs")
    model_kwargs.pop("readout", None)
    return controlled


def validate_config_pair(
    add_config: Mapping[str, Any], mean_config: Mapping[str, Any],
) -> None:
    if add_config.get("model_kwargs", {}).get("readout") != "add":
        raise ValueError("parent SchNet configuration is not additive")
    if mean_config.get("model_kwargs", {}).get("readout") != "mean":
        raise ValueError("sensitivity SchNet configuration is not mean pooled")
    if fixed_config(add_config) != fixed_config(mean_config):
        raise ValueError("SchNet readout sensitivity changes uncontrolled fields")


def read_samples(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples = {}
    for row in rows:
        index = int(row["sample_index"])
        samples[index] = {
            "host": str(row["host"]),
            "natoms": int(row["natoms"]),
            "target_eV": float(row["target_eV"]),
        }
    return samples


def run_key(config: Mapping[str, Any]) -> tuple[str, int]:
    return Path(str(config["split_path"])).stem, int(config["seed"])


def load_run(
    result_root: Path,
    expected: ExpectedConfig,
    protocol_dir: Path,
    expected_data_sha256: str,
    expected_readout: str,
) -> dict[str, Any]:
    manifest_path = result_root / expected.config["output_dir"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "prm_run_manifest_v1":
        raise ValueError(f"unsupported training manifest: {manifest_path}")
    if manifest.get("git", {}).get("dirty"):
        raise ValueError(f"dirty training run is inadmissible: {manifest_path}")
    validate_manifest_config(manifest, {expected.config["output_dir"]: expected}, manifest_path)
    validate_training_completion(manifest, manifest_path)
    split_id, seed = run_key(expected.config)
    if manifest["split"]["split_id"] != split_id or int(manifest["seed"]) != seed:
        raise ValueError(f"run identity mismatch: {manifest_path}")
    if manifest["data"]["data_sha256"] != expected_data_sha256:
        raise ValueError(f"dataset mismatch: {manifest_path}")
    split_path = protocol_dir / "splits" / f"{split_id}.json"
    if manifest["split"]["sha256"] != file_sha256(split_path):
        raise ValueError(f"split mismatch: {manifest_path}")
    if manifest["config"]["model_kwargs"].get("readout") != expected_readout:
        raise ValueError(f"readout mismatch: {manifest_path}")
    prediction_path = manifest_path.parent / "test_predictions.npz"
    indices, targets, predictions = load_prediction_array(
        prediction_path, "schnet", expected_split_id=split_id
    )
    return {
        "model": f"schnet_{expected_readout}",
        "split_id": split_id,
        "seed": seed,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256(prediction_path),
        "git_commit": manifest["git"]["commit"],
        "config_sha256": manifest["config_sha256"],
        "indices": indices,
        "targets": targets,
        "predictions": predictions,
        "validation_mae_eV": float(manifest["metrics"]["validation"]["mae"]),
        "test_mae_eV": float(manifest["metrics"]["test"]["mae"]),
    }


def seed_average(rows: Sequence[Mapping[str, Any]], split_id: str) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    members = sorted(
        (row for row in rows if row["split_id"] == split_id),
        key=lambda row: int(row["seed"]),
    )
    if len(members) != 3:
        raise ValueError(f"expected three seeds for {split_id}, found {len(members)}")
    reference_indices = np.asarray(members[0]["indices"])
    reference_targets = np.asarray(members[0]["targets"])
    predictions = []
    for member in members:
        if (
            not np.array_equal(member["indices"], reference_indices)
            or not np.allclose(
                member["targets"], reference_targets, rtol=0.0, atol=1e-10
            )
        ):
            raise ValueError(f"seed predictions do not align for {split_id}")
        predictions.append(np.asarray(member["predictions"], dtype=float))
    return reference_indices, reference_targets, np.mean(predictions, axis=0)


def paired_summary(
    targets: np.ndarray,
    add_predictions: np.ndarray,
    mean_predictions: np.ndarray,
    hosts: Sequence[str],
    natoms: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    add_metrics = regression_metrics(targets, add_predictions, hosts=hosts)
    mean_metrics = regression_metrics(targets, mean_predictions, hosts=hosts)
    paired = paired_sample_comparison(
        targets,
        add_predictions,
        mean_predictions,
        seed=20260729,
        draws=20_000,
        groups=hosts,
    )
    paired["mae_difference_mean_minus_add_eV"] = paired.pop(
        "mae_difference_comparator_minus_dart_eV"
    )
    labels = np.asarray(hosts)
    atom_counts = np.asarray(natoms, dtype=float)
    host_rows = []
    for host in sorted(set(hosts)):
        mask = labels == host
        host_rows.append(
            {
                "host": host,
                "n": int(mask.sum()),
                "median_natoms": float(np.median(atom_counts[mask])),
                "add_mae_eV": float(np.mean(np.abs(add_predictions[mask] - targets[mask]))),
                "mean_mae_eV": float(np.mean(np.abs(mean_predictions[mask] - targets[mask]))),
            }
        )
    for row in host_rows:
        row["mean_minus_add_mae_eV"] = row["mean_mae_eV"] - row["add_mae_eV"]
    return (
        {
            "add": add_metrics,
            "mean": mean_metrics,
            "paired_host_cluster_bootstrap": paired,
            "sample_natoms_vs_absolute_error_spearman": {
                "add": finite_spearman(
                    atom_counts,
                    np.abs(add_predictions - targets),
                    context="add-readout size/error association",
                ),
                "mean": finite_spearman(
                    atom_counts,
                    np.abs(mean_predictions - targets),
                    context="mean-readout size/error association",
                ),
            },
            "host_median_natoms_vs_mae_spearman": {
                "add": finite_spearman(
                    [row["median_natoms"] for row in host_rows],
                    [row["add_mae_eV"] for row in host_rows],
                    context="add-readout host size/error association",
                ),
                "mean": finite_spearman(
                    [row["median_natoms"] for row in host_rows],
                    [row["mean_mae_eV"] for row in host_rows],
                    context="mean-readout host size/error association",
                ),
            },
        },
        host_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=ROOT / "artifacts/prm_protocol_v2",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "configs/prm/sensitivity/schnet_mean_host",
    )
    parser.add_argument(
        "--parent-config-dir",
        type=Path,
        default=ROOT / "configs/prm/generated/schnet",
    )
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        default=ROOT / "artifacts/prm_results/comparison/manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts/prm_results/sensitivity/schnet_readout",
    )
    args = parser.parse_args()

    collector_git = git_snapshot()
    require_clean_git_snapshot(collector_git, context="readout sensitivity collector")
    result_root = args.result_root.resolve()
    protocol_dir = args.protocol_dir.resolve()
    protocol = json.loads((protocol_dir / "manifest.json").read_text())
    protocol_targets = load_protocol_targets(protocol_dir / "samples.csv")
    samples = read_samples(protocol_dir / "samples.csv")
    config_manifest_path = args.config_dir / "manifest.json"
    config_manifest = json.loads(config_manifest_path.read_text())
    if (
        config_manifest.get("schema_version")
        != "prm_schnet_readout_sensitivity_configs_v1"
        or config_manifest.get("n_configs") != 15
    ):
        raise ValueError("readout sensitivity config manifest is invalid")

    mean_expected = load_expected_configs(sorted(args.config_dir.glob("*.yaml")))
    add_expected = load_expected_configs(
        sorted(args.parent_config_dir.glob("host_cv5_f*_seed*.yaml"))
    )
    if len(mean_expected) != 15 or len(add_expected) != 15:
        raise ValueError("readout sensitivity requires exactly 15 configs per arm")
    mean_by_key = {run_key(item.config): item for item in mean_expected.values()}
    add_by_key = {run_key(item.config): item for item in add_expected.values()}
    if set(mean_by_key) != set(add_by_key):
        raise ValueError("add and mean readout configurations do not align")

    add_rows = []
    mean_rows = []
    for key in sorted(add_by_key):
        add_item = add_by_key[key]
        mean_item = mean_by_key[key]
        validate_config_pair(add_item.config, mean_item.config)
        add_rows.append(
            load_run(
                result_root, add_item, protocol_dir, protocol["data_sha256"], "add"
            )
        )
        mean_rows.append(
            load_run(
                result_root, mean_item, protocol_dir, protocol["data_sha256"], "mean"
            )
        )

    add_commits = sorted({str(row["git_commit"]) for row in add_rows})
    mean_commits = sorted({str(row["git_commit"]) for row in mean_rows})
    if len(add_commits) != 1 or len(mean_commits) != 1:
        raise ValueError("each readout arm must share one clean training commit")

    all_indices = []
    all_targets = []
    all_add = []
    all_mean = []
    fold_rows = []
    for fold in range(5):
        split_id = f"host_cv5_f{fold}"
        add_indices, add_targets, add_predictions = seed_average(add_rows, split_id)
        mean_indices, mean_targets, mean_predictions = seed_average(mean_rows, split_id)
        if (
            not np.array_equal(add_indices, mean_indices)
            or not np.allclose(add_targets, mean_targets, rtol=0.0, atol=1e-10)
        ):
            raise ValueError(f"readout arms do not align for {split_id}")
        canonical_targets = validate_protocol_targets(
            add_indices,
            add_targets,
            protocol_targets,
            context=f"{split_id} readout sensitivity",
        )
        add_mae = float(np.mean(np.abs(add_predictions - canonical_targets)))
        mean_mae = float(np.mean(np.abs(mean_predictions - canonical_targets)))
        fold_rows.append(
            {
                "split_id": split_id,
                "n": len(add_indices),
                "add_mae_eV": add_mae,
                "mean_mae_eV": mean_mae,
                "mean_minus_add_mae_eV": mean_mae - add_mae,
                "mean_better": mean_mae < add_mae,
            }
        )
        all_indices.append(add_indices)
        all_targets.append(canonical_targets)
        all_add.append(add_predictions)
        all_mean.append(mean_predictions)

    indices = np.concatenate(all_indices)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("host-CV test folds are not disjoint")
    order = np.argsort(indices)
    indices = indices[order]
    targets = np.concatenate(all_targets)[order]
    add_predictions = np.concatenate(all_add)[order]
    mean_predictions = np.concatenate(all_mean)[order]
    if set(indices.tolist()) != set(protocol_targets):
        raise ValueError("host-CV predictions do not cover the canonical dataset")
    hosts = [samples[int(index)]["host"] for index in indices]
    natoms = [samples[int(index)]["natoms"] for index in indices]
    summary, host_rows = paired_summary(
        targets, add_predictions, mean_predictions, hosts, natoms
    )
    summary["fold_directional_consistency"] = {
        "mean_better_folds": sum(bool(row["mean_better"]) for row in fold_rows),
        "n_folds": len(fold_rows),
    }
    summary["question"] = config_manifest["question"]
    summary["analysis_role"] = config_manifest["analysis_role"]
    summary["intervention"] = config_manifest["intervention"]
    summary["fixed_conditions"] = config_manifest["fixed_conditions"]
    summary["comparability"] = (
        "Direct post-hoc host-CV sensitivity: only graph readout and controlled "
        "output location differ; it does not replace the prespecified comparator."
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [
        {
            key: row[key]
            for key in (
                "model", "split_id", "seed", "validation_mae_eV", "test_mae_eV",
                "git_commit", "config_sha256", "manifest_sha256", "prediction_sha256",
            )
        }
        for row in add_rows + mean_rows
    ]
    write_csv(out_dir / "run_metrics.csv", run_rows)
    write_csv(out_dir / "fold_metrics.csv", fold_rows)
    write_csv(out_dir / "host_metrics.csv", host_rows)
    np.savez_compressed(
        out_dir / "predictions.npz",
        schema_version=np.asarray("prm_schnet_readout_sensitivity_predictions_v1"),
        indices=indices,
        targets=targets,
        add_predictions=add_predictions,
        mean_predictions=mean_predictions,
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    archived_runs = archive_training_artifacts(
        [Path(row["manifest_path"]) for row in mean_rows],
        out_dir / "runs",
        repository_root=ROOT,
        strip_output_prefix="sensitivity/schnet_readout/mean",
    )
    output_files = {
        "summary": "summary.json",
        "run_metrics": "run_metrics.csv",
        "fold_metrics": "fold_metrics.csv",
        "host_metrics": "host_metrics.csv",
        "predictions": "predictions.npz",
    }
    manifest = {
        "schema_version": "prm_schnet_readout_sensitivity_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector_git": collector_git,
        "training_commits": {"add": add_commits[0], "mean": mean_commits[0]},
        "data_sha256": protocol["data_sha256"],
        "config_manifest": {
            "path": str(config_manifest_path.resolve()),
            "sha256": file_sha256(config_manifest_path),
        },
        "parent_comparison_manifest": {
            "path": str(args.comparison_manifest.resolve()),
            "sha256": file_sha256(args.comparison_manifest),
        },
        "n_runs": {"add_reference": len(add_rows), "mean_sensitivity": len(mean_rows)},
        "n_archived_runs": len(archived_runs),
        "archived_runs": archived_runs,
        "archive_policy": {
            "scope": "15 mean-readout sensitivity runs",
            "included": [
                "run_manifest.json", "metrics.json", "split_indices.npz",
                "val_predictions.npz", "test_predictions.npz",
            ],
            "checkpoint": "SHA-256 recorded; binary retained outside Git",
        },
        "output_sha256": {
            key: file_sha256(out_dir / relative)
            for key, relative in output_files.items()
        },
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    (out_dir / "manifest.json").write_text(encoded + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
