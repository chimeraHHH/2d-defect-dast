"""Out-of-fold materials screening and error analysis for the selected DART."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from ase.data import atomic_numbers

from src.prm_metrics import finite_spearman
from src.prm_provenance import (
    ExpectedConfig,
    load_verified_factorial_selection,
    load_protocol_targets,
    load_expected_configs,
    require_clean_git_snapshot,
    validate_dart_assets,
    validate_manifest_config,
    validate_protocol_targets,
    validate_training_completion,
)


ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = ("use_gated_pooling", "use_env_enrichment", "use_prenorm_local")
ENERGY_TIE_TOLERANCE_EV = 1e-8


def strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prediction_array(
    path: Path, expected_split_id: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        expected_fields = {
            "schema_version", "split_id", "split", "indices", "preds", "targets",
        }
        if set(archive.files) != expected_fields:
            raise ValueError(f"prediction fields are incomplete: {path}")
        if str(archive["schema_version"].item()) != "prm_predictions_v1":
            raise ValueError(f"unsupported prediction schema: {path}")
        if str(archive["split"].item()) != "test":
            raise ValueError(f"prediction partition mismatch: {path}")
        split_id = str(archive["split_id"].item())
        if expected_split_id is not None and split_id != expected_split_id:
            raise ValueError(f"prediction split ID mismatch: {path}")
        raw_indices = np.asarray(archive["indices"])
        predictions = np.asarray(archive["preds"], dtype=float)
        targets = np.asarray(archive["targets"])
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError(f"prediction indices are not integers: {path}")
    if (
        not np.issubdtype(targets.dtype, np.floating)
        or targets.dtype.itemsize < np.dtype(np.float32).itemsize
    ):
        raise ValueError(f"prediction targets have unsupported dtype: {path}")
    indices = raw_indices.astype(np.int64, copy=False)
    if (
        indices.ndim != 1
        or predictions.ndim != 1
        or targets.ndim != 1
        or len(indices) == 0
        or len(indices) != len(predictions)
        or len(indices) != len(targets)
        or len(np.unique(indices)) != len(indices)
        or not np.isfinite(predictions).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError(f"prediction vectors are not finite and uniquely aligned: {path}")
    order = np.argsort(indices)
    return indices[order], predictions[order], targets[order]


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


def read_sample_table(path: Path) -> Dict[int, Dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples = {}
    for row in rows:
        index = int(row["sample_index"])
        samples[index] = {
            **row,
            "sample_index": index,
            "id": int(row["id"]),
            "target_eV": float(row["target_eV"]),
            "natoms": int(row["natoms"]),
        }
    return samples


def dopant_period(symbol: str) -> int:
    z = atomic_numbers.get(symbol, 0)
    for period, upper in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if z <= upper:
            return period
    return 0


def load_oof_predictions(
    run_dirs: Sequence[Path], selection: Mapping[str, Any],
    protocol_dir: Path,
    expected_configs: Mapping[str, ExpectedConfig],
    expected_data_sha256: str | None = None,
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    expected_variant = selection["selected_variant"]
    expected_bits = [digit == "1" for digit in expected_variant[1:]]
    predictions: Dict[int, float] = {}
    sources = []
    observed_splits = set()
    commits = set()
    protocol_targets = load_protocol_targets(protocol_dir / "samples.csv")
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete pair-OOF run: {manifest_path}")
        if manifest.get("schema_version") != "prm_run_manifest_v1":
            raise ValueError(f"unsupported pair-OOF run manifest: {manifest_path}")
        if manifest.get("git", {}).get("dirty"):
            raise ValueError(f"dirty pair-OOF run: {manifest_path}")
        commit = manifest.get("git", {}).get("commit")
        if not isinstance(commit, str) or not commit:
            raise ValueError(f"pair-OOF run has no recorded code commit: {manifest_path}")
        expected_config = validate_manifest_config(
            manifest, expected_configs, manifest_path
        )
        validate_training_completion(manifest, manifest_path)
        validate_dart_assets(manifest, manifest_path)
        data_sha256 = manifest["data"]["data_sha256"]
        if expected_data_sha256 is not None and data_sha256 != expected_data_sha256:
            raise ValueError(f"dataset hash mismatch: {manifest_path}")
        bits = [bool(manifest["config"]["model_kwargs"][name]) for name in COMPONENTS]
        if bits != expected_bits:
            raise ValueError(f"run does not use selected architecture: {manifest_path}")
        split_id = manifest["split"]["split_id"]
        split_path = protocol_dir / "splits" / f"{split_id}.json"
        if manifest["split"].get("sha256") != file_sha256(split_path):
            raise ValueError(f"split hash mismatch: {manifest_path}")
        observed_splits.add(split_id)
        commits.add(commit)
        prediction_path = run_dir / "test_predictions.npz"
        indices, values, targets = load_prediction_array(
            prediction_path, expected_split_id=split_id
        )
        validate_protocol_targets(
            indices, targets, protocol_targets,
            context=f"pair-OOF predictions in {prediction_path}",
        )
        for index, value in zip(indices, values):
            if int(index) in predictions:
                raise ValueError(f"duplicate out-of-fold prediction for sample {index}")
            predictions[int(index)] = float(value)
        sources.append(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "prediction_sha256": file_sha256(prediction_path),
                "git": manifest["git"], "seed": int(manifest["seed"]),
                "split_id": split_id,
                "data_sha256": data_sha256,
                "split_sha256": manifest["split"].get("sha256"),
                "config_sha256": manifest["config_sha256"],
                "expected_config": str(expected_config.path),
            }
        )
    expected_splits = {f"pair_cv5_f{fold}" for fold in range(5)}
    if observed_splits != expected_splits:
        raise ValueError(f"pair-OOF folds mismatch: {sorted(observed_splits)}")
    if len(commits) != 1 or None in commits:
        raise ValueError("pair-OOF runs do not share one recorded code commit")
    expected_indices = set()
    for split_id in expected_splits:
        split = json.loads((protocol_dir / "splits" / f"{split_id}.json").read_text())
        expected_indices.update(int(index) for index in split["test"])
    if set(predictions) != expected_indices:
        raise ValueError("pair-OOF predictions do not cover the canonical retained set")
    return predictions, sources


def bootstrap_mean(
    values: Sequence[float], seed: int, draws: int = 20_000,
) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        raise ValueError("bootstrap mean requires at least two decision units")
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("bootstrap mean requires a finite one-dimensional sample")
    if draws < 1:
        raise ValueError("bootstrap mean requires at least one draw")
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        chunks.append(array[indices].mean(axis=1))
    means = np.concatenate(chunks)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(array.mean()), "std": float(array.std(ddof=1)),
        "ci_low": float(low), "ci_high": float(high), "n": len(array),
    }


def cluster_bootstrap_mean(
    rows: Sequence[Mapping[str, Any]], value_key: str, seed: int,
    draws: int = 20_000,
) -> Dict[str, Any]:
    """Bootstrap a row-weighted mean while retaining whole chemical pairs."""
    if draws < 1:
        raise ValueError("cluster bootstrap requires at least one draw")
    grouped: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    for row in rows:
        cluster = (str(row["host"]), str(row["dopant"]))
        grouped[cluster].append(float(row[value_key]))
    if len(grouped) < 2:
        raise ValueError("cluster bootstrap requires at least two chemical units")
    values = np.asarray(
        [value for cluster in grouped.values() for value in cluster], dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("cluster bootstrap requires finite values")
    totals = np.asarray([sum(cluster) for cluster in grouped.values()], dtype=float)
    counts = np.asarray([len(cluster) for cluster in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        indices = rng.integers(0, len(grouped), size=(size, len(grouped)))
        chunks.append(totals[indices].sum(axis=1) / counts[indices].sum(axis=1))
    means = np.concatenate(chunks)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()), "std": float(values.std(ddof=1)),
        "ci_low": float(low), "ci_high": float(high), "n": len(values),
        "n_clusters": len(grouped),
        "resampling_unit": "host_dopant_pair",
    }


def group_error_rows(
    samples: Sequence[Mapping[str, Any]], key: str,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample[key])].append(sample)
    rows = []
    for value, members in sorted(groups.items()):
        residual = np.asarray([member["prediction_eV"] - member["target_eV"] for member in members])
        rows.append(
            {
                "axis": key, "group": value, "n": len(members),
                "mae_eV": float(np.mean(np.abs(residual))),
                "rmse_eV": float(np.sqrt(np.mean(residual ** 2))),
                "bias_eV": float(np.mean(residual)),
            }
        )
    return rows


def summarize_error_heterogeneity(
    group_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize descriptive group errors without using them for selection."""
    output: Dict[str, Any] = {
        "basis": (
            "pair-held-out out-of-fold predictions; descriptive associations "
            "not used for model selection"
        )
    }
    for axis in ("host", "dopant"):
        members = [row for row in group_rows if row["axis"] == axis]
        if len(members) < 2:
            raise ValueError(f"{axis} error summary requires at least two groups")
        counts = np.asarray([int(row["n"]) for row in members], dtype=float)
        errors = np.asarray([float(row["mae_eV"]) for row in members], dtype=float)
        if (
            not np.isfinite(counts).all()
            or not np.isfinite(errors).all()
            or len(np.unique(counts)) < 2
            or len(np.unique(errors)) < 2
        ):
            raise ValueError(f"{axis} group-size/error association is undefined")
        worst = sorted(
            members, key=lambda row: (-float(row["mae_eV"]), str(row["group"]))
        )[:5]
        output[axis] = {
            "n_groups": len(members),
            "sample_count_mae_spearman": finite_spearman(
                counts,
                errors,
                context=f"{axis} sample-count/group-MAE Spearman correlation",
            ),
            "group_mae_mean_eV": float(np.mean(errors)),
            "group_mae_median_eV": float(np.median(errors)),
            "worst_groups": [
                {
                    "group": str(row["group"]),
                    "n": int(row["n"]),
                    "mae_eV": float(row["mae_eV"]),
                    "bias_eV": float(row["bias_eV"]),
                }
                for row in worst
            ],
        }
    for axis in ("defecttype", "site"):
        members = sorted(
            (row for row in group_rows if row["axis"] == axis),
            key=lambda row: str(row["group"]),
        )
        if not members:
            raise ValueError(f"{axis} error summary is empty")
        output[axis] = [
            {
                "group": str(row["group"]),
                "n": int(row["n"]),
                "mae_eV": float(row["mae_eV"]),
                "rmse_eV": float(row["rmse_eV"]),
                "bias_eV": float(row["bias_eV"]),
            }
            for row in members
        ]
    return output


def analyse_preferences(
    samples: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pairs: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        pairs[(str(sample["host"]), str(sample["dopant"]))].append(sample)
    pair_rows = []
    site_rows = []
    for (host, dopant), members in sorted(pairs.items()):
        by_type = {
            defect_type: [row for row in members if row["defecttype"] == defect_type]
            for defect_type in ("adsorbate", "interstitial")
        }
        for defect_type, typed in by_type.items():
            if len(typed) < 2:
                continue
            true_order = sorted(typed, key=lambda row: (row["target_eV"], row["sample_index"]))
            predicted_order = sorted(
                typed, key=lambda row: (row["prediction_eV"], row["sample_index"])
            )
            selected = predicted_order[0]
            true_best = true_order[0]
            true_best_indices = {
                row["sample_index"] for row in true_order
                if row["target_eV"] <= true_best["target_eV"] + ENERGY_TIE_TOLERANCE_EV
            }
            top_two = {row["sample_index"] for row in predicted_order[:2]}
            top2_eligible = len(typed) >= 3
            site_rows.append(
                {
                    "host": host, "dopant": dopant, "dopant_Z": atomic_numbers[dopant],
                    "dopant_period": dopant_period(dopant), "defecttype": defect_type,
                    "n_sites": len(typed), "true_best_site": true_best["site"],
                    "n_tied_true_best_sites": len(true_best_indices),
                    "predicted_best_site": selected["site"],
                    "exact_site_correct": int(
                        selected["sample_index"] in true_best_indices
                    ),
                    "top2_eligible": int(top2_eligible),
                    "true_best_in_predicted_top2": (
                        int(bool(true_best_indices & top_two))
                        if top2_eligible else None
                    ),
                    "screening_regret_eV": float(selected["target_eV"] - true_best["target_eV"]),
                }
            )
        if not by_type["adsorbate"] or not by_type["interstitial"]:
            continue
        true_min = {
            name: min(rows, key=lambda row: (row["target_eV"], row["sample_index"]))
            for name, rows in by_type.items()
        }
        predicted_min = {
            name: min(rows, key=lambda row: (row["prediction_eV"], row["sample_index"]))
            for name, rows in by_type.items()
        }
        true_margin = true_min["interstitial"]["target_eV"] - true_min["adsorbate"]["target_eV"]
        predicted_margin = (
            predicted_min["interstitial"]["prediction_eV"]
            - predicted_min["adsorbate"]["prediction_eV"]
        )
        preference_eligible = abs(true_margin) > ENERGY_TIE_TOLERANCE_EV
        true_preference = (
            "adsorbate" if true_margin > ENERGY_TIE_TOLERANCE_EV
            else "interstitial" if true_margin < -ENERGY_TIE_TOLERANCE_EV
            else "tie"
        )
        predicted_preference = "adsorbate" if predicted_margin >= 0 else "interstitial"
        true_global = min(members, key=lambda row: (row["target_eV"], row["sample_index"]))
        true_global_indices = {
            row["sample_index"] for row in members
            if row["target_eV"] <= (
                true_global["target_eV"] + ENERGY_TIE_TOLERANCE_EV
            )
        }
        predicted_global = min(
            members, key=lambda row: (row["prediction_eV"], row["sample_index"])
        )
        pair_rows.append(
            {
                "host": host, "dopant": dopant, "dopant_Z": atomic_numbers[dopant],
                "dopant_period": dopant_period(dopant), "n_sites": len(members),
                "true_preference": true_preference,
                "predicted_preference": predicted_preference,
                "preference_eligible": int(preference_eligible),
                "preference_correct": (
                    int(true_preference == predicted_preference)
                    if preference_eligible else None
                ),
                "true_margin_eV": float(true_margin),
                "predicted_margin_eV": float(predicted_margin),
                "margin_absolute_error_eV": float(abs(predicted_margin - true_margin)),
                "true_global_best": f"{true_global['defecttype']}:{true_global['site']}",
                "n_tied_true_global_best_sites": len(true_global_indices),
                "predicted_global_best": f"{predicted_global['defecttype']}:{predicted_global['site']}",
                "global_site_correct": int(
                    predicted_global["sample_index"] in true_global_indices
                ),
                "global_screening_regret_eV": float(
                    predicted_global["target_eV"] - true_global["target_eV"]
                ),
            }
        )
    return pair_rows, site_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path,
        default=ROOT / "artifacts/prm_results/factorial/selection.json",
    )
    parser.add_argument("--factorial-bundle", type=Path, default=None)
    parser.add_argument(
        "--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts/prm_results/materials",
    )
    parser.add_argument(
        "--promoted-config-root", type=Path,
        default=ROOT / "configs/prm/promoted",
    )
    args = parser.parse_args()

    selection, factorial_bundle = load_verified_factorial_selection(
        args.selection, args.factorial_bundle
    )
    variant = selection["selected_variant"]
    protocol_manifest = json.loads((args.protocol_dir / "manifest.json").read_text())
    if selection["data_sha256"] != protocol_manifest["data_sha256"]:
        raise ValueError("factorial selection does not match the frozen protocol dataset")
    expected_configs = load_expected_configs(
        sorted(
            (args.promoted_config_root.resolve() / variant / "transfer").glob(
                "pair_cv5_*.yaml"
            )
        )
    )
    run_dirs = sorted(
        path for path in args.result_root.resolve().glob(
            f"selected/{variant}/transfer/pair_cv5_f*/seed242"
        )
        if path.is_dir()
    )
    predictions, sources = load_oof_predictions(
        run_dirs, selection, args.protocol_dir.resolve(),
        expected_configs,
        protocol_manifest["data_sha256"],
    )
    sample_table = read_sample_table(args.protocol_dir / "samples.csv")
    sample_rows = []
    for index, prediction in sorted(predictions.items()):
        row = dict(sample_table[index])
        row["prediction_eV"] = prediction
        row["residual_eV"] = prediction - row["target_eV"]
        row["absolute_error_eV"] = abs(row["residual_eV"])
        sample_rows.append(row)

    pair_rows, site_rows = analyse_preferences(sample_rows)
    preference_rows = [row for row in pair_rows if row["preference_eligible"]]
    top2_site_rows = [row for row in site_rows if row["top2_eligible"]]
    residual = np.asarray([row["residual_eV"] for row in sample_rows])
    group_rows = []
    for axis in ("host", "dopant", "defecttype", "site"):
        group_rows.extend(group_error_rows(sample_rows, axis))
    collector_git = git_snapshot()
    require_clean_git_snapshot(collector_git, context="materials collection")
    summary = {
        "schema_version": "prm_materials_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector_git": collector_git,
        "selection": {
            "path": str(args.selection.resolve()), "sha256": file_sha256(args.selection),
            "selected_variant": variant, "selection_data": selection["selection_data"],
            "factorial_bundle_sha256": selection["factorial_bundle_sha256"],
            "factorial_training_commit": selection["training_commits"][0],
            "factorial_collector_git": factorial_bundle["collector_git"],
        },
        "data_sha256": protocol_manifest["data_sha256"],
        "training_commit": sources[0]["git"]["commit"],
        "sources": sources,
        "sample_oof": {
            "n": len(sample_rows), "mae_eV": float(np.mean(np.abs(residual))),
            "rmse_eV": float(np.sqrt(np.mean(residual ** 2))),
            "bias_eV": float(np.mean(residual)),
        },
        "defect_type_preference": {
            "eligibility": {
                "binary_accuracy": "non-tied reference adsorbate/interstitial minima",
                "true_energy_tie_tolerance_eV": ENERGY_TIE_TOLERANCE_EV,
                "n_pairs_with_both_defect_types": len(pair_rows),
                "n_binary_eligible_pairs": len(preference_rows),
            },
            "accuracy": cluster_bootstrap_mean(
                preference_rows, "preference_correct", seed=20263001,
            ),
            "margin_mae_eV": cluster_bootstrap_mean(
                pair_rows, "margin_absolute_error_eV", seed=20263002,
            ),
            "margin_spearman": finite_spearman(
                [row["true_margin_eV"] for row in pair_rows],
                [row["predicted_margin_eV"] for row in pair_rows],
                context="adsorbate-interstitial margin Spearman correlation",
            ),
            "global_screening_regret_eV": cluster_bootstrap_mean(
                pair_rows, "global_screening_regret_eV", seed=20263003,
            ),
            "global_exact_site_accuracy": cluster_bootstrap_mean(
                pair_rows, "global_site_correct", seed=20263004,
            ),
        },
        "within_defect_type_site_selection": {
            "eligibility": {
                "exact_and_regret": "at least two candidate sites",
                "top2": "at least three candidate sites",
                "true_energy_tie_tolerance_eV": ENERGY_TIE_TOLERANCE_EV,
                "n_exact_candidate_sets": len(site_rows),
                "n_top2_candidate_sets": len(top2_site_rows),
            },
            "exact_accuracy": cluster_bootstrap_mean(
                site_rows, "exact_site_correct", seed=20263005,
            ),
            "top2_accuracy": cluster_bootstrap_mean(
                top2_site_rows, "true_best_in_predicted_top2", seed=20263006,
            ),
            "screening_regret_eV": cluster_bootstrap_mean(
                site_rows, "screening_regret_eV", seed=20263007,
            ),
        },
        "error_heterogeneity": summarize_error_heterogeneity(group_rows),
        "interpretation_boundary": (
            "Out-of-fold associations and screening regret are descriptive; "
            "they are not causal mechanisms or external DFT validation."
        ),
    }
    strict_json(summary)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "sample_predictions.csv", sample_rows)
    write_csv(out_dir / "pair_preferences.csv", pair_rows)
    write_csv(out_dir / "site_selection.csv", site_rows)
    write_csv(out_dir / "group_errors.csv", group_rows)
    summary["output_sha256"] = {
        name: file_sha256(out_dir / filename)
        for name, filename in {
            "sample_predictions": "sample_predictions.csv",
            "pair_preferences": "pair_preferences.csv",
            "site_selection": "site_selection.csv",
            "group_errors": "group_errors.csv",
        }.items()
    }
    (out_dir / "summary.json").write_text(strict_json(summary) + "\n")
    print(strict_json(summary))


if __name__ == "__main__":
    main()
