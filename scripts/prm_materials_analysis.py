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
from scipy.stats import spearmanr

from src.prm_provenance import (
    ExpectedConfig,
    load_verified_factorial_selection,
    load_protocol_targets,
    load_expected_configs,
    validate_dart_assets,
    validate_manifest_config,
    validate_protocol_targets,
    validate_training_completion,
)


ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = ("use_gated_pooling", "use_env_enrichment", "use_prenorm_local")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prediction_array(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"].item()) != "prm_predictions_v1":
            raise ValueError(f"unsupported prediction schema: {path}")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        predictions = np.asarray(archive["preds"], dtype=float)
        targets = np.asarray(archive["targets"])
    return indices, predictions, targets


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
        indices, values, targets = load_prediction_array(prediction_path)
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
            if not typed:
                continue
            true_order = sorted(typed, key=lambda row: (row["target_eV"], row["sample_index"]))
            predicted_order = sorted(
                typed, key=lambda row: (row["prediction_eV"], row["sample_index"])
            )
            selected = predicted_order[0]
            true_best = true_order[0]
            top_two = {row["sample_index"] for row in predicted_order[:2]}
            site_rows.append(
                {
                    "host": host, "dopant": dopant, "dopant_Z": atomic_numbers[dopant],
                    "dopant_period": dopant_period(dopant), "defecttype": defect_type,
                    "n_sites": len(typed), "true_best_site": true_best["site"],
                    "predicted_best_site": selected["site"],
                    "exact_site_correct": int(selected["sample_index"] == true_best["sample_index"]),
                    "true_best_in_predicted_top2": int(true_best["sample_index"] in top_two),
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
        true_preference = "adsorbate" if true_margin >= 0 else "interstitial"
        predicted_preference = "adsorbate" if predicted_margin >= 0 else "interstitial"
        true_global = min(members, key=lambda row: (row["target_eV"], row["sample_index"]))
        predicted_global = min(
            members, key=lambda row: (row["prediction_eV"], row["sample_index"])
        )
        pair_rows.append(
            {
                "host": host, "dopant": dopant, "dopant_Z": atomic_numbers[dopant],
                "dopant_period": dopant_period(dopant), "n_sites": len(members),
                "true_preference": true_preference,
                "predicted_preference": predicted_preference,
                "preference_correct": int(true_preference == predicted_preference),
                "true_margin_eV": float(true_margin),
                "predicted_margin_eV": float(predicted_margin),
                "margin_absolute_error_eV": float(abs(predicted_margin - true_margin)),
                "true_global_best": f"{true_global['defecttype']}:{true_global['site']}",
                "predicted_global_best": f"{predicted_global['defecttype']}:{predicted_global['site']}",
                "global_site_correct": int(
                    true_global["sample_index"] == predicted_global["sample_index"]
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
    preference_correct = [row["preference_correct"] for row in pair_rows]
    margin_errors = [row["margin_absolute_error_eV"] for row in pair_rows]
    global_regret = [row["global_screening_regret_eV"] for row in pair_rows]
    exact_site = [row["exact_site_correct"] for row in site_rows]
    site_regret = [row["screening_regret_eV"] for row in site_rows]
    residual = np.asarray([row["residual_eV"] for row in sample_rows])
    summary = {
        "schema_version": "prm_materials_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector_git": git_snapshot(),
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
            "accuracy": bootstrap_mean(preference_correct, seed=20263001),
            "margin_mae_eV": bootstrap_mean(margin_errors, seed=20263002),
            "margin_spearman": float(
                spearmanr(
                    [row["true_margin_eV"] for row in pair_rows],
                    [row["predicted_margin_eV"] for row in pair_rows],
                ).statistic
            ),
            "global_screening_regret_eV": bootstrap_mean(global_regret, seed=20263003),
            "global_exact_site_accuracy": bootstrap_mean(
                [row["global_site_correct"] for row in pair_rows], seed=20263004,
            ),
        },
        "within_defect_type_site_selection": {
            "exact_accuracy": bootstrap_mean(exact_site, seed=20263005),
            "top2_accuracy": bootstrap_mean(
                [row["true_best_in_predicted_top2"] for row in site_rows], seed=20263006,
            ),
            "screening_regret_eV": bootstrap_mean(site_regret, seed=20263007),
        },
        "interpretation_boundary": (
            "Out-of-fold associations and screening regret are descriptive; "
            "they are not causal mechanisms or external DFT validation."
        ),
    }

    group_rows = []
    for axis in ("host", "dopant", "defecttype", "site"):
        group_rows.extend(group_error_rows(sample_rows, axis))
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
