"""Validation-independent ensemble calibration and UQ evaluation for PRM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from src.prm_metrics import finite_spearman
from src.prm_provenance import (
    ExpectedConfig,
    archive_training_artifacts,
    load_verified_factorial_selection,
    load_protocol_targets,
    load_expected_configs,
    require_clean_git_snapshot,
    validate_dart_assets,
    validate_manifest_config,
    validate_protocol_targets,
    validate_training_completion,
)

COMPONENTS = ("use_gated_pooling", "use_env_enrichment", "use_prenorm_local")
CALIBRATION_SUBSET_SEED = 6201


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
    return {"commit": commit or None, "dirty": bool(status), "status_porcelain": status.splitlines()}


def load_aligned_predictions(
    run_dirs: Sequence[Path], split_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_indices = None
    reference_targets = None
    members = []
    for run_dir in run_dirs:
        path = run_dir / f"{split_name}_predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            expected_fields = {
                "schema_version", "split_id", "split", "indices", "preds", "targets",
            }
            if set(archive.files) != expected_fields:
                raise ValueError(f"prediction fields are incomplete: {path}")
            if str(archive["schema_version"].item()) != "prm_predictions_v1":
                raise ValueError(f"unsupported prediction schema: {path}")
            if str(archive["split"].item()) != split_name:
                raise ValueError(f"prediction split mismatch: {path}")
            if str(archive["split_id"].item()) != "uq_calibration_s62":
                raise ValueError(f"prediction split ID mismatch: {path}")
            raw_indices = np.asarray(archive["indices"])
            targets = np.asarray(archive["targets"])
            predictions = np.asarray(archive["preds"], dtype=float)
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
            or targets.ndim != 1
            or predictions.ndim != 1
            or len(indices) == 0
            or len(indices) != len(targets)
            or len(indices) != len(predictions)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError(f"unaligned prediction vectors: {path}")
        if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
            raise ValueError(f"non-finite prediction values: {path}")
        order = np.argsort(indices)
        indices, targets, predictions = indices[order], targets[order], predictions[order]
        if reference_indices is None:
            reference_indices, reference_targets = indices, targets
        elif not np.array_equal(indices, reference_indices):
            raise ValueError(f"member indices do not align: {path}")
        elif targets.dtype != reference_targets.dtype or not np.array_equal(
            targets, reference_targets
        ):
            raise ValueError(f"member targets do not align: {path}")
        members.append(predictions)
    if reference_indices is None or reference_targets is None:
        raise ValueError("no ensemble members supplied")
    return reference_indices, reference_targets, np.stack(members)


def calibration_subsets(
    indices: np.ndarray, seed: int = CALIBRATION_SUBSET_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly split calibration rows with a fixed seed and no labels."""
    indices = np.asarray(indices)
    if (
        indices.ndim != 1
        or len(indices) < 2
        or not np.issubdtype(indices.dtype, np.integer)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(
            "calibration indices must be a unique one-dimensional integer vector"
        )
    order = np.random.default_rng(seed).permutation(len(indices))
    variance_fit = np.sort(order[::2])
    conformal = np.sort(order[1::2])
    if not len(variance_fit) or not len(conformal):
        raise ValueError("calibration partition must contain at least two samples")
    return variance_fit, conformal


def calibrated_sigma(raw_std: np.ndarray, scale: float, floor: float) -> np.ndarray:
    raw_std = np.asarray(raw_std, dtype=float)
    if (
        raw_std.ndim != 1
        or not np.isfinite(raw_std).all()
        or np.any(raw_std < 0.0)
        or not math.isfinite(scale)
        or not math.isfinite(floor)
        or scale < 0.0
        or floor <= 0.0
    ):
        raise ValueError("variance calibration parameters must be finite and admissible")
    return np.sqrt(np.square(scale * raw_std) + floor ** 2)


def gaussian_nll(targets: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if (
        targets.shape != mean.shape
        or targets.shape != sigma.shape
        or targets.ndim != 1
        or len(targets) == 0
        or not np.isfinite(targets).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(sigma).all()
        or np.any(sigma <= 0.0)
    ):
        raise ValueError("Gaussian NLL requires finite aligned vectors and positive sigma")
    residual = targets - mean
    value = float(
        np.mean(
            0.5 * np.log(2.0 * np.pi * sigma ** 2)
            + 0.5 * (residual / sigma) ** 2
        )
    )
    if not math.isfinite(value):
        raise ValueError("Gaussian NLL is non-finite")
    return value


def fit_variance_calibration(
    targets: np.ndarray, mean: np.ndarray, raw_std: np.ndarray,
) -> Dict[str, Any]:
    targets = np.asarray(targets, dtype=float)
    mean = np.asarray(mean, dtype=float)
    raw_std = np.asarray(raw_std, dtype=float)
    if (
        targets.shape != mean.shape
        or targets.shape != raw_std.shape
        or targets.ndim != 1
        or len(targets) == 0
        or not np.isfinite(targets).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(raw_std).all()
        or np.any(raw_std < 0.0)
    ):
        raise ValueError("variance calibration requires finite aligned vectors")
    residual = targets - mean
    residual_scale = max(float(np.sqrt(np.mean(residual ** 2))), 1e-4)

    def objective(parameters: np.ndarray) -> float:
        scale, floor = np.exp(parameters)
        return gaussian_nll(targets, mean, calibrated_sigma(raw_std, scale, floor))

    result = minimize(
        objective,
        x0=np.log([1.0, max(0.25 * residual_scale, 1e-4)]),
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0), (math.log(1e-5), math.log(20.0 * residual_scale))],
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"variance calibration failed: {result.message}")
    scale, floor = np.exp(result.x)
    return {
        "scale": float(scale), "floor_eV": float(floor),
        "calibration_nll": float(result.fun),
        "optimizer": "L-BFGS-B on Gaussian NLL",
        "success": bool(result.success),
    }


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    scores = np.sort(np.asarray(scores, dtype=float))
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("conformal scores must be a non-empty finite vector")
    rank = min(len(scores), int(math.ceil((len(scores) + 1) * coverage)))
    return float(scores[rank - 1])


def gaussian_crps(targets: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if (
        targets.shape != mean.shape
        or targets.shape != sigma.shape
        or targets.ndim != 1
        or len(targets) == 0
        or not np.isfinite(targets).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(sigma).all()
        or np.any(sigma <= 0.0)
    ):
        raise ValueError("Gaussian CRPS requires finite aligned vectors and positive sigma")
    z = (targets - mean) / sigma
    phi = np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if (
        targets.shape != predictions.shape
        or targets.ndim != 1
        or len(targets) < 2
        or not np.isfinite(targets).all()
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("regression metrics require finite aligned vectors")
    residual = predictions - targets
    denominator = float(np.sum((targets - np.mean(targets)) ** 2))
    if denominator <= 0.0:
        raise ValueError("regression metrics require non-constant targets")
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - np.sum(residual ** 2) / denominator),
        "spearman": finite_spearman(
            targets, predictions, context="point-prediction Spearman correlation"
        ),
    }


def risk_coverage(
    targets: np.ndarray, predictions: np.ndarray, uncertainty: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if (
        targets.shape != predictions.shape
        or targets.shape != uncertainty.shape
        or targets.ndim != 1
        or len(targets) == 0
        or not np.isfinite(targets).all()
        or not np.isfinite(predictions).all()
        or not np.isfinite(uncertainty).all()
        or np.any(uncertainty < 0.0)
    ):
        raise ValueError("risk coverage requires finite aligned vectors")
    error = np.abs(predictions - targets)
    order = np.argsort(uncertainty, kind="stable")
    sorted_error = error[order]
    coverage = np.arange(1, len(error) + 1, dtype=float) / len(error)
    risk = np.cumsum(sorted_error) / np.arange(1, len(error) + 1)
    oracle_error = np.sort(error)
    oracle_risk = np.cumsum(oracle_error) / np.arange(1, len(error) + 1)
    aurc = float(np.mean(risk))
    oracle_aurc = float(np.mean(oracle_risk))
    rows = [
        {
            "coverage": float(c), "mae_risk_eV": float(r),
            "oracle_mae_risk_eV": float(o),
        }
        for c, r, o in zip(coverage, risk, oracle_risk)
    ]
    at_coverage = {}
    for requested in (0.25, 0.50, 0.75, 0.90, 1.00):
        count = max(1, int(math.floor(requested * len(error))))
        at_coverage[f"{requested:.2f}"] = {
            "n": count, "mae_eV": float(risk[count - 1]),
        }
    return {
        "aurc_eV": aurc,
        "oracle_aurc_eV": oracle_aurc,
        "excess_aurc_eV": aurc - oracle_aurc,
        "risk_at_coverage": at_coverage,
    }, rows


def validate_runs(
    run_dirs: Sequence[Path], selection: Mapping[str, Any], expected_members: int,
    expected_configs: Mapping[str, ExpectedConfig],
    expected_data_sha256: str,
    expected_split_sha256: str,
) -> List[Dict[str, Any]]:
    if len(run_dirs) != expected_members:
        raise ValueError(f"expected {expected_members} UQ members, found {len(run_dirs)}")
    selected = selection["selected_variant"]
    expected_bits = [digit == "1" for digit in selected[1:]]
    sources = []
    seeds = set()
    split_ids = set()
    split_hashes = set()
    data_hashes = set()
    commits = set()
    for run_dir in run_dirs:
        path = run_dir / "run_manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete UQ member: {path}")
        if manifest.get("schema_version") != "prm_run_manifest_v1":
            raise ValueError(f"unsupported UQ run manifest: {path}")
        if manifest.get("git", {}).get("dirty"):
            raise ValueError(f"dirty UQ member is not admissible: {path}")
        commit = manifest.get("git", {}).get("commit")
        if not isinstance(commit, str) or not commit:
            raise ValueError(f"UQ member has no recorded code commit: {path}")
        expected_config = validate_manifest_config(manifest, expected_configs, path)
        validate_training_completion(manifest, path)
        validate_dart_assets(manifest, path)
        bits = [bool(manifest["config"]["model_kwargs"][name]) for name in COMPONENTS]
        if bits != expected_bits:
            raise ValueError(f"UQ member does not use selected architecture: {path}")
        if "calibration_predictions" not in manifest.get("outputs", {}):
            raise ValueError(f"UQ member lacks held-out calibration predictions: {path}")
        seeds.add(int(manifest["seed"]))
        split_ids.add(manifest["split"]["split_id"])
        split_hashes.add(manifest["split"].get("sha256"))
        data_hashes.add(manifest["data"]["data_sha256"])
        commits.add(commit)
        if manifest["data"]["data_sha256"] != expected_data_sha256:
            raise ValueError(f"UQ dataset hash mismatch: {path}")
        if manifest["split"].get("sha256") != expected_split_sha256:
            raise ValueError(f"UQ split hash mismatch: {path}")
        sources.append(
            {
                "manifest": str(path), "manifest_sha256": file_sha256(path),
                "git": manifest["git"], "seed": int(manifest["seed"]),
                "data_sha256": manifest["data"]["data_sha256"],
                "split_id": manifest["split"]["split_id"],
                "split_sha256": manifest["split"].get("sha256"),
                "config_sha256": manifest["config_sha256"],
                "expected_config": str(expected_config.path),
            }
        )
    if len(seeds) != expected_members:
        raise ValueError("UQ ensemble seeds are not unique")
    if split_ids != {"uq_calibration_s62"}:
        raise ValueError(f"unexpected UQ split IDs: {sorted(split_ids)}")
    if len(split_hashes) != 1 or None in split_hashes:
        raise ValueError("UQ members do not share a recorded split definition")
    if len(data_hashes) != 1:
        raise ValueError("UQ members use different datasets")
    if len(commits) != 1 or None in commits:
        raise ValueError("UQ members do not share one recorded code commit")
    return sources


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
        "--out-dir", type=Path, default=ROOT / "artifacts/prm_results/uq",
    )
    parser.add_argument(
        "--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2",
    )
    parser.add_argument(
        "--promoted-config-root", type=Path,
        default=ROOT / "configs/prm/promoted",
    )
    parser.add_argument("--expected-members", type=int, default=5)
    args = parser.parse_args()
    if args.expected_members != 5:
        raise ValueError("the frozen UQ contract requires exactly five ensemble members")

    selection, factorial_bundle = load_verified_factorial_selection(
        args.selection, args.factorial_bundle
    )
    variant = selection["selected_variant"]
    protocol_dir = args.protocol_dir.resolve()
    protocol = json.loads((protocol_dir / "manifest.json").read_text())
    if selection["data_sha256"] != protocol["data_sha256"]:
        raise ValueError("factorial selection does not match the frozen protocol dataset")
    expected_split_sha256 = file_sha256(
        protocol_dir / "splits/uq_calibration_s62.json"
    )
    expected_configs = load_expected_configs(
        sorted((args.promoted_config_root.resolve() / variant / "uq").glob("*.yaml"))
    )
    run_dirs = sorted(
        path for path in args.result_root.resolve().glob(
            f"selected/{variant}/uq/uq_calibration_s62/seed*"
        )
        if path.is_dir()
    )
    sources = validate_runs(
        run_dirs,
        selection,
        args.expected_members,
        expected_configs,
        protocol["data_sha256"],
        expected_split_sha256,
    )
    cal_indices, cal_targets, cal_members = load_aligned_predictions(run_dirs, "calibration")
    test_indices, test_targets, test_members = load_aligned_predictions(run_dirs, "test")
    protocol_targets = load_protocol_targets(protocol_dir / "samples.csv")
    cal_targets = validate_protocol_targets(
        cal_indices, cal_targets, protocol_targets,
        context="UQ calibration predictions",
    )
    test_targets = validate_protocol_targets(
        test_indices, test_targets, protocol_targets,
        context="UQ test predictions",
    )

    cal_mean = cal_members.mean(axis=0)
    cal_raw_std = cal_members.std(axis=0, ddof=1)
    test_mean = test_members.mean(axis=0)
    test_raw_std = test_members.std(axis=0, ddof=1)
    variance_rows, conformal_rows = calibration_subsets(cal_indices)
    variance = fit_variance_calibration(
        cal_targets[variance_rows], cal_mean[variance_rows], cal_raw_std[variance_rows],
    )
    cal_sigma = calibrated_sigma(cal_raw_std, variance["scale"], variance["floor_eV"])
    test_sigma = calibrated_sigma(test_raw_std, variance["scale"], variance["floor_eV"])

    conformal_scores = (
        np.abs(cal_targets[conformal_rows] - cal_mean[conformal_rows])
        / cal_sigma[conformal_rows]
    )
    interval_rows = []
    intervals = {}
    for coverage in (0.50, 0.80, 0.90, 0.95):
        quantile = conformal_quantile(conformal_scores, coverage)
        half_width = quantile * test_sigma
        observed = float(np.mean(np.abs(test_targets - test_mean) <= half_width))
        row = {
            "nominal_coverage": coverage,
            "observed_test_coverage": observed,
            "mean_test_width_eV": float(np.mean(2.0 * half_width)),
            "conformal_quantile": quantile,
            "n_conformal_calibration": len(conformal_rows),
        }
        interval_rows.append(row)
        intervals[f"{coverage:.2f}"] = row

    test_crps = gaussian_crps(test_targets, test_mean, test_sigma)
    point_metrics = regression_metrics(test_targets, test_mean)
    uncertainty_error_spearman = finite_spearman(
        test_sigma,
        np.abs(test_targets - test_mean),
        context="uncertainty-error Spearman correlation",
    )
    risk_metrics, risk_rows = risk_coverage(
        test_targets, test_mean, test_sigma,
    )
    collector_git = git_snapshot()
    require_clean_git_snapshot(collector_git, context="UQ collection")
    metrics = {
        "schema_version": "prm_uq_results_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector_git": collector_git,
        "selection": {
            "path": str(args.selection.resolve()),
            "sha256": file_sha256(args.selection),
            "selected_variant": variant,
            "selection_data": selection["selection_data"],
            "factorial_bundle_sha256": selection["factorial_bundle_sha256"],
            "factorial_training_commit": selection["training_commits"][0],
            "factorial_collector_git": factorial_bundle["collector_git"],
        },
        "data_sha256": sources[0]["data_sha256"],
        "split_sha256": sources[0]["split_sha256"],
        "calibration_contract": {
            "split_id": "uq_calibration_s62",
            "dedicated_calibration_partition": len(cal_indices),
            "variance_fit_subset": len(variance_rows),
            "conformal_subset": len(conformal_rows),
            "subset_rule": (
                "fixed-seed random partition of calibration rows without labels"
            ),
            "subset_seed": CALIBRATION_SUBSET_SEED,
            "variance_calibration": variance,
        },
        "n_members": len(run_dirs),
        "training_commit": sources[0]["git"]["commit"],
        "member_sources": sources,
        "test": {
            "n": len(test_indices),
            "point_prediction": point_metrics,
            "gaussian_nll": gaussian_nll(test_targets, test_mean, test_sigma),
            "mean_gaussian_crps_eV": float(np.mean(test_crps)),
            "uncertainty_absolute_error_spearman": uncertainty_error_spearman,
            "intervals": intervals,
            "selective_prediction": risk_metrics,
        },
    }
    strict_json(metrics)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    archived_runs = archive_training_artifacts(
        [run_dir / "run_manifest.json" for run_dir in run_dirs],
        out_dir / "runs",
        repository_root=ROOT,
    )
    if len(archived_runs) != args.expected_members:
        raise ValueError("UQ evidence archive does not contain every ensemble member")
    metrics.update(
        {
            "archive_policy": {
                "included": [
                    "run_manifest.json", "metrics.json", "split_indices.npz",
                    "val_predictions.npz", "calibration_predictions.npz",
                    "test_predictions.npz",
                ],
                "checkpoint": "SHA-256 recorded; binary retained outside Git",
            },
            "n_archived_runs": len(archived_runs),
            "archived_runs": archived_runs,
        }
    )
    np.savez_compressed(
        out_dir / "predictions.npz",
        schema_version=np.asarray("prm_uq_predictions_v1"),
        calibration_indices=cal_indices,
        calibration_targets=cal_targets,
        calibration_member_predictions=cal_members,
        calibration_mean=cal_mean,
        calibration_raw_std=cal_raw_std,
        calibration_sigma=cal_sigma,
        variance_fit_rows=variance_rows,
        conformal_rows=conformal_rows,
        test_indices=test_indices,
        test_targets=test_targets,
        test_member_predictions=test_members,
        test_mean=test_mean,
        test_raw_std=test_raw_std,
        test_sigma=test_sigma,
    )
    write_csv(out_dir / "interval_calibration.csv", interval_rows)
    write_csv(out_dir / "risk_coverage.csv", risk_rows)
    metrics["output_sha256"] = {
        name: file_sha256(out_dir / filename)
        for name, filename in {
            "predictions": "predictions.npz",
            "interval_calibration": "interval_calibration.csv",
            "risk_coverage": "risk_coverage.csv",
        }.items()
    }
    (out_dir / "metrics.json").write_text(strict_json(metrics) + "\n")
    print(strict_json(metrics["test"]))


if __name__ == "__main__":
    main()
