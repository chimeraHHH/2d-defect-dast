"""W1 extensivity diagnostic (frozen preregistration, milestone J-R3).

Tests whether the additive-readout SchNet's signed out-of-fold residual
scales with system size while the mean-readout arm and the repaired DART do
not — the preregistered mechanism behind conclusion C3 (defect formation
energy is unsuited to atomwise additive pooling).

Inputs are hash-verified archives only: the additive SchNet host-CV campaign
runs, the 15-run mean-readout sensitivity bundle, and the repaired G2 DART
OOF bound by the accepted ``prm_g2_acceptance_v1``.  The frozen analysis is a
Huber regression of the signed OOF residual on the total atom count with
host-fold indicators, host-clustered bootstrap with 2,000 draws and seed
20260810 shared across arms so the additive-minus-DART slope difference is a
paired draw.  Outputs are exclusive-create.

Precondition: the committed pipeline-independence audit
(``artifacts/prm_g3/w1_pipeline_independence_audit.md``).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.prm_g3_physics_analysis as g3mod  # noqa: E402
from scripts.prm_g3_physics_analysis import (  # noqa: E402
    EXPECTED_DATA_SHA256,
    IRLS_DIAGNOSTICS,
    bootstrap_two_sided_p,
    file_sha256,
    fit_huber_fixed_effects,
    load_split,
)

SCHNET_SEEDS = (342, 343, 344)
DART_SEEDS = (242, 243, 244)
FOLDS = tuple(range(5))
EXPECTED_N = 10_224


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json(payload: Any) -> str:
    return json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n"


def resolve_alias(value: str) -> Path:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError(f"unresolved path alias: {value}")
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_prediction_npz(
    path: Path, expected_split_id: str, expected_sha256: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"].item()) != "prm_predictions_v1":
            raise ValueError(f"prediction schema mismatch: {path}")
        if str(archive["split_id"].item()) != expected_split_id:
            raise ValueError(f"prediction split mismatch: {path}")
        if str(archive["split"].item()) != "test":
            raise ValueError(f"prediction partition mismatch: {path}")
        return (
            np.asarray(archive["indices"], dtype=np.int64),
            np.asarray(archive["preds"], dtype=float),
            np.asarray(archive["targets"], dtype=float),
        )


def seed_averaged_arm(
    run_dirs: Mapping[int, list[tuple[Path, str | None]]],
    protocol_dir: Path, prefix: str,
    target_by_index: Mapping[int, float],
) -> tuple[dict[int, float], dict[int, int]]:
    """Average per-sample predictions across seeds inside each fold."""
    predictions: dict[int, float] = {}
    fold_by_index: dict[int, int] = {}
    for fold in FOLDS:
        split = load_split(protocol_dir, f"{prefix}_cv5_f{fold}")
        expected = np.asarray(sorted(int(i) for i in split["test"]), dtype=np.int64)
        per_seed = []
        for path, expected_sha in run_dirs[fold]:
            indices, preds, targets = load_prediction_npz(
                path, f"{prefix}_cv5_f{fold}", expected_sha,
            )
            order = np.argsort(indices)
            indices, preds, targets = indices[order], preds[order], targets[order]
            if not np.array_equal(indices, expected):
                raise ValueError(f"fold {fold} indices differ from the frozen split")
            expected_targets = np.asarray(
                [target_by_index[int(i)] for i in indices]
            )
            if not np.allclose(targets, expected_targets, rtol=0.0, atol=1e-5):
                raise ValueError(f"fold {fold} targets differ from the protocol")
            per_seed.append(preds)
        mean_preds = np.mean(np.column_stack(per_seed), axis=1)
        for index, value in zip(expected.tolist(), mean_preds.tolist()):
            predictions[index] = float(value)
            fold_by_index[index] = fold
    if len(predictions) != EXPECTED_N:
        raise ValueError(
            f"{prefix} arm covers {len(predictions)} rows, expected {EXPECTED_N}"
        )
    return predictions, fold_by_index


def slope_fit(
    residual: np.ndarray, natoms: np.ndarray, fold: np.ndarray,
    weights: np.ndarray | None = None,
    extra: np.ndarray | None = None,
) -> float:
    columns = [natoms.astype(float)]
    for level in range(1, 5):
        columns.append((fold == level).astype(float))
    if extra is not None:
        columns.append(np.asarray(extra, dtype=float))
    fit = fit_huber_fixed_effects(
        residual, np.column_stack(columns), [], base_weights=weights,
    )
    return float(fit["beta"][0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(ROOT / "artifacts/prm_protocol_v2/samples.csv"))
    parser.add_argument("--protocol-dir", default=str(ROOT / "artifacts/prm_protocol_v2"))
    parser.add_argument("--schnet-root", default=str(
        ROOT / "artifacts/prm_results/comparison/runs/baselines/schnet"
    ))
    parser.add_argument("--readout-bundle", default=str(
        ROOT / "artifacts/prm_results/sensitivity/schnet_readout"
    ))
    parser.add_argument("--g2-acceptance", default=str(ROOT / "artifacts/prm_g2/g2_acceptance.json"))
    parser.add_argument("--audit-report", default=str(
        ROOT / "artifacts/prm_g3/w1_pipeline_independence_audit.md"
    ))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument(
        "--cycle-beta-bound", type=float, default=None,
        help="override the IRLS limit-cycle beta bound for the W1 fit "
             "family; calibrated by a scan run and recorded in the output",
    )
    parser.add_argument("--cycle-robust-bound", type=float, default=None)
    args = parser.parse_args()
    if args.cycle_beta_bound is not None:
        g3mod.IRLS_CYCLE_BETA_REL = float(args.cycle_beta_bound)
    if args.cycle_robust_bound is not None:
        g3mod.IRLS_CYCLE_ROBUST = float(args.cycle_robust_bound)
    IRLS_DIAGNOSTICS["limit_cycle_accepts"] = 0
    IRLS_DIAGNOSTICS["max_cycle_beta_rel"] = 0.0
    IRLS_DIAGNOSTICS["max_cycle_robust"] = 0.0
    if args.bootstrap_draws != 2000 or args.bootstrap_seed != 20260810:
        raise ValueError("W1 requires the frozen bootstrap count and seed")

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise FileExistsError(f"W1 output directory already exists: {out_dir}")
    audit_path = Path(args.audit_report)
    if not audit_path.is_file():
        raise FileNotFoundError(
            "the pipeline-independence audit must be committed before W1 runs"
        )

    protocol_dir = Path(args.protocol_dir)
    rows = list(csv.DictReader(open(args.samples, newline="")))
    canonical = [
        row for row in rows
        if str(row.get("canonical_retained", "")).lower() == "true"
    ]
    if len(canonical) != EXPECTED_N:
        raise ValueError("protocol canonical membership changed")
    natoms_by_index = {int(r["sample_index"]): int(r["natoms"]) for r in canonical}
    host_by_index = {int(r["sample_index"]): str(r["host"]) for r in canonical}
    target_by_index = {int(r["sample_index"]): float(r["target_eV"]) for r in canonical}

    # ── additive SchNet: campaign runs, seed-averaged ────────────────
    schnet_root = Path(args.schnet_root)
    add_dirs = {
        fold: [
            (
                schnet_root / f"host_cv5_f{fold}" / f"seed{seed}" / "test_predictions.npz",
                json.loads(
                    (schnet_root / f"host_cv5_f{fold}" / f"seed{seed}" / "run_manifest.json").read_text()
                ).get("output_sha256", {}).get("test_predictions"),
            )
            for seed in SCHNET_SEEDS
        ]
        for fold in FOLDS
    }
    add_pred, fold_by_index = seed_averaged_arm(
        add_dirs, protocol_dir, "host", target_by_index,
    )

    # ── mean-readout SchNet: hash-verified sensitivity bundle ────────
    bundle_dir = Path(args.readout_bundle)
    bundle_manifest = json.loads((bundle_dir / "manifest.json").read_text())
    if bundle_manifest.get("data_sha256") != EXPECTED_DATA_SHA256:
        raise ValueError("readout bundle is not bound to the frozen source data")
    bundle_sha = bundle_manifest.get("output_sha256", {}).get("predictions")
    if bundle_sha and file_sha256(bundle_dir / "predictions.npz") != bundle_sha:
        raise ValueError("readout bundle predictions hash mismatch")
    with np.load(bundle_dir / "predictions.npz", allow_pickle=False) as archive:
        b_indices = np.asarray(archive["indices"], dtype=np.int64)
        b_targets = np.asarray(archive["targets"], dtype=float)
        b_add = np.asarray(archive["add_predictions"], dtype=float)
        b_mean = np.asarray(archive["mean_predictions"], dtype=float)
    if sorted(b_indices.tolist()) != sorted(natoms_by_index):
        raise ValueError("readout bundle does not cover the canonical population")
    expected_targets = np.asarray([target_by_index[int(i)] for i in b_indices])
    if not np.allclose(b_targets, expected_targets, rtol=0.0, atol=1e-5):
        raise ValueError("readout bundle targets differ from the protocol")
    mean_pred = {int(i): float(v) for i, v in zip(b_indices, b_mean)}

    # ── repaired DART: accepted G2 host-CV OOF, seed-averaged ────────
    acceptance = json.loads(Path(args.g2_acceptance).read_text())
    if (
        acceptance.get("schema_version") != "prm_g2_acceptance_v1"
        or acceptance.get("status") != "accepted"
    ):
        raise ValueError("G2 acceptance is not an accepted prm_g2_acceptance_v1")
    dart_dirs = {}
    for fold in FOLDS:
        entries = acceptance["prediction_sources"]["host_cv"][str(fold)]
        seeds = sorted(int(e["seed"]) for e in entries)
        if seeds != sorted(DART_SEEDS):
            raise ValueError(f"G2 host fold {fold} seed law violated")
        dart_dirs[fold] = [
            (resolve_alias(e["prediction_path"]), str(e["prediction_sha256"]))
            for e in entries
        ]
    dart_pred, dart_fold = seed_averaged_arm(
        dart_dirs, protocol_dir, "host", target_by_index,
    )
    if dart_fold != fold_by_index:
        raise ValueError("DART and SchNet host-fold assignments differ")

    # ── aligned arrays ───────────────────────────────────────────────
    order = np.asarray(sorted(natoms_by_index), dtype=np.int64)
    natoms = np.asarray([natoms_by_index[int(i)] for i in order], dtype=float)
    fold = np.asarray([fold_by_index[int(i)] for i in order], dtype=int)
    hosts = [host_by_index[int(i)] for i in order]
    targets = np.asarray([target_by_index[int(i)] for i in order], dtype=float)
    abs_target = np.abs(targets)
    residuals = {
        "schnet_add": np.asarray([add_pred[int(i)] for i in order]) - targets,
        "schnet_mean": np.asarray([mean_pred[int(i)] for i in order]) - targets,
        "dart_repaired": np.asarray([dart_pred[int(i)] for i in order]) - targets,
    }

    unique_hosts = sorted(set(hosts))
    host_lookup = {host: k for k, host in enumerate(unique_hosts)}
    row_host = np.asarray([host_lookup[host] for host in hosts], dtype=int)

    point = {arm: slope_fit(res, natoms, fold) for arm, res in residuals.items()}
    point_abs = {
        arm: slope_fit(np.abs(res), natoms, fold) for arm, res in residuals.items()
    }
    point_adjusted = {
        arm: slope_fit(res, natoms, fold, extra=abs_target)
        for arm, res in residuals.items()
    }

    rng = np.random.default_rng(args.bootstrap_seed)
    draws: dict[str, list[float]] = {arm: [] for arm in residuals}
    diff_draws: list[float] = []
    for _ in range(args.bootstrap_draws):
        selected = rng.integers(0, len(unique_hosts), size=len(unique_hosts))
        weights = np.bincount(
            selected, minlength=len(unique_hosts),
        ).astype(float)[row_host]
        slopes = {
            arm: slope_fit(res, natoms, fold, weights=weights)
            for arm, res in residuals.items()
        }
        for arm, value in slopes.items():
            draws[arm].append(value)
        diff_draws.append(slopes["schnet_add"] - slopes["dart_repaired"])

    fold_signs = {}
    for arm, res in residuals.items():
        signs = []
        for level in FOLDS:
            mask = fold == level
            signs.append(float(np.sign(slope_fit(
                res[mask], natoms[mask], np.zeros(int(mask.sum()), dtype=int),
            ))))
        fold_signs[arm] = signs

    def ci(values: list[float]) -> tuple[float, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    add_ci = ci(draws["schnet_add"])
    diff_ci = ci(diff_draws)
    add_sign = np.sign(point["schnet_add"])
    add_same_folds = int(sum(
        sign == add_sign and sign != 0 for sign in fold_signs["schnet_add"]
    ))
    established = bool(
        (add_ci[0] > 0 or add_ci[1] < 0)
        and add_same_folds >= 4
        and (diff_ci[0] > 0 or diff_ci[1] < 0)
    )

    out_dir.mkdir(parents=True)
    arm_stats = {}
    for arm in residuals:
        low, high = ci(draws[arm])
        arm_stats[arm] = {
            "signed_slope_eV_per_atom": point[arm],
            "signed_slope_ci_low": low,
            "signed_slope_ci_high": high,
            "signed_slope_bootstrap_p": bootstrap_two_sided_p(draws[arm]),
            "abs_error_slope_eV_per_atom": point_abs[arm],
            "target_adjusted_signed_slope_eV_per_atom": point_adjusted[arm],
            "fold_slope_signs": fold_signs[arm],
        }
    payload = {
        "schema_version": "prm_w1_extensivity_v1",
        "created_at": utc_now(),
        "evidence_tier": "canonical",
        "evidence_label": "CANONICAL — REPAIRED G2 OOF (DART) + AUDITED SCHNET ARCHIVES",
        "preregistration": "paper_Q1/review/w1_extensivity_preregistration.md",
        "pipeline_independence_audit": {
            "path": "artifacts/prm_g3/w1_pipeline_independence_audit.md",
            "sha256": file_sha256(audit_path),
        },
        "protocol": "host_cv",
        "n_rows": int(len(order)),
        "n_hosts": len(unique_hosts),
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "cluster": "host",
        "seed_pooling": "per-sample mean prediction across seeds inside each fold",
        "arms": arm_stats,
        "additive_minus_dart_slope_difference": {
            "point": point["schnet_add"] - point["dart_repaired"],
            "ci_low": diff_ci[0],
            "ci_high": diff_ci[1],
            "bootstrap_p": bootstrap_two_sided_p(diff_draws),
        },
        "gate": {
            "additive_ci_excludes_zero": bool(add_ci[0] > 0 or add_ci[1] < 0),
            "additive_same_direction_folds": add_same_folds,
            "difference_ci_excludes_zero": bool(diff_ci[0] > 0 or diff_ci[1] < 0),
            "extensivity_explanation_established": established,
        },
        "irls": {
            "limit_cycle_accepts": IRLS_DIAGNOSTICS["limit_cycle_accepts"],
            "max_cycle_beta_rel": IRLS_DIAGNOSTICS["max_cycle_beta_rel"],
            "max_cycle_robust": IRLS_DIAGNOSTICS["max_cycle_robust"],
            "cycle_beta_bound": g3mod.IRLS_CYCLE_BETA_REL,
            "cycle_robust_bound": g3mod.IRLS_CYCLE_ROBUST,
        },
        "inputs": {
            "samples_sha256": file_sha256(Path(args.samples)),
            "g2_acceptance_sha256": file_sha256(Path(args.g2_acceptance)),
            "readout_bundle_manifest_sha256": file_sha256(bundle_dir / "manifest.json"),
        },
        "bundle_add_consistency_max_abs_diff_eV": float(np.max(np.abs(
            np.asarray([add_pred[int(i)] for i in b_indices]) - b_add
        ))),
    }
    (out_dir / "w1_extensivity.json").write_text(strict_json(payload))
    with open(out_dir / "w1_rows.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_index", "host", "host_fold", "natoms", "target_eV",
            "residual_schnet_add_eV", "residual_schnet_mean_eV",
            "residual_dart_repaired_eV",
        ])
        for k, index in enumerate(order.tolist()):
            writer.writerow([
                index, hosts[k], int(fold[k]), int(natoms[k]), targets[k],
                residuals["schnet_add"][k], residuals["schnet_mean"][k],
                residuals["dart_repaired"][k],
            ])
    print(strict_json({
        "established": established,
        "slopes": {arm: arm_stats[arm]["signed_slope_eV_per_atom"] for arm in arm_stats},
        "output": str(out_dir),
    }))


if __name__ == "__main__":
    main()
