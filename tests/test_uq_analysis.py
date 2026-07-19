import numpy as np
import pytest

from scripts.prm_uq_analysis import (
    calibration_subsets,
    conformal_quantile,
    load_aligned_predictions,
    risk_coverage,
)


def test_prediction_members_align_by_sample_index(tmp_path):
    first = tmp_path / "seed1"
    second = tmp_path / "seed2"
    first.mkdir()
    second.mkdir()
    np.savez(
        first / "test_predictions.npz",
        schema_version=np.asarray("prm_predictions_v1"),
        split=np.asarray("test"), split_id=np.asarray("uq_calibration_s62"),
        indices=np.asarray([2, 1]),
        targets=np.asarray([20.0, 10.0], dtype=np.float32),
        preds=np.asarray([19.0, 9.0]),
    )
    np.savez(
        second / "test_predictions.npz",
        schema_version=np.asarray("prm_predictions_v1"),
        split=np.asarray("test"), split_id=np.asarray("uq_calibration_s62"),
        indices=np.asarray([1, 2]),
        targets=np.asarray([10.0, 20.0], dtype=np.float32),
        preds=np.asarray([11.0, 21.0]),
    )
    indices, targets, members = load_aligned_predictions([first, second], "test")
    assert indices.tolist() == [1, 2]
    assert targets.tolist() == [10.0, 20.0]
    assert targets.dtype == np.float32
    assert members.tolist() == [[9.0, 19.0], [11.0, 21.0]]


def test_prediction_loader_rejects_nonfinite_member_values(tmp_path):
    run = tmp_path / "seed1"
    run.mkdir()
    np.savez(
        run / "test_predictions.npz",
        schema_version=np.asarray("prm_predictions_v1"),
        split=np.asarray("test"), split_id=np.asarray("uq_calibration_s62"),
        indices=np.asarray([1, 2]), targets=np.asarray([1.0, 2.0]),
        preds=np.asarray([1.0, np.nan]),
    )
    with pytest.raises(ValueError, match="non-finite prediction"):
        load_aligned_predictions([run], "test")


def test_calibration_halves_are_label_independent_and_disjoint():
    indices = np.asarray([8, 2, 6, 4, 0, 10])
    variance, conformal = calibration_subsets(indices)
    assert set(variance).isdisjoint(conformal)
    assert sorted(np.concatenate([variance, conformal]).tolist()) == list(range(6))
    assert indices[variance].tolist() == [0, 4, 8]


def test_finite_sample_conformal_quantile_uses_higher_rank():
    scores = np.arange(1.0, 11.0)
    assert conformal_quantile(scores, 0.8) == 9.0


def test_conformal_quantile_rejects_empty_or_nonfinite_scores():
    with pytest.raises(ValueError, match="non-empty finite"):
        conformal_quantile(np.asarray([]), 0.8)
    with pytest.raises(ValueError, match="non-empty finite"):
        conformal_quantile(np.asarray([1.0, np.nan]), 0.8)


def test_risk_coverage_recovers_oracle_when_uncertainty_orders_error():
    targets = np.zeros(4)
    predictions = np.asarray([0.1, 0.2, 1.0, 2.0])
    uncertainty = np.asarray([0.1, 0.2, 1.0, 2.0])
    metrics, rows = risk_coverage(targets, predictions, uncertainty)
    assert metrics["excess_aurc_eV"] == 0.0
    assert rows[0]["mae_risk_eV"] == 0.1
    assert rows[0]["oracle_mae_risk_eV"] == 0.1
