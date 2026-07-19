from __future__ import annotations

import numpy as np
import pytest

from src.prm_metrics import (
    finite_spearman,
    low_energy_metrics,
    macro_group_mae,
    paired_bootstrap_delta,
    regression_metrics,
)


def test_finite_spearman_rejects_undefined_or_nonfinite_statistics():
    with pytest.raises(ValueError, match="constant ranked values"):
        finite_spearman([1.0, 2.0], [3.0, 3.0], context="test correlation")
    with pytest.raises(ValueError, match="non-finite"):
        finite_spearman([1.0, np.nan], [2.0, 3.0], context="test correlation")


def test_regression_metrics_exact_prediction():
    metrics = regression_metrics([0, 1, 2], [0, 1, 2])
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1


def test_macro_group_mae_weights_groups_equally():
    metrics = macro_group_mae([0, 0, 0], [1, 1, 3], ["large", "large", "small"])
    assert metrics["macro_mae"] == 2
    assert metrics["n_groups"] == 2


def test_low_energy_recall():
    metrics = low_energy_metrics(np.arange(10), np.arange(10), fraction=0.2)
    assert metrics["k"] == 2
    assert metrics["top_k_recall"] == 1


def test_paired_bootstrap_detects_better_model():
    targets = np.arange(20, dtype=float)
    better = targets + 0.1
    worse = targets + 1.0
    result = paired_bootstrap_delta(targets, better, worse, n_bootstrap=1000)
    assert result["delta_mae_a_minus_b_eV"] < 0
    assert result["ci95_high_eV"] < 0
