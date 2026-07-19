"""Paper-facing metrics with deterministic uncertainty summaries."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr


def regression_metrics(
    targets: Sequence[float], predictions: Sequence[float]
) -> Dict[str, float]:
    y = np.asarray(targets, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    if y.shape != pred.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("targets and predictions must be non-empty aligned vectors")
    error = pred - y
    target_ss = float(np.sum((y - y.mean()) ** 2))
    pearson = float(np.corrcoef(y, pred)[0, 1]) if y.std() > 0 and pred.std() > 0 else float("nan")
    spearman = float(spearmanr(y, pred).statistic) if len(y) > 1 else float("nan")
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "pearson": pearson,
        "spearman": spearman,
        "r2": float(1.0 - np.sum(error**2) / target_ss) if target_ss > 0 else float("nan"),
    }


def macro_group_mae(
    targets: Sequence[float], predictions: Sequence[float], groups: Iterable[str]
) -> Dict[str, float]:
    y = np.asarray(targets, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    labels = np.asarray([str(group) for group in groups])
    if not (len(y) == len(pred) == len(labels)):
        raise ValueError("targets, predictions and groups must align")
    per_group = {
        label: float(np.mean(np.abs(pred[labels == label] - y[labels == label])))
        for label in sorted(set(labels))
    }
    return {
        "macro_mae": float(np.mean(list(per_group.values()))),
        "worst_group_mae": float(max(per_group.values())),
        "n_groups": len(per_group),
        "per_group_mae": per_group,
    }


def low_energy_metrics(
    targets: Sequence[float], predictions: Sequence[float], fraction: float = 0.1
) -> Dict[str, float]:
    y = np.asarray(targets, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    k = max(1, int(round(len(y) * fraction)))
    true_low = set(np.argsort(y)[:k].tolist())
    pred_low = set(np.argsort(pred)[:k].tolist())
    overlap = len(true_low.intersection(pred_low))
    return {
        "fraction": float(fraction),
        "k": int(k),
        "top_k_recall": float(overlap / k),
        "mean_true_energy_in_predicted_top_k_eV": float(np.mean(y[list(pred_low)])),
    }


def paired_bootstrap_delta(
    targets: Sequence[float],
    predictions_a: Sequence[float],
    predictions_b: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Return MAE(A)-MAE(B); negative values favor A."""
    y = np.asarray(targets, dtype=float)
    a = np.abs(np.asarray(predictions_a, dtype=float) - y)
    b = np.abs(np.asarray(predictions_b, dtype=float) - y)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("paired vectors must align")
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, 500):
        stop = min(start + 500, n_bootstrap)
        draw = rng.integers(0, len(y), size=(stop - start, len(y)))
        deltas[start:stop] = np.mean(a[draw] - b[draw], axis=1)
    observed = float(np.mean(a - b))
    return {
        "delta_mae_a_minus_b_eV": observed,
        "ci95_low_eV": float(np.quantile(deltas, 0.025)),
        "ci95_high_eV": float(np.quantile(deltas, 0.975)),
        "p_two_sided": float(
            min(1.0, 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
        ),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }
