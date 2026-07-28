from __future__ import annotations

import numpy as np
import pytest

from scripts.prm_generate_schnet_readout_sensitivity import (
    make_mean_readout_config,
)
from scripts.prm_schnet_readout_sensitivity import (
    fixed_config,
    paired_summary,
    read_samples,
    validate_config_pair,
)


def _parent_config() -> dict:
    return {
        "output_dir": "baselines/schnet/host_cv5_f0/seed342",
        "split_path": "artifacts/prm_protocol_v2/splits/host_cv5_f0.json",
        "seed": 342,
        "epochs": 150,
        "model_kwargs": {"hidden_channels": 128, "readout": "add"},
    }


def test_mean_readout_config_changes_only_declared_fields():
    parent = _parent_config()
    mean = make_mean_readout_config(parent)

    assert parent["model_kwargs"]["readout"] == "add"
    assert mean["model_kwargs"]["readout"] == "mean"
    assert mean["output_dir"] == (
        "sensitivity/schnet_readout/mean/host_cv5_f0/seed342"
    )
    assert fixed_config(parent) == fixed_config(mean)
    validate_config_pair(parent, mean)


def test_readout_pair_rejects_uncontrolled_training_change():
    parent = _parent_config()
    mean = make_mean_readout_config(parent)
    mean["epochs"] = 149

    with pytest.raises(ValueError, match="uncontrolled"):
        validate_config_pair(parent, mean)


def test_paired_summary_uses_host_clusters_and_reports_mean_minus_add():
    targets = np.arange(6, dtype=float)
    add = targets + np.asarray([2.0, 2.0, 1.0, 1.0, 0.6, 0.6])
    mean = targets + np.asarray([1.0, 1.0, 0.5, 0.5, 0.2, 0.2])
    hosts = ["A", "A", "B", "B", "C", "C"]
    natoms = [8, 9, 12, 13, 16, 17]

    summary, host_rows = paired_summary(targets, add, mean, hosts, natoms)

    paired = summary["paired_host_cluster_bootstrap"]
    assert paired["mae_difference_mean_minus_add_eV"] < 0.0
    assert paired["ci_high_eV"] < 0.0
    assert paired["n_resampling_units"] == 3
    assert len(host_rows) == 3


def test_read_samples_keeps_only_canonical_rows(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "sample_index,host,natoms,target_eV,canonical_retained\n"
        "1,MoS2,12,-0.5,True\n"
        "2,WS2,15,0.2,False\n"
    )

    samples = read_samples(path)

    assert samples == {
        1: {"host": "MoS2", "natoms": 12, "target_eV": -0.5}
    }
