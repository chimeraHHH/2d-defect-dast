from pathlib import Path

import numpy as np

from scripts.prm_make_protocol_figure import build_summary, latex_formula


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "artifacts/prm_protocol_v1"


def test_protocol_figure_summary_matches_canonical_audit():
    summary, retained, (hosts, dopants, matrix) = build_summary(PROTOCOL)

    assert summary["schema_version"] == "prm_protocol_figure_v1"
    assert summary["counts"]["raw_rows"] == 17_364
    assert summary["counts"]["source_filtered_rows"] == 10_641
    assert summary["counts"]["raw_component_exclusions"] == 4
    assert summary["counts"]["duplicate_exclusions"] == 65
    assert summary["counts"]["canonical_rows"] == 10_572
    assert len(retained) == 10_572
    assert len(hosts) == 44
    assert len(dopants) == 65
    assert int(np.sum(matrix)) == 10_572
    assert summary["inputs"]["data_audit"]["path"] == (
        "artifacts/prm_protocol_v1/data_audit.json"
    )


def test_partition_profiles_cover_every_retained_sample():
    summary, _, _ = build_summary(PROTOCOL)

    assert len(summary["partition_profiles"]) == 7
    assert all(
        np.isclose(profile["total"], 10_572)
        for profile in summary["partition_profiles"]
    )
    uq = next(profile for profile in summary["partition_profiles"] if profile["label"] == "UQ holdout")
    assert uq["calibration"] == 528


def test_formula_labels_use_subscripts_only_for_valid_formulas():
    assert latex_formula("MoS2") == "MoS$_{2}$"
    assert latex_formula("C2H2") == "C$_{2}$H$_{2}$"
    assert latex_formula("not-a-formula") == "not-a-formula"
