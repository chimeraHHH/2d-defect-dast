from pathlib import Path

import numpy as np

from scripts.prm_make_protocol_figure import (
    build_architecture_summary,
    build_summary,
    latex_formula,
)


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "artifacts/prm_protocol_v2"


def test_protocol_figure_summary_matches_canonical_audit():
    summary, retained, (hosts, dopants, matrix) = build_summary(PROTOCOL)

    assert summary["schema_version"] == "prm_protocol_figure_v3"
    assert summary["counts"]["raw_rows"] == 17_364
    assert summary["counts"]["source_filtered_rows"] == 10_641
    assert summary["counts"]["raw_component_exclusions"] == 4
    assert summary["counts"]["ambiguous_identity_exclusions"] == 349
    assert summary["counts"]["duplicate_exclusions"] == 64
    assert summary["counts"]["canonical_rows"] == 10_224
    assert len(retained) == 10_224
    assert len(hosts) == 44
    assert len(dopants) == 65
    assert int(np.sum(matrix)) == 10_224
    assert summary["inputs"]["data_audit"]["path"] == (
        "artifacts/prm_protocol_v2/data_audit.json"
    )


def test_partition_profiles_cover_every_retained_sample():
    summary, _, _ = build_summary(PROTOCOL)

    assert len(summary["partition_profiles"]) == 7
    assert all(
        np.isclose(profile["total"], 10_224)
        for profile in summary["partition_profiles"]
    )
    uq = next(profile for profile in summary["partition_profiles"] if profile["label"] == "UQ holdout")
    assert uq["calibration"] == 511


def test_formula_labels_use_subscripts_only_for_valid_formulas():
    assert latex_formula("MoS2") == "MoS$_{2}$"
    assert latex_formula("C2H2") == "C$_{2}$H$_{2}$"
    assert latex_formula("not-a-formula") == "not-a-formula"


def test_architecture_summary_matches_promoted_g111_config():
    summary = build_architecture_summary()
    architecture = summary["architecture"]

    assert architecture["selected_variant"] == "g111"
    assert architecture["selection_data"] == "validation only"
    assert architecture["atom_input_dimension"] == 137
    assert architecture["hidden_dimension"] == 128
    assert architecture["local_layers"] == 3
    assert architecture["global_layers"] == 2
    assert architecture["attention_heads"] == 4
    assert architecture["local_graph_radius_A"] == 5.0
    assert architecture["radial_bias_grid_endpoint_A"] == 12.0
    assert architecture["radial_bias_grid_is_attention_cutoff"] is False
    assert architecture["modules"] == {"G": True, "E": True, "P": True}
    assert len(architecture["environment_features"]) == 4
