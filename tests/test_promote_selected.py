import pytest

from scripts.prm_promote_selected import ROOT, promoted_config, selected_bits


def test_selected_bits_requires_validation_only_selection():
    assert selected_bits(
        {"selection_data": "validation only", "selected_variant": "g101"}
    ) == [True, False, True]
    with pytest.raises(ValueError, match="validation-only"):
        selected_bits({"selection_data": "test", "selected_variant": "g101"})


def test_promoted_config_sets_only_selected_components():
    base = {
        "seed": 1,
        "split_path": "old",
        "output_dir": "old",
        "model_kwargs": {
            "use_gated_pooling": False,
            "use_env_enrichment": False,
            "use_prenorm_local": False,
            "hidden_dim": 32,
        },
    }
    root_split = ROOT / "split.json"
    config = promoted_config(base, [True, False, True], root_split, "new", 9, "hash")
    assert config["seed"] == 9
    assert config["data_sha256"] == "hash"
    assert config["model_kwargs"]["use_gated_pooling"] is True
    assert config["model_kwargs"]["use_env_enrichment"] is False
    assert config["model_kwargs"]["use_prenorm_local"] is True
    assert base["model_kwargs"]["use_gated_pooling"] is False
