from __future__ import annotations

import pytest

from src.prm_provenance import (
    config_sha256,
    load_expected_configs,
    validate_manifest_config,
)


def _write_config(path, output_dir, epochs=150):
    path.write_text(
        f"output_dir: {output_dir}\n"
        "split_path: artifacts/split.json\n"
        "seed: 142\n"
        f"epochs: {epochs}\n"
    )


def test_manifest_configuration_must_match_controlled_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, "factorial/g000/split42_seed142")
    expected = load_expected_configs([config_path])
    config = expected["factorial/g000/split42_seed142"].config
    manifest = {"config": config, "config_sha256": config_sha256(config)}

    record = validate_manifest_config(manifest, expected, tmp_path / "run_manifest.json")
    assert record.path == config_path.resolve()

    stale = {**config, "epochs": 151}
    stale_manifest = {"config": stale, "config_sha256": config_sha256(stale)}
    with pytest.raises(ValueError, match="stale run configuration"):
        validate_manifest_config(stale_manifest, expected, tmp_path / "run_manifest.json")


def test_manifest_rejects_an_internally_false_hash(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, "factorial/g000/split42_seed142")
    expected = load_expected_configs([config_path])
    config = expected["factorial/g000/split42_seed142"].config
    manifest = {"config": config, "config_sha256": "0" * 64}

    with pytest.raises(ValueError, match="config/hash mismatch"):
        validate_manifest_config(manifest, expected, tmp_path / "run_manifest.json")


def test_expected_configs_require_unique_output_directories(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    _write_config(first, "same/output")
    _write_config(second, "same/output")
    with pytest.raises(ValueError, match="duplicate expected output_dir"):
        load_expected_configs([first, second])
