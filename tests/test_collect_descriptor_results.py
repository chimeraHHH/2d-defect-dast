import pytest

from scripts.prm_collect_descriptor_results import (
    archive_descriptor_artifacts,
    validate_descriptor_manifest,
)
from scripts.prm_collect_results import file_sha256


DATA_SHA = "data-sha"
PROTOCOL_SHA = "protocol-sha"


def valid_manifest():
    splits = [f"split_{index}" for index in range(27)]
    return {
        "schema_version": "prm_descriptor_manifest_v2",
        "status": "complete",
        "formal_split_coverage": {
            "complete": 27, "total": 27, "all_complete": True,
        },
        "selection_data": "validation only",
        "data_sha256": DATA_SHA,
        "data_file_sha256": DATA_SHA,
        "protocol_manifest_sha256": PROTOCOL_SHA,
        "git": {"dirty": False, "commit": "abc"},
        "splits": splits,
        "split_artifacts": {split: {} for split in splits},
    }


def test_complete_descriptor_manifest_matches_frozen_protocol():
    validate_descriptor_manifest(
        valid_manifest(), {"data_sha256": DATA_SHA}, PROTOCOL_SHA,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "incomplete"),
        ("selection_data", "test"),
        ("data_sha256", "stale"),
        ("data_file_sha256", "stale"),
        ("protocol_manifest_sha256", "stale"),
    ],
)
def test_invalid_descriptor_manifest_is_rejected(field, value):
    manifest = valid_manifest()
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_descriptor_manifest(
            manifest, {"data_sha256": DATA_SHA}, PROTOCOL_SHA,
        )


def test_dirty_descriptor_batch_is_rejected():
    manifest = valid_manifest()
    manifest["git"]["dirty"] = True
    with pytest.raises(ValueError):
        validate_descriptor_manifest(
            manifest, {"data_sha256": DATA_SHA}, PROTOCOL_SHA,
        )


def test_descriptor_artifacts_are_archived_with_verified_hashes(tmp_path):
    descriptor_root = tmp_path / "source"
    out_dir = tmp_path / "results"
    split_id = "id_cv5_f0"
    split_dir = descriptor_root / split_id
    split_dir.mkdir(parents=True)
    (descriptor_root / "manifest.json").write_text('{"status": "complete"}\n')
    (split_dir / "metrics.json").write_text('{"mae": 0.5}\n')
    (split_dir / "predictions.npz").write_bytes(b"predictions")
    manifest = {
        "splits": [split_id],
        "split_artifacts": {
            split_id: {
                "metrics_sha256": file_sha256(split_dir / "metrics.json"),
                "predictions_sha256": file_sha256(split_dir / "predictions.npz"),
            },
        },
    }
    stale_dir = out_dir / "runs" / "stale"
    stale_dir.mkdir(parents=True)
    (stale_dir / "partial.txt").write_text("stale")

    archive = archive_descriptor_artifacts(descriptor_root, out_dir, manifest)

    archived_dir = out_dir / "runs" / split_id
    assert (archived_dir / "metrics.json").read_bytes() == (
        split_dir / "metrics.json"
    ).read_bytes()
    assert (archived_dir / "predictions.npz").read_bytes() == b"predictions"
    assert not stale_dir.exists()
    assert archive["manifest"]["sha256"] == file_sha256(
        out_dir / "runs" / "manifest.json"
    )
    assert archive["runs"][0]["metrics_sha256"] == manifest[
        "split_artifacts"
    ][split_id]["metrics_sha256"]
