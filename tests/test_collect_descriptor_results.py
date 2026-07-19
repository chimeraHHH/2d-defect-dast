import pytest

from scripts.prm_collect_descriptor_results import validate_descriptor_manifest


DATA_SHA = "data-sha"
PROTOCOL_SHA = "protocol-sha"


def valid_manifest():
    return {
        "schema_version": "prm_descriptor_manifest_v1",
        "status": "complete",
        "formal_split_coverage": {
            "complete": 27, "total": 27, "all_complete": True,
        },
        "selection_data": "validation only",
        "data_sha256": DATA_SHA,
        "protocol_manifest_sha256": PROTOCOL_SHA,
        "git": {"dirty": False, "commit": "abc"},
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
