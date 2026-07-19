"""Fail-closed provenance checks shared by PRM schedulers and collectors."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


@dataclass(frozen=True)
class ExpectedConfig:
    path: Path
    config: Dict[str, Any]
    sha256: str


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_expected_configs(paths: Iterable[Path]) -> Dict[str, ExpectedConfig]:
    expected: Dict[str, ExpectedConfig] = {}
    for path in paths:
        resolved = path.resolve()
        config = yaml.safe_load(resolved.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"configuration is not a mapping: {resolved}")
        output_dir = str(config.get("output_dir", ""))
        if not output_dir:
            raise ValueError(f"configuration has no output_dir: {resolved}")
        if output_dir in expected:
            raise ValueError(
                f"duplicate expected output_dir {output_dir!r}: "
                f"{expected[output_dir].path} and {resolved}"
            )
        expected[output_dir] = ExpectedConfig(
            path=resolved,
            config=config,
            sha256=config_sha256(config),
        )
    return expected


def validate_manifest_config(
    manifest: Mapping[str, Any],
    expected_configs: Mapping[str, ExpectedConfig],
    manifest_path: Path,
) -> ExpectedConfig:
    embedded = manifest.get("config")
    if not isinstance(embedded, dict):
        raise ValueError(f"run manifest has no embedded configuration: {manifest_path}")
    output_dir = str(embedded.get("output_dir", ""))
    expected = expected_configs.get(output_dir)
    if expected is None:
        raise ValueError(
            f"run output {output_dir!r} has no controlled configuration: {manifest_path}"
        )
    embedded_hash = config_sha256(embedded)
    recorded_hash = manifest.get("config_sha256")
    if recorded_hash != embedded_hash:
        raise ValueError(f"run manifest config/hash mismatch: {manifest_path}")
    if embedded_hash != expected.sha256:
        raise ValueError(
            f"stale run configuration at {manifest_path}; expected {expected.path}"
        )
    return expected
