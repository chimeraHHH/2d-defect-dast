from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIRECT_ENTRYPOINTS = (
    "prm_collect_descriptor_results.py",
    "prm_collect_factorial.py",
    "prm_collect_results.py",
    "prm_materials_analysis.py",
    "prm_normalize_descriptor_metrics.py",
    "prm_promote_selected.py",
    "prm_scheduler.py",
    "prm_uq_analysis.py",
)


@pytest.mark.parametrize("script_name", DIRECT_ENTRYPOINTS)
def test_prm_script_supports_direct_execution_from_outside_repo(
    script_name: str,
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
