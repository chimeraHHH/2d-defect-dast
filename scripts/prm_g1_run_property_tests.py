"""Run the stdlib G1 numerical property suite on the assigned server GPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prm_g1_diagnose_geometry import (  # noqa: E402
    gpu_identity,
    git_snapshot,
    sha256_file,
)
from src.graph import GRAPH_BUILDER_VERSION  # noqa: E402
from src.models.crystal_v2 import (  # noqa: E402
    ENV_ZERO_NEIGHBOR_CORRECTED,
    ENV_ZERO_NEIGHBOR_LEGACY,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    git = git_snapshot()
    gpu = gpu_identity()
    receipt_path = args.output_receipt.expanduser().resolve()
    if receipt_path.exists():
        raise FileExistsError("versioned property-test receipt already exists")
    result = subprocess.run(
        [
            sys.executable, "-m", "unittest", "-v",
            "tests.test_g1_graph_correctness",
            "tests.test_prm_g1_acceptance_contract",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or "skipped" in output.lower():
        raise SystemExit("G1 property suite failed or skipped a CUDA test:\n" + output)
    if "Ran 15 tests" not in output or "OK" not in output:
        raise SystemExit("G1 property suite did not execute the frozen 15 tests")
    payload = {
        "schema_version": "prm_g1_property_tests_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_profile": gpu["host_profile"],
        "gpu": {"uuid": gpu["uuid"], "name": gpu["name"]},
        "git": git,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "graph_builder_source_sha256": sha256_file(ROOT / "src/graph.py"),
        "runner_source": {
            "repository_path": "scripts/prm_g1_run_property_tests.py",
            "sha256": sha256_file(
                ROOT / "scripts/prm_g1_run_property_tests.py"
            ),
        },
        "env_zero_neighbor_contract": {
            "corrected": ENV_ZERO_NEIGHBOR_CORRECTED,
            "legacy_g1a_compatibility": ENV_ZERO_NEIGHBOR_LEGACY,
            "model_source_sha256": sha256_file(
                ROOT / "src/models/crystal_v2.py"
            ),
        },
        "test_sources": {
            "graph_correctness": {
                "repository_path": "tests/test_g1_graph_correctness.py",
                "sha256": sha256_file(
                    ROOT / "tests/test_g1_graph_correctness.py"
                ),
                "tests": 10,
            },
            "acceptance_contract": {
                "repository_path": "tests/test_prm_g1_acceptance_contract.py",
                "sha256": sha256_file(
                    ROOT / "tests/test_prm_g1_acceptance_contract.py"
                ),
                "tests": 5,
            },
        },
        "test_count": 15,
        "return_code": result.returncode,
        "passed": True,
        "stdout_tail": output[-4000:],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"passed": True, "receipt_sha256": sha256_file(receipt_path)}))


if __name__ == "__main__":
    main()
