"""Negative contract tests for the G1 final acceptance trust boundary."""
from __future__ import annotations

import unittest

import numpy as np
from ase import Atoms

from scripts.prm_g1_collect_acceptance import (
    require_corrected_model_contract,
    require_exact_graph,
    require_exact_record,
    require_prediction_roundtrip,
    require_protocol_prediction_identity,
)
from src.graph import build_graph
from src.models.crystal_v2 import ENV_ZERO_NEIGHBOR_CORRECTED


class TestG1AcceptanceNegativeContracts(unittest.TestCase):
    def test_forged_graph_is_rejected_against_current_builder(self) -> None:
        atoms = Atoms(
            numbers=[14, 8],
            positions=[[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]],
            cell=np.diag([8.0, 8.0, 18.0]),
            pbc=[True, True, False],
        )
        expected = build_graph(atoms, cutoff=5.0)
        forged = {key: np.array(value, copy=True) for key, value in expected.items()}
        forged["edge_dist"][0] += np.float32(0.125)
        with self.assertRaises(ValueError):
            require_exact_graph(forged, expected, "forged graph")

    def test_record_path_or_config_override_is_rejected(self) -> None:
        expected = {"path": "fixed.yaml", "config": {"epochs": 30}}
        forged = {"path": "other.yaml", "config": {"epochs": 30}}
        with self.assertRaises(ValueError):
            require_exact_record(forged, expected, "freeze record")

    def test_wrong_environment_mode_or_model_sha_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_corrected_model_contract(
                {
                    "env_zero_neighbor_mode": "legacy_batch_dependent_v0",
                    "model_source_sha256": "0" * 64,
                },
                "pilot",
            )
        with self.assertRaises(ValueError):
            require_corrected_model_contract(
                {
                    "env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_CORRECTED,
                    "model_source_sha256": "0" * 64,
                },
                "pilot",
            )

    def test_wrong_target_or_canonical_coverage_is_rejected(self) -> None:
        rows = [
            {"sample_index": 0, "target_eV": 1.0},
            {"sample_index": 1, "target_eV": 2.0},
        ]
        with self.assertRaises(ValueError):
            require_protocol_prediction_identity(
                np.asarray([0]), np.asarray([1.0]), rows, [0, 1], "predictions"
            )
        with self.assertRaises(ValueError):
            require_protocol_prediction_identity(
                np.asarray([0, 1]), np.asarray([1.0, 3.0]),
                rows, [0, 1], "predictions",
            )

    def test_checkpoint_prediction_mismatch_is_rejected(self) -> None:
        indices = np.asarray([0, 1], dtype=np.int64)
        targets = np.asarray([1.0, 2.0])
        with self.assertRaises(ValueError):
            require_prediction_roundtrip(
                indices, np.asarray([0.9, 2.1]), targets,
                indices, np.asarray([0.9, 2.2]), targets,
                "checkpoint roundtrip", atol=1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
