"""Contract tests for the preregistered G3 pipeline.

Run these tests on WHUServer-L40S from the same clean commit as the formal G3
command. They use only synthetic arrays and do not inspect scientific results.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.prm_g3_physics_analysis import (
    ADJUSTED_PROFILE_FEATURES,
    CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
    DOPANT_TO_SERIES,
    EXPECTED_ATOM_FEATURE_SHA256,
    HOST_TO_FAMILY,
    LEGACY_ENV_ZERO_NEIGHBOR_MODE,
    LEGACY_ZERO_NEIGHBOR_SAMPLE_INDICES,
    PROFILE_GRID_POINTS,
    alternating_projection,
    assert_joined_rows_equal,
    backtransformed_contrast,
    benjamini_hochberg,
    build_design,
    canonical_json_sha256,
    canonical_main_text_claim_eligible,
    e_module_effective_scalars,
    eligible_adjusted_profile_features,
    fit_huber_fixed_effects,
    fit_identity_novelty,
    graph_samples_from_blob,
    local_structure_descriptors,
    p3_positive_claim_gate,
    prepare_primary_analysis_rows,
    profile_design,
    pushed_ancestor,
    require_canonical_env_zero_neighbor_mode,
    target_decile,
    run_p4_case_selection,
    validate_canonical_initialization_fields,
    validate_canonical_run_model_contract,
    validate_canonical_seed_entries,
    validate_zero_neighbor_oof_batch_activation,
    weighted_quantile,
)


class TaxonomyContractTest(unittest.TestCase):
    def test_frozen_identity_counts(self) -> None:
        self.assertEqual(len(HOST_TO_FAMILY), 44)
        self.assertEqual(len(DOPANT_TO_SERIES), 65)
        self.assertEqual(sum(value == "chalcogenide" for value in HOST_TO_FAMILY.values()), 27)
        self.assertEqual(sum(value == "3d" for value in DOPANT_TO_SERIES.values()), 10)
        self.assertEqual(sum(value == "4d" for value in DOPANT_TO_SERIES.values()), 10)
        self.assertEqual(sum(value == "5d" for value in DOPANT_TO_SERIES.values()), 9)


class StatisticalPrimitiveTest(unittest.TestCase):
    def test_weighted_quantile_respects_multiplicity(self) -> None:
        values = np.asarray([0.0, 1.0, 2.0])
        weights = np.asarray([1.0, 8.0, 1.0])
        self.assertAlmostEqual(weighted_quantile(values, 0.5, weights), 1.0)

    def test_alternating_projection_removes_crossed_group_means(self) -> None:
        matrix = np.asarray([[1.0], [2.0], [4.0], [8.0], [3.0], [6.0]])
        left = np.asarray([0, 0, 1, 1, 2, 2])
        right = np.asarray([0, 1, 0, 1, 0, 1])
        residual = alternating_projection(matrix, [left, right], np.ones(6))[:, 0]
        for codes in (left, right):
            for code in np.unique(codes):
                self.assertAlmostEqual(float(residual[codes == code].mean()), 0.0, places=8)

    def test_huber_fixed_effect_slope_is_stable_to_one_outlier(self) -> None:
        host = np.repeat(np.arange(6), 10)
        impurity = np.tile(np.arange(10), 6)
        x = np.sin((host + 1.0) * (impurity + 2.0))
        y = 1.75 * x + 0.2 * host - 0.1 * impurity
        y[-1] += 50.0
        fit = fit_huber_fixed_effects(y, x[:, None], [host, impurity])
        self.assertAlmostEqual(float(fit["beta"][0]), 1.75, delta=0.08)

    def test_bh_adjustment_is_monotone_in_rank(self) -> None:
        pvalues = [0.001, 0.02, 0.03, 0.9]
        adjusted = benjamini_hochberg(pvalues)
        ordered = [adjusted[index] for index in np.argsort(pvalues)]
        self.assertTrue(all(left <= right for left, right in zip(ordered, ordered[1:])))
        self.assertTrue(all(p <= q for p, q in zip(pvalues, adjusted)))

    def test_profile_backtransform_has_frozen_direction(self) -> None:
        numeric = {"feature": np.asarray([0.0, 1.0, 2.0, 3.0])}
        classes = np.asarray(["adsorbate", "adsorbate", "interstitial", "interstitial"])
        design, _, scaling = build_design(numeric, classes, ["feature"])
        medians = {"adsorbate": {"feature": 0.5}, "interstitial": {"feature": 2.5}}
        low = profile_design(["feature"], scaling, medians, "adsorbate", "feature", 0.0)
        high = profile_design(["feature"], scaling, medians, "adsorbate", "feature", 3.0)
        fit = {
            "beta": np.asarray([0.0, 0.4, 0.0]),
            "x_mean": design.mean(axis=0),
            "y_mean": np.log(0.55),
        }
        self.assertGreater(backtransformed_contrast(fit, low, high)[2], 0.0)


class DescriptorSemanticsTest(unittest.TestCase):
    def test_legacy_zero_neighbor_population_is_frozen_before_outcome_analysis(self) -> None:
        self.assertEqual(len(LEGACY_ZERO_NEIGHBOR_SAMPLE_INDICES), 14)
        self.assertEqual(len(set(LEGACY_ZERO_NEIGHBOR_SAMPLE_INDICES)), 14)
        self.assertEqual(tuple(sorted(LEGACY_ZERO_NEIGHBOR_SAMPLE_INDICES)),
                         LEGACY_ZERO_NEIGHBOR_SAMPLE_INDICES)

    def test_model_edge_features_retain_periodic_impurity_self_images(self) -> None:
        from ase import Atoms

        atoms = Atoms(
            symbols=["Li", "H"],
            positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 0.0]],
            cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]],
            pbc=[True, True, False],
        )
        from src.graph import build_graph

        descriptor = local_structure_descriptors(atoms, "Li", build_graph(atoms))
        self.assertGreater(
            descriptor["model_edge_periodic_impurity_self_image_count_5A"], 0.0
        )
        self.assertEqual(
            descriptor["cn_5A"],
            descriptor["host_only_cn_5A"]
            + descriptor["model_edge_periodic_impurity_self_image_count_5A"],
        )

    def test_zero_neighbor_numeric_branch_matches_model_modes(self) -> None:
        encoded_en = np.asarray([0.0, 0.25, 0.75])
        legacy = e_module_effective_scalars(
            np.asarray([]), np.asarray([], dtype=int), 1, encoded_en,
            LEGACY_ENV_ZERO_NEIGHBOR_MODE,
        )
        canonical = e_module_effective_scalars(
            np.asarray([]), np.asarray([], dtype=int), 1, encoded_en,
            CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        )
        self.assertEqual(
            [legacy[key] for key in (
                "cn_5A", "e_module_effective_mean_distance_A",
                "e_module_effective_max_distance_A",
                "e_module_effective_abs_electronegativity_contrast",
            )],
            [0.0, 0.0, 0.0, 0.25],
        )
        self.assertEqual(
            canonical["e_module_effective_abs_electronegativity_contrast"], 0.0,
        )

    def test_actual_module_constants_and_zero_neighbor_residual_modes(self) -> None:
        import torch
        from src.models.crystal_v2 import (
            ENV_ZERO_NEIGHBOR_CORRECTED,
            ENV_ZERO_NEIGHBOR_LEGACY,
            LocalEnvEnrichment,
        )

        self.assertEqual(ENV_ZERO_NEIGHBOR_LEGACY, LEGACY_ENV_ZERO_NEIGHBOR_MODE)
        self.assertEqual(
            ENV_ZERO_NEIGHBOR_CORRECTED, CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        )

        legacy = LocalEnvEnrichment(
            hidden_dim=8, zero_neighbor_mode=ENV_ZERO_NEIGHBOR_LEGACY,
        )
        corrected = LocalEnvEnrichment(
            hidden_dim=8, zero_neighbor_mode=ENV_ZERO_NEIGHBOR_CORRECTED,
        )
        for module in (legacy, corrected):
            with torch.no_grad():
                for parameter in module.parameters():
                    parameter.zero_()
                module.proj[-1].bias.fill_(1.0)
        captured: list[torch.Tensor] = []
        handle = legacy.proj[0].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        inputs = {
            "h": torch.zeros(2, 1, 8),
            "defect_mask": torch.ones(2, 1, dtype=torch.long),
            # Sample 0 has no outgoing edge; sample 1 has one.  This prevents
            # the historical batch-level early return while leaving sample 0
            # on the exact empty-neighbour branch.
            "edge_index_flat": torch.tensor([[1], [1]], dtype=torch.long),
            "edge_dist_flat": torch.tensor([2.0]),
            "flat_defect_mask": torch.ones(2, dtype=torch.long),
            "flat_en": torch.tensor([0.25, 0.75]),
            "flat_indices": torch.tensor([0, 1], dtype=torch.long),
            "num_atoms_list": [1, 1],
        }
        try:
            with torch.no_grad():
                legacy_output = legacy(**inputs)
                corrected_output = corrected(**inputs)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 1)
        np.testing.assert_allclose(
            captured[0][0, 0].numpy(), [0.0, 0.0, 0.0, 0.25], atol=1e-7,
        )
        np.testing.assert_allclose(legacy_output[:, 0].numpy(), 1.0, atol=1e-7)
        np.testing.assert_allclose(corrected_output[0, 0].numpy(), 0.0, atol=1e-7)
        np.testing.assert_allclose(corrected_output[1, 0].numpy(), 1.0, atol=1e-7)

    def test_isolated_impurity_keeps_physical_distances_undefined(self) -> None:
        from ase import Atoms
        from src.graph import build_graph

        atoms = Atoms(
            symbols=["Li", "H"], positions=[[0, 0, 0], [10, 0, 0]],
            cell=[30, 30, 30], pbc=False,
        )
        descriptor = local_structure_descriptors(
            atoms, "Li", build_graph(atoms), LEGACY_ENV_ZERO_NEIGHBOR_MODE,
        )
        self.assertEqual(descriptor["model_edge_zero_defect_neighbor_5A"], 1.0)
        self.assertEqual(descriptor["e_module_effective_mean_distance_A"], 0.0)
        self.assertTrue(np.isnan(descriptor["neighbor_distance_mean_A"]))
        self.assertTrue(np.isnan(descriptor["neighbor_distance_max_A"]))
        self.assertTrue(np.isnan(descriptor["postrelaxation_min_clearance_A"]))

    def test_zero_neighbor_oof_batch_activation_is_fail_closed(self) -> None:
        test_orders = {
            regime: {0: [0, 1, 2], 1: [], 2: [], 3: [], 4: []}
            for regime in ("pair", "host", "dopant")
        }
        audit = validate_zero_neighbor_oof_batch_activation(
            {0: 0, 1: 4, 2: 0}, test_orders,
        )
        self.assertEqual(set(audit[0]), {"pair", "host", "dopant"})
        self.assertGreater(audit[2]["pair"]["total_defect_center_edges"], 0)
        isolated_orders = {
            regime: {0: [0], 1: [], 2: [], 3: [], 4: []}
            for regime in ("pair", "host", "dopant")
        }
        with self.assertRaisesRegex(RuntimeError, "early-return"):
            validate_zero_neighbor_oof_batch_activation({0: 0}, isolated_orders)

        # Index 64 would form an inactive singleton if the code silently
        # sorted by sample index.  The archived order puts it in the first,
        # active 64-row batch and must therefore be preserved exactly.
        counts = {index: 1 for index in range(65)}
        counts[64] = 0
        archived = [64, *range(63)]
        archived.append(63)
        order_sensitive = {
            regime: {0: archived, 1: [], 2: [], 3: [], 4: []}
            for regime in ("pair", "host", "dopant")
        }
        order_audit = validate_zero_neighbor_oof_batch_activation(
            counts, order_sensitive,
        )
        self.assertEqual(order_audit[64]["pair"]["batch_start"], 0)

    def test_zero_neighbor_physical_imputation_is_flagged_and_retained(self) -> None:
        rows = [
            {
                "evidence_tier": "exploratory", "evidence_label": "test",
                "sample_index": 0, "raw_row_id": 1, "defecttype": "adsorbate",
                "model_edge_zero_defect_neighbor_5A": 0,
                "postrelaxation_min_clearance_A": 0.4,
                "pair_absolute_error_eV": 0.1,
            },
            {
                "evidence_tier": "exploratory", "evidence_label": "test",
                "sample_index": 1, "raw_row_id": 2, "defecttype": "adsorbate",
                "model_edge_zero_defect_neighbor_5A": 1,
                "postrelaxation_min_clearance_A": float("nan"),
                "pair_absolute_error_eV": 0.2,
            },
            {
                "evidence_tier": "exploratory", "evidence_label": "test",
                "sample_index": 2, "raw_row_id": 3, "defecttype": "interstitial",
                "model_edge_zero_defect_neighbor_5A": 0,
                "postrelaxation_min_clearance_A": 0.8,
                "pair_absolute_error_eV": 0.3,
            },
        ]
        complete, ledger, medians = prepare_primary_analysis_rows(
            rows, ["postrelaxation_min_clearance_A"],
        )
        self.assertEqual(len(complete), 3)
        self.assertEqual(complete[1]["postrelaxation_min_clearance_A"], 0.4)
        self.assertEqual(
            ledger[1]["zero_neighbor_imputed_features"],
            "postrelaxation_min_clearance_A",
        )
        self.assertEqual(medians["adsorbate"]["postrelaxation_min_clearance_A"], 0.4)

        invalid = [{**rows[1], "abs_log_extension_factor": float("nan")}]
        with self.assertRaisesRegex(RuntimeError, "remain excluded"):
            prepare_primary_analysis_rows(invalid, ["abs_log_extension_factor"])


class CanonicalTrustContractTest(unittest.TestCase):
    def test_list_graph_container_scans_every_repaired_row(self) -> None:
        version = "exact_mic_invariant_triplets_v1"
        samples = [{"graph_builder_version": version} for _ in range(10_641)]
        self.assertIs(graph_samples_from_blob(samples, version), samples)
        samples[-1] = {"graph_builder_version": "legacy"}
        with self.assertRaisesRegex(ValueError, "not all 10,641"):
            graph_samples_from_blob(samples, version)

    def test_joined_csv_tamper_fails_source_reconstruction(self) -> None:
        expected = [
            {"sample_index": index, "target_eV": float(index), "host": f"h{index % 3}"}
            for index in range(10_224)
        ]
        observed = [
            {key: str(value) for key, value in row.items()} for row in expected
        ]
        assert_joined_rows_equal(observed, expected)
        observed[4_321]["target_eV"] = "-999.0"
        with self.assertRaisesRegex(ValueError, "source reconstruction mismatch"):
            assert_joined_rows_equal(observed, expected)

    def test_canonical_seed_ensemble_rejects_missing_extra_and_duplicate(self) -> None:
        validate_canonical_seed_entries("pair", [{"seed": 242}])
        validate_canonical_seed_entries(
            "host", [{"seed": seed} for seed in (242, 243, 244)],
        )
        for regime, entries in (
            ("pair", [{"seed": 242}, {"seed": 243}]),
            ("host", [{"seed": 242}, {"seed": 242}, {"seed": 244}]),
            ("dopant", [{"seed": 242}, {"seed": 243}]),
        ):
            with self.assertRaisesRegex(ValueError, "seed ensemble changed"):
                validate_canonical_seed_entries(regime, entries)

    def test_canonical_arm_and_asset_are_exact(self) -> None:
        asset_sha = "a" * 64
        valid = {
            "initialization_arm": "corrected_pretrain",
            "pretrained_asset": {"sha256": asset_sha},
            "atom_feature_table_sha256": EXPECTED_ATOM_FEATURE_SHA256,
        }
        validate_canonical_initialization_fields(valid, asset_sha)
        for patch in (
            {"initialization_arm": "scratch"},
            {"pretrained_asset": {"sha256": "b" * 64}},
            {"atom_feature_table_sha256": "c" * 64},
        ):
            invalid = {**valid, **patch}
            with self.assertRaisesRegex(ValueError, "initialization lineage"):
                validate_canonical_initialization_fields(invalid, asset_sha)

    def test_canonical_zero_neighbor_mode_is_mandatory(self) -> None:
        self.assertEqual(
            require_canonical_env_zero_neighbor_mode({
                "env_zero_neighbor_mode": "zero_residual_v1",
            }),
            "zero_residual_v1",
        )
        for payload in ({}, {"env_zero_neighbor_mode": "legacy_batch_dependent_v0"}):
            with self.assertRaisesRegex(ValueError, "zero_residual_v1"):
                require_canonical_env_zero_neighbor_mode(payload)

    def test_canonical_run_rejects_embedded_legacy_or_missing_mode(self) -> None:
        source_sha = "a" * 64
        kwargs = {
            "use_env_enrichment": True,
            "env_zero_neighbor_mode": CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        }
        recipe_sha = canonical_json_sha256(kwargs)
        manifest = {
            "env_zero_neighbor_mode": CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
            "model_recipe_scope": "config.model_kwargs",
            "model_recipe_sha256": recipe_sha,
            "model_source_sha256": source_sha,
            "config": {"model": "v2", "model_kwargs": kwargs},
        }
        validate_canonical_run_model_contract(manifest, recipe_sha, source_sha)
        for embedded_mode in (LEGACY_ENV_ZERO_NEIGHBOR_MODE, None):
            invalid_kwargs = dict(kwargs)
            if embedded_mode is None:
                invalid_kwargs.pop("env_zero_neighbor_mode")
            else:
                invalid_kwargs["env_zero_neighbor_mode"] = embedded_mode
            invalid = {
                **manifest,
                "config": {"model": "v2", "model_kwargs": invalid_kwargs},
            }
            with self.assertRaisesRegex(ValueError, "embedded zero-neighbor modes"):
                validate_canonical_run_model_contract(
                    invalid, recipe_sha, source_sha,
                )

    @mock.patch("scripts.prm_g3_physics_analysis.subprocess.run")
    def test_g2_commit_requires_ancestry_and_pushed_remote(self, run: mock.Mock) -> None:
        run.side_effect = [
            SimpleNamespace(returncode=0),
            SimpleNamespace(stdout="  origin/main\n"),
        ]
        self.assertTrue(pushed_ancestor("a" * 40, "b" * 40))
        run.side_effect = [
            SimpleNamespace(returncode=1),
            SimpleNamespace(stdout="  origin/main\n"),
        ]
        self.assertFalse(pushed_ancestor("a" * 40, "b" * 40))
        run.side_effect = [
            SimpleNamespace(returncode=0),
            SimpleNamespace(stdout=""),
        ]
        self.assertFalse(pushed_ancestor("a" * 40, "b" * 40))

    def test_exploratory_evidence_never_becomes_main_text_eligible(self) -> None:
        self.assertFalse(canonical_main_text_claim_eligible(
            "exploratory", True, True, True,
        ))
        self.assertFalse(canonical_main_text_claim_eligible(
            "canonical", False, False, False,
        ))
        self.assertTrue(canonical_main_text_claim_eligible(
            "canonical", False, True, False,
        ))


class NoveltyAndCasePrimitiveTest(unittest.TestCase):
    def test_identity_novelty_uses_unique_training_identities_and_missing_rule(self) -> None:
        train_ids = [f"train-{index}" for index in range(7)]
        train = np.asarray([
            [0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, np.nan],
            [4.0, 2.0], [5.0, 2.5], [6.0, 3.0],
        ])
        test_ids = ["test-a", "test-b"]
        test = np.asarray([[1.5, 0.75], [7.0, np.nan]])
        novelty, audit = fit_identity_novelty(
            train_ids, train, test_ids, test, ["left", "right"],
        )
        self.assertEqual(set(novelty), set(test_ids))
        self.assertTrue(all(np.isfinite(value) and value >= 0 for value in novelty.values()))
        self.assertEqual(audit["knn_k"], 5)
        self.assertLessEqual(audit["pca_components"], 10)

    def test_target_decile_ties_use_right_side(self) -> None:
        cutpoints = np.arange(1.0, 10.0)
        self.assertEqual(target_decile(0.5, cutpoints), 0)
        self.assertEqual(target_decile(1.0, cutpoints), 1)
        self.assertEqual(target_decile(9.0, cutpoints), 9)

    def test_p3_claim_requires_prespecified_positive_direction(self) -> None:
        positive = p3_positive_claim_gate(
            "canonical", [0.01, 0.20], [0.02, 0.30],
            [0.1, 0.2, 0.3, 0.4, -0.1], [0.2, 0.1, 0.3, 0.4, -0.2],
        )
        negative = p3_positive_claim_gate(
            "canonical", [-0.20, -0.01], [-0.30, -0.02],
            [-0.1] * 5, [-0.2] * 5,
        )
        exploratory = p3_positive_claim_gate(
            "exploratory", [0.01, 0.20], [0.02, 0.30], [0.1] * 5, [0.2] * 5,
        )
        self.assertEqual(positive, (1, 4, 4))
        self.assertEqual(negative[0], 0)
        self.assertEqual(exploratory[0], 0)

    @staticmethod
    def _p4_fixture(include_controls: bool = True) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index in range(10_224):
            rows.append({
                "sample_index": index,
                "raw_row_id": index + 1,
                "host": f"default-host-{index % 2}",
                "host_family": "default-family",
                "dopant": f"default-dopant-{index % 2}",
                "impurity_series": "default-series",
                "defecttype": "adsorbate",
                "site": "ads0",
                "natoms": 20,
                "target_eV": 0.0,
                "pair_absolute_error_eV": 1.0,
                "extension_factor": 1.0,
                "abs_conv2": 100.0 if index < 104 else 0.5,
                "cn_5A": 4.0,
                "postrelaxation_min_clearance_A": 0.1,
            })
        for offset, index in enumerate(range(200, 204)):
            rows[index].update({
                "host": f"high-host-{offset}",
                "host_family": f"family-{offset}",
                "dopant": f"high-dopant-{offset}",
                "impurity_series": f"series-{offset}",
                "defecttype": "interstitial",
                "site": "int0",
                "natoms": 30 + offset,
                "pair_absolute_error_eV": 10.0 - offset,
                "abs_conv2": 0.5,
            })
        for offset, index in enumerate(range(300, 304)):
            rows[index].update({
                "host": f"control-host-{offset}",
                "host_family": f"family-{offset}" if include_controls else "no-match",
                "dopant": f"control-dopant-{offset}",
                "impurity_series": f"series-{offset}",
                "natoms": 30 + offset,
                "pair_absolute_error_eV": 0.0,
                "abs_conv2": 0.5,
            })
        return rows

    def test_p4_top103_nonpathological_matching_and_no_fallback(self) -> None:
        pool, selection, summary = run_p4_case_selection(
            self._p4_fixture(), "canonical", "test", True,
        )
        self.assertEqual(summary["n_conv2_pathology"], 103)
        pathology = {
            int(row["sample_index"])
            for row in pool if row["top_1pct_abs_conv2_pathology_rank"] != ""
        }
        self.assertEqual(pathology, set(range(103)))
        high = [row for row in selection if row["role"].startswith("high_error")]
        controls = [row for row in selection if row["role"].startswith("matched")]
        self.assertEqual((len(high), len(controls)), (4, 4))
        self.assertEqual(len({row["host"] for row in high}), 4)
        self.assertEqual(len({row["dopant"] for row in high}), 4)
        pool_by_index = {int(row["sample_index"]): row for row in pool}
        self.assertTrue(all(
            pool_by_index[int(row["sample_index"])]["nonpathological_eligible"] == 1
            for row in controls
        ))
        self.assertTrue(summary["case_panel_complete"])
        self.assertEqual(summary["main_text_gate"], 1)

        _, incomplete_selection, incomplete = run_p4_case_selection(
            self._p4_fixture(include_controls=False), "canonical", "test", True,
        )
        self.assertEqual(
            sum(row["role"].startswith("matched") for row in incomplete_selection), 0,
        )
        self.assertFalse(incomplete["case_panel_complete"])
        self.assertFalse(incomplete["matching_fallback_used"])
        self.assertEqual(incomplete["main_text_gate"], 0)

    def test_adjusted_profile_contract_is_frozen(self) -> None:
        self.assertEqual(PROFILE_GRID_POINTS, 41)
        self.assertEqual(
            ADJUSTED_PROFILE_FEATURES,
            ("cn_5A", "postrelaxation_min_clearance_A"),
        )
        self.assertEqual(
            eligible_adjusted_profile_features(["cn_5A", "some_other_feature"]),
            ("cn_5A",),
        )
        with self.assertRaisesRegex(RuntimeError, "neither frozen panel-a"):
            eligible_adjusted_profile_features(["some_other_feature"])


if __name__ == "__main__":
    unittest.main()
