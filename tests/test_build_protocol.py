import numpy as np

from scripts.prm_build_protocol import (
    audit_defect_identity,
    coordinate_fingerprint,
    invariant_distance_fingerprint,
    merged_duplicate_groups,
)
from src.defect_identity import (
    dopant_candidate_indices,
    permutation_safe_defect_mask,
    unique_defect_index,
)


def test_distance_fingerprint_is_translation_and_permutation_invariant():
    first = {
        "numbers": np.asarray([6, 8, 1]),
        "positions": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        "cell": np.eye(3) * 10.0,
        "dist_matrix": np.asarray(
            [[0.0, 1.0, 2.0], [1.0, 0.0, np.sqrt(5.0)], [2.0, np.sqrt(5.0), 0.0]]
        ),
    }
    order = np.asarray([2, 0, 1])
    second = {
        "numbers": first["numbers"][order],
        "positions": first["positions"][order] + np.asarray([3.0, 1.0, 0.0]),
        "cell": first["cell"],
        "dist_matrix": first["dist_matrix"][np.ix_(order, order)],
    }
    assert coordinate_fingerprint(first) != coordinate_fingerprint(second)
    assert invariant_distance_fingerprint(first) == invariant_distance_fingerprint(second)


def test_duplicate_groups_merge_transitive_fingerprint_matches():
    groups = merged_duplicate_groups(
        [
            ["a", "a", "b", "c"],
            ["x", "y", "y", "z"],
        ]
    )
    assert groups == [[0, 1, 2], [3]]


def test_impurity_identity_requires_one_composition_match():
    unique = {
        "id": 1,
        "numbers": np.asarray([6, 8]),
        "metadata": {"host": "C", "dopant": "O"},
    }
    ambiguous = {
        "id": 2,
        "numbers": np.asarray([6, 8, 8]),
        "metadata": {
            "host": "CO", "dopant": "O", "defecttype": "interstitial",
            "site": "int0",
        },
    }
    assert dopant_candidate_indices(unique).tolist() == [1]
    assert unique_defect_index(unique) == 1
    assert permutation_safe_defect_mask(unique).tolist() == [0, 1]
    assert permutation_safe_defect_mask(ambiguous).tolist() == [0, 0, 0]
    with np.testing.assert_raises_regex(ValueError, "unique impurity identity"):
        unique_defect_index(ambiguous)

    audit = audit_defect_identity([unique, ambiguous])
    assert audit["n_unique"] == 1
    assert audit["n_ambiguous"] == 1
    assert audit["ambiguous_indices"] == [1]
