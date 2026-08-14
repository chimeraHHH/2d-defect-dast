import pytest

from scripts.prm_verify_paper_numbers import canonicalize_json_numbers


def test_canonicalize_json_numbers_removes_platform_float_tails():
    macos = {"uq": [0.32428945577191803, 1.0e-18]}
    linux = {"uq": [0.3242894557719179, 1.0e-18]}

    assert canonicalize_json_numbers(macos) == canonicalize_json_numbers(linux)
    assert canonicalize_json_numbers(macos) == {
        "uq": [0.324289455771918, 1.0e-18]
    }


def test_canonicalize_json_numbers_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="nonfinite"):
        canonicalize_json_numbers({"invalid": float("nan")})
