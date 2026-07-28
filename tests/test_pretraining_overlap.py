import pytest

from scripts.prm_pretraining_overlap import reduced_formula, summarize_overlap


def test_reduced_formula_removes_supercell_multiplicity():
    assert reduced_formula("Mo2S4") == "MoS2"
    assert reduced_formula("C2H2") == "CH"


def test_pretraining_overlap_distinguishes_formula_and_element_coverage():
    imp2d_rows = [
        {"host": "Mo2S4", "dopant": "Fe"},
        {"host": "MoS2", "dopant": "Co"},
        {"host": "WS2", "dopant": "Ni"},
    ]
    pretraining_samples = [
        {
            "metadata": {"host": "MoS2", "source": "jarvis_dft_3d"},
            "numbers": [42, 16, 16],
        },
        {
            "metadata": {"host": "FeO", "source": "jarvis_dft_3d"},
            "numbers": [26, 8],
        },
        {
            "metadata": {"host": "Co2", "source": "jarvis_dft_3d"},
            "numbers": [27, 27],
        },
    ]

    result = summarize_overlap(imp2d_rows, pretraining_samples)

    assert result["imp2d"]["n_hosts"] == 3
    assert result["overlap"]["host_reduced_formula"]["n_imp2d_hosts_present"] == 2
    assert result["overlap"]["host_reduced_formula"]["n_imp2d_reduced_formulas"] == 2
    assert (
        result["overlap"]["host_reduced_formula"][
            "n_imp2d_reduced_formulas_present"
        ]
        == 1
    )
    assert result["overlap"]["host_reduced_formula"]["n_pretraining_records"] == 1
    assert result["overlap"]["host_reduced_formula"]["details"] == [
        {
            "imp2d_host": "Mo2S4",
            "reduced_formula": "MoS2",
            "pretraining_record_count": 1,
        },
        {
            "imp2d_host": "MoS2",
            "reduced_formula": "MoS2",
            "pretraining_record_count": 1,
        },
    ]
    impurity = result["overlap"]["impurity_element"]
    assert impurity["present"] == ["Co", "Fe"]
    assert impurity["absent"] == ["Ni"]
    assert impurity["fraction_imp2d_impurities_present"] == pytest.approx(2 / 3)
    assert "not structural identity" in result["interpretation_boundary"]
