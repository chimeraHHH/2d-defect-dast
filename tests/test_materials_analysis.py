from scripts.prm_materials_analysis import analyse_preferences


def row(index, host, dopant, defecttype, site, target, prediction):
    return {
        "sample_index": index,
        "host": host,
        "dopant": dopant,
        "defecttype": defecttype,
        "site": site,
        "target_eV": target,
        "prediction_eV": prediction,
    }


def test_preference_and_screening_regret_are_computed_by_host_dopant_pair():
    samples = [
        row(0, "H1", "C", "adsorbate", "ads0", 0.0, 1.0),
        row(1, "H1", "C", "adsorbate", "ads1", 1.0, 0.0),
        row(2, "H1", "C", "interstitial", "int0", 2.0, 3.0),
        row(3, "H1", "C", "interstitial", "int1", 3.0, 2.0),
        row(4, "H2", "O", "adsorbate", "ads0", 2.0, 0.0),
        row(5, "H2", "O", "interstitial", "int0", 0.0, 1.0),
    ]
    pairs, sites = analyse_preferences(samples)
    first = next(item for item in pairs if item["host"] == "H1")
    second = next(item for item in pairs if item["host"] == "H2")
    assert first["preference_correct"] == 1
    assert first["global_site_correct"] == 0
    assert first["global_screening_regret_eV"] == 1.0
    assert second["preference_correct"] == 0
    adsorbate = next(
        item for item in sites
        if item["host"] == "H1" and item["defecttype"] == "adsorbate"
    )
    assert adsorbate["exact_site_correct"] == 0
    assert adsorbate["screening_regret_eV"] == 1.0
