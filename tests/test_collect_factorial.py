import pytest

from scripts.prm_collect_factorial import contract_hash, summarize_factorial


def synthetic_rows():
    rows = []
    for repeat in range(42, 47):
        for mask in range(8):
            bits = tuple((mask >> shift) & 1 for shift in (2, 1, 0))
            enabled = sum(bits)
            row = {
                "variant": f"g{mask:03b}",
                "bits": bits,
                "repeat": repeat,
            }
            for metric in ("mae", "rmse", "bias", "spearman", "r2"):
                if metric in ("mae", "rmse"):
                    row[f"validation_{metric}"] = 2.0 - 0.1 * enabled + 0.001 * repeat
                    row[f"test_{metric}"] = float(mask) + 0.001 * repeat
                else:
                    row[f"validation_{metric}"] = 0.01 * mask
                    row[f"test_{metric}"] = 0.02 * mask
            rows.append(row)
    return rows


def test_selection_uses_validation_even_when_test_ranking_disagrees():
    summary = summarize_factorial(synthetic_rows(), bootstrap_samples=500)
    assert summary["selection"]["selected_variant"] == "g111"
    assert summary["selection"]["locked_test"]["mae"]["mean"] > 7.0
    gated_mae = next(
        effect for effect in summary["effects"]
        if effect["split"] == "validation"
        and effect["metric"] == "mae"
        and effect["term"] == "G"
    )
    assert gated_mae["mean"] < 0.0


def test_contract_hash_ignores_pairing_fields_and_component_flags():
    first = {
        "seed": 1,
        "split_path": "a.json",
        "output_dir": "a",
        "epochs": 10,
        "model_kwargs": {
            "hidden_dim": 32,
            "use_gated_pooling": False,
            "use_env_enrichment": False,
            "use_prenorm_local": False,
        },
    }
    second = {
        **first,
        "seed": 2,
        "split_path": "b.json",
        "output_dir": "b",
        "model_kwargs": {
            **first["model_kwargs"],
            "use_gated_pooling": True,
            "use_env_enrichment": True,
        },
    }
    assert contract_hash(first) == contract_hash(second)


def test_partial_factorial_cannot_select_an_architecture():
    partial = [row for row in synthetic_rows() if row["repeat"] == 42]

    with pytest.raises(ValueError, match="paired repeats 42--46"):
        summarize_factorial(partial, bootstrap_samples=100)
