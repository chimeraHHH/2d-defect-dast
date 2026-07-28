import numpy as np

from src.features import _build_feature_table


def test_element_feature_table_is_finite_and_standardized():
    table = _build_feature_table(max_z=100)

    assert table.shape == (101, 9)
    assert np.array_equal(table[0], np.zeros(9, dtype=np.float32))
    assert np.isfinite(table).all()
    assert np.allclose(table[1:].mean(axis=0), 0.0, atol=2e-6)
    assert np.allclose(table[1:].std(axis=0), 1.0, atol=2e-6)
