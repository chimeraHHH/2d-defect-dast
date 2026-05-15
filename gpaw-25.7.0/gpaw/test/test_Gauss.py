import pytest
from gpaw.gauss import test_derivatives as check


def test_Gauss():
    for i in range(3):
        i1, i2 = check((1.0, -3.4, 1.2),
                       (0, 0, 0), (1, 0, 0), 1.4, 3.0, i)
        assert i1 == pytest.approx(i2, abs=4e-10)
        i1, i2 = check((1.0, -3.4, 1.2),
                       (0, 1, 0), (0, 0, 1), 1.4, 3.0, i)
        assert i1 == pytest.approx(i2, abs=2e-10)
        i1, i2 = check((1.0, -3.4, 1.2),
                       (0, 1, 0), (1, 0, 1), 1.4, 3.0, i)
        assert i1 == pytest.approx(i2, abs=4e-11)
        i1, i2 = check((1.0, -3.4, 1.2),
                       (0, 2, 0), (1, 0, 1), 1.4, 3.0, i)
        assert i1 == pytest.approx(i2, abs=6e-10)
