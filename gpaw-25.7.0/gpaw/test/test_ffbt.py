import pytest

import numpy as np
from scipy.special import spherical_jn

from gpaw.ffbt import spherical_bessel


@pytest.mark.parametrize('l', range(7))
def test_spherical_bessel(l):
    x_g = np.arange(101, dtype=float)
    jl_g = spherical_bessel(l, x_g)
    assert jl_g == pytest.approx(spherical_jn(l, x_g))
