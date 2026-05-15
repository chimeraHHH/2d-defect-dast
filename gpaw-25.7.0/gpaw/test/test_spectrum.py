from math import exp, pi, sqrt
import numpy as np

from gpaw.gauss import Gauss
import pytest
from gpaw.utilities.folder import Folder, Lorentz, Voigt  # noqa

# Gauss and Lorentz functions


def test_spectrum():
    width = 0.5
    x = 1.5

    assert Gauss(width).get(x) == pytest.approx(
        exp(- x**2 / 2 / width**2) / sqrt(2 * pi) / width, abs=1.e-15)
    assert Gauss(width).fwhm == pytest.approx(
        width * np.sqrt(8 * np.log(2)), abs=1.e-15)
    assert Lorentz(width).get(x) == pytest.approx(
        width / (x**2 + width**2) / pi, abs=1.e-15)
    assert Lorentz(width).fwhm == pytest.approx(width * 2, abs=1.e-15)

    # folder function

    for func in [Gauss, Lorentz, Voigt]:
        folder = Folder(width, func(width).__class__.__name__)

        x = [0, 2]
        y = [[2, 0, 1], [1, 1, 1]]

        xl, yl = folder.fold(x, y, dx=.7)

        # check first value
        yy = np.dot(np.array(y)[:, 0], func(width).get(xl[0] - np.array(x)))
        assert yl[0, 0] == pytest.approx(yy, abs=1.e-15)
