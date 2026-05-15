import numpy as np
import pytest

from gpaw.utilities.folder import Folder


def test_vering_fold():
    x = [1, 3, 5, 7, 9]
    y = [2, 6, 1, 9, 3]

    width = 0.2
    width2 = 0.4
    x1 = 4
    x2 = 8
    for folding in ['Gauss', 'Lorentz']:
        x_c, y_c = Folder(width, folding).fold(x, y)

        x_v, y_v = Folder(width, folding).varing_fold(x, y,
                                                      width2=width2,
                                                      x1=x1, x2=x2)
        assert (x_c == x_v).all()

        i = np.where(x_c < 4)
        assert y_c[i] == pytest.approx(y_v[i], abs=1e-01)

        i2 = np.where(x_c > 8)
        assert max(y_c[i2]) > max(y_v[i2])
