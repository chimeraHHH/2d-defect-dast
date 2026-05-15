from math import sqrt
import pytest

from ase import Atoms

from gpaw.cluster import Cluster


def test_CO(recwarn):
    R = 2.0
    box = 4.
    h = 0.2
    CO = Cluster(Atoms('CO', [(1, 0, 0), (1, 0, R)]))

    CO.rotate(90, 'y')
    assert CO.positions[1, 0] == pytest.approx(R, abs=1e-10)

    # translate
    CO.translate(-CO.get_center_of_mass())
    p = CO.positions.copy()
    for i in range(2):
        assert p[i, 1] == pytest.approx(0, abs=1e-10)
        assert p[i, 2] == pytest.approx(0, abs=1e-10)

    CO.rotate(p[1] - p[0], (1, 1, 1))
    q = CO.positions.copy()
    for c in range(3):
        assert q[0, c] == pytest.approx(p[0, 0] / sqrt(3), abs=1e-10)
        assert q[1, c] == pytest.approx(p[1, 0] / sqrt(3), abs=1e-10)

    CO.minimal_box(box, h)
    w = recwarn.pop(FutureWarning)

    assert (str(w.message) ==
            'Please use adjust_cell from gpaw.utilities.adjust_cell instead.')

    CO.find_connected(0)
    w = recwarn.pop(FutureWarning)

    assert (str(w.message) ==
            'Please use connected_indices from ase.build.connected instead.')
