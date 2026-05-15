import pytest
from ase import Atoms
from gpaw import GPAW, PW


@pytest.mark.ci
def test_pw_smallanglecell(in_tmp_dir):
    a = 3.0
    ec = 200
    h2 = Atoms('H2', [[0, 0, 0], [0, 0, 0.8]],
               cell=[a, a, a], pbc=1)
    h2.calc = GPAW(mode=PW(ec), txt='sc.txt')
    e0 = h2.get_potential_energy()
    h2.cell[1, 0] = a
    with pytest.warns(UserWarning):
        e = h2.get_potential_energy()
    assert abs(e - e0) < 0.001, e - e0
