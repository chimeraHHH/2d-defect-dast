import pytest
from ase import Atoms
from gpaw import GPAW


def test_noncollinear_GGA():
    a = Atoms('H', [[0, 0, 0]], magmoms=[1])
    a.center(vacuum=2.5)
    a.calc = GPAW(mode='pw',
                  xc='PBE',
                  symmetry='off',
                  txt=None,
                  experimental={'magmoms': [[0, 0.5, 0.5]]})
    with pytest.raises(ValueError, match='Only LDA supported'):
        a.get_potential_energy()
