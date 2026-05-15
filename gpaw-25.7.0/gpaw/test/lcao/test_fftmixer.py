import pytest
from ase import Atoms

from gpaw import GPAW, LCAO
from gpaw.mixer import FFTMixer


def test_lcao_fftmixer():
    bulk = Atoms('Li', pbc=True,
                 cell=[2.6, 2.6, 2.6])
    k = 4
    bulk.calc = GPAW(mode=LCAO(),
                     kpts=(k, k, k),
                     mixer=FFTMixer())
    e = bulk.get_potential_energy()
    assert e == pytest.approx(-1.710364, abs=1e-4)
