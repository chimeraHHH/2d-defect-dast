from ase.build import bulk
from gpaw import GPAW, PW
import pytest


@pytest.mark.stress
def test_pseudopotential_ah(in_tmp_dir):
    si = bulk('Si', 'diamond', a=5.5, cubic=not True)
    si.calc = GPAW(mode=PW(200),
                   setups='ah',
                   kpts=(2, 2, 2))
    f = si.get_forces()
    assert f == pytest.approx(0.0)
    s = si.get_stress()
    assert s[3:] == pytest.approx(0.0)
    assert s[:3] == pytest.approx(0.299, abs=0.001)
    si.calc.write('Si.gpw', 'all')
    GPAW('Si.gpw')
