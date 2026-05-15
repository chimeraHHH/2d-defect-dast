import pytest
from gpaw import GPAW


def test_atomic_el_pot(gpw_files):
    calc = GPAW(gpw_files['h2_pw'])
    values = calc.get_atomic_electrostatic_potentials()
    ref = -49.48543067
    assert values == pytest.approx([ref, ref], abs=1e-4)
