"""Make sure we get a warning when mode is not supplied."""
from ase.build import molecule
from gpaw.calculator import DeprecatedParameterWarning
from gpaw import GPAW
import pytest


@pytest.mark.ci
def test_no_mode_supplied(gpaw_new: bool) -> None:
    if gpaw_new:
        with pytest.raises(TypeError):
            GPAW()
        return
    a = 6.0
    hydrogen = molecule('H2', cell=[a, a, a])
    hydrogen.center()
    with pytest.warns(DeprecatedParameterWarning):
        hydrogen.calc = GPAW()
        hydrogen.calc.initialize(hydrogen)
