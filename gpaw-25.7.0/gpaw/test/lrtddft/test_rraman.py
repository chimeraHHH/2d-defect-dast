import pytest
from ase.build import molecule

from gpaw import GPAW, FermiDirac
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.lrtddft import LrTDDFT


@pytest.mark.lrtddft
def test_lrtddft(in_tmp_dir):
    from ase.vibrations.resonant_raman import ResonantRamanCalculator
    h = 0.25
    H2 = molecule('H2')
    adjust_cell(H2, 3., h=h)
    H2.calc = GPAW(mode='fd', h=h, occupations=FermiDirac(width=0.2),
                   symmetry='off')

    rr = ResonantRamanCalculator(
        H2, LrTDDFT, exkwargs={'restrict': {'energy_range': 15}})
    rr.run()
