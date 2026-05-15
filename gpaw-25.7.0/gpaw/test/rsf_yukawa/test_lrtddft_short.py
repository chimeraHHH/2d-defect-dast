"""Check TDDFT ionizations with Yukawa potential."""
import pytest
from ase import Atoms
from ase.units import Hartree

import gpaw.cgpaw as cgpaw
from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.eigensolvers import RMMDIIS
from gpaw.lrtddft import LrTDDFT
from gpaw.mpi import world
from gpaw.occupations import FermiDirac


@pytest.mark.hybrids
def test_rsf_yukawa_lrtddft_short(in_tmp_dir):
    libxc_version = getattr(cgpaw, 'libxc_version', '2.x.y')
    if int(libxc_version.split('.')[0]) < 3:
        from unittest import SkipTest
        raise SkipTest

    o_plus = Atoms('Be', positions=[[0, 0, 0]])
    o_plus.set_initial_magnetic_moments([1.0])
    adjust_cell(o_plus, 2.5, h=0.35)

    def get_paw(**kwargs):
        """Return calculator object."""
        c = {'energy': 0.05, 'eigenstates': 0.05, 'density': 0.05}
        return GPAW(mode='fd', convergence=c, eigensolver=RMMDIIS(),
                    nbands=3,
                    xc='PBE',
                    parallel={'domain': world.size}, h=0.35,
                    occupations=FermiDirac(width=0.0, fixmagmom=True),
                    **kwargs)

    calc_plus = get_paw(txt='Be_plus_PBE.log', charge=1)
    o_plus.calc = calc_plus
    o_plus.get_potential_energy()

    calc_plus = calc_plus.new(xc='LCY-PBE:omega=0.83:unocc=True',
                              txt='Be_plus_LCY_PBE_083.log')
    o_plus.calc = calc_plus
    o_plus.get_potential_energy()

    lr = LrTDDFT(calc_plus, txt='LCY_TDDFT_Be.log',
                 restrict={'istart': 0, 'jend': 1})
    assert lr.xc.omega == pytest.approx(0.83, abs=0.0)
    lr.write('LCY_TDDFT_Be.ex.gz')
    e_ion = 9.3
    ip_i = 13.36
    # reading is problematic with EXX on more than one core
    if world.rank == 0:
        lr2 = LrTDDFT.read('LCY_TDDFT_Be.ex.gz')
        lr2.diagonalize()
        assert lr2.xc.omega == pytest.approx(0.83, abs=0.0)
        ion_i = lr2[0].get_energy() * Hartree + e_ion
        assert ion_i == pytest.approx(ip_i, abs=0.3)
