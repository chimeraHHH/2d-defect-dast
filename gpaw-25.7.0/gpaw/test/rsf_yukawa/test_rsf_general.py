"""Check for tunability of gamma for yukawa potential."""
import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.eigensolvers import RMMDIIS
from gpaw.xc.hybrid import HybridXC
from gpaw.occupations import FermiDirac
from gpaw.test import gen
import gpaw.cgpaw as cgpaw


@pytest.mark.hybrids
def test_rsf_yukawa_rsf_general(in_tmp_dir, add_cwd_to_setup_paths):
    libxc_version = getattr(cgpaw, 'libxc_version', '2.x.y')
    if int(libxc_version.split('.')[0]) < 3:
        from unittest import SkipTest
        raise SkipTest

    for atom in ['Be']:
        gen(atom, xcname='PBE', scalarrel=True, exx=True,
            yukawa_gamma=0.83, gpernode=149)

    h = 0.35
    be = Atoms('Be', positions=[[0, 0, 0]])
    adjust_cell(be, 3.0, h=h)

    c = {'energy': 0.05, 'eigenstates': 0.05, 'density': 0.05}

    IP = 8.79

    xc = HybridXC('LCY-PBE', omega=0.83)
    fname = 'Be_rsf.gpw'

    calc = GPAW(mode='fd', txt='Be.txt', xc=xc, convergence=c,
                eigensolver=RMMDIIS(), h=h,
                occupations=FermiDirac(width=0.0), spinpol=False)
    be.calc = calc
    energy_083 = be.get_potential_energy()
    (eps_homo, eps_lumo) = calc.get_homo_lumo()
    assert eps_homo == pytest.approx(-IP, abs=0.15)
    xc2 = 'LCY-PBE'
    energy_075 = calc.get_xc_difference(HybridXC(xc2))
    assert energy_083 - energy_075 == pytest.approx(21.13, abs=0.2)
    calc.write(fname)
    calc2 = GPAW(fname)
    xc = calc2.hamiltonian.xc
    assert xc.name == 'LCY-PBE', 'wrong name for functional'
    assert xc.omega == 0.83, 'wrong value for RSF omega'
