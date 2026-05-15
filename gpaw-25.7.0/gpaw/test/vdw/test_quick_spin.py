import pytest
from ase import Atoms
from gpaw import GPAW, FermiDirac


@pytest.mark.libxc
def test_vdw_quick_spin(in_tmp_dir):
    L = 2.5
    a = Atoms('H', cell=(L, L, L), pbc=True)
    calc = GPAW(mode='fd',
                xc='vdW-DF',
                occupations=FermiDirac(width=0.001),
                txt='H.vdW-DF.txt')
    a.calc = calc
    e1 = a.get_potential_energy()

    calc = GPAW(mode='fd',
                xc='vdW-DF',
                txt='H.vdW-DF.spinpol.txt',
                spinpol=True,
                occupations=FermiDirac(width=0.001, fixmagmom=True))
    a.calc = calc
    e2 = a.get_potential_energy()

    assert abs(calc.get_eigenvalues(spin=0)[0] -
               calc.get_eigenvalues(spin=1)[0]) < 1e-10

    assert abs(e1 - e2) < 2e-6, abs(e1 - e2)
