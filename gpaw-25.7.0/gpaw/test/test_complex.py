from gpaw import GPAW, restart, FD
from ase.build import molecule
import pytest


def test_complex(in_tmp_dir, gpaw_new):
    Eini0 = -17.8037610364
    energy_eps = 0.0005

    calc = GPAW(xc='LDA',
                h=0.21,
                convergence={'eigenstates': 3.5e-5, 'energy': energy_eps},
                mode=FD(force_complex_dtype=True))

    mol = molecule('N2')
    mol.center(vacuum=3.0)
    mol.calc = calc

    Eini = mol.get_potential_energy()
    assert Eini == pytest.approx(
        Eini0, abs=energy_eps * calc.get_number_of_electrons())

    calc.write('N2_complex.gpw', mode='all')

    mol, calc = restart('N2_complex.gpw')

    if gpaw_new:
        calc.dft.converge({'eigenstates': 3.5e-9,
                           'energy': energy_eps})
        assert calc.dft.ibzwfs.dtype == complex
    else:
        assert calc.wfs.dtype == complex
        assert calc.wfs.kpt_u[0].psit_nG.dtype == complex

        convergence = {'eigenstates': 3.5e-9, 'energy': energy_eps}
        mol.calc = calc.new(convergence=convergence)
    E = mol.get_potential_energy()
    assert E == pytest.approx(
        Eini, abs=energy_eps * calc.get_number_of_electrons())
