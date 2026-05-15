import pytest
import numpy as np
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE, read_bse_eigenvalues
from gpaw.response.df import read_response_function
from gpaw.test import findpeak


@pytest.mark.response
def test_response_bse_silicon(in_tmp_dir, scalapack):
    GS = 1
    nosym = 1
    bse = 1
    check = 1

    if GS:
        a = 5.431  # From PRB 73,045112 (2006)
        atoms = bulk('Si', 'diamond', a=a)
        atoms.positions -= a / 8
        calc = GPAW(mode='pw',
                    kpts={'size': (2, 2, 2), 'gamma': True},
                    occupations=FermiDirac(0.001),
                    nbands=12,
                    convergence={'bands': -4})
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('Si.gpw', 'all')

    if bse:
        eshift = 0.8
        bse = BSE('Si.gpw',
                  ecut=50.,
                  valence_bands=range(4),
                  conduction_bands=range(4, 8),
                  eshift=eshift,
                  nbands=8)
        bse.get_dielectric_function(eta=0.2,
                                    w_w=np.linspace(0, 10, 2001))
        w_w, epsreal_w, epsimag_w = read_response_function('df_bse.csv')
    if check:
        w_ = 2.552
        I_ = 421.15
        w, I = findpeak(w_w, epsimag_w)
        assert w == pytest.approx(w_, abs=0.01)
        assert I == pytest.approx(I_, abs=0.1)

    if GS and nosym:
        atoms = bulk('Si', 'diamond', a=a)
        calc = GPAW(mode='pw',
                    kpts={'size': (2, 2, 2), 'gamma': True},
                    occupations=FermiDirac(0.001),
                    nbands=12,
                    symmetry='off',
                    convergence={'bands': -4})
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('Si.gpw', 'all')

    if bse and nosym:
        bse = BSE('Si.gpw',
                  ecut=50.,
                  valence_bands=range(4),
                  conduction_bands=range(4, 8),
                  eshift=eshift,
                  nbands=8)
        w_w, eps_w = bse.get_dielectric_function(filename=None,
                                                 eta=0.2,
                                                 w_w=np.linspace(0, 10, 2001))

    if check and nosym:
        w, I = findpeak(w_w, eps_w.imag)
        assert w == pytest.approx(w_, abs=0.01)
        assert I == pytest.approx(I_, abs=0.1)

        # Read eigenvalues file and test first 3 weights:
        _, C_w = read_bse_eigenvalues('eig.dat')
        assert C_w[0] == pytest.approx(22.37, abs=0.05)
        # These two have degenerate eigenvalues:
        assert np.sum(C_w[1:3]) == pytest.approx(44.29, abs=0.05)
