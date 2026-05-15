import pytest
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl
import numpy as np
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.test import findpeak
from gpaw.response.bse import BSE

pytestmark = pytest.mark.skipif(
    world.size < 4 or not compiled_with_sl(),
    reason='world.size < 4 or not compiled_with_sl()')


@pytest.mark.response
def test_response_bse_magnon(in_tmp_dir):
    calc = GPAW(mode='pw',
                xc='PBE',
                nbands='nao',
                occupations=FermiDirac(0.001),
                convergence={'bands': -5},
                kpts={'size': (3, 3, 1), 'gamma': True})

    a = 3.945
    c = 8.0
    layer = Atoms(symbols='ScSe2',
                  cell=[a, a, c, 90, 90, 120],
                  pbc=(1, 1, 0),
                  scaled_positions=[(0, 0, 0),
                                    (2 / 3, 1 / 3, 0.0),
                                    (2 / 3, 1 / 3, 0.0)])
    layer.positions[1, 2] += 1.466
    layer.positions[2, 2] -= 1.466
    layer.center(axis=2)
    layer.set_initial_magnetic_moments([1.0, 0, 0])
    layer.calc = calc
    layer.get_potential_energy()
    calc.write('ScSe2.gpw', mode='all')

    eshift = 4.2
    bse = BSE('ScSe2.gpw',
              ecut=10,
              valence_bands=[22],
              conduction_bands=[23],
              eshift=eshift,
              nbands=15,
              mode='BSE',
              truncation='2D')

    w_w = np.linspace(-2, 2, 4001)
    chi_Gw = bse.get_magnetic_susceptibility(eta=0.1,
                                             write_eig='chi+-_0_',
                                             w_w=w_w)

    w, I = findpeak(w_w, -chi_Gw[0].imag)
    assert np.abs(w + 0.0195) < 0.001
    assert np.abs(I - 4.676) < 0.01

    bse = BSE('ScSe2.gpw',
              ecut=10,
              q_c=[1 / 3, 1 / 3, 0],
              valence_bands=[22],
              conduction_bands=[23],
              eshift=eshift,
              nbands=15,
              mode='BSE',
              truncation='2D')

    w_w = np.linspace(-2, 2, 4001)
    chi_Gw = bse.get_magnetic_susceptibility(eta=0.1,
                                             write_eig='chi+-_1_',
                                             w_w=w_w)

    w, I = findpeak(w_w, -chi_Gw[0].imag)
    assert np.abs(w + 0.0153) < 0.001
    assert np.abs(I - 7.624) < 0.01
