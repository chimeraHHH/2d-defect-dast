import pytest
from ase import Atoms
from gpaw.mpi import world
from gpaw import GPAW, PW
from gpaw.response.df import DielectricFunction
from gpaw.test import findpeak

# Comparing the EELS spectrum of sodium for different block
# parallelizations. Intended to be run with 8 cores.

pytestmark = pytest.mark.skipif(world.size < 4, reason='world.size < 4')


@pytest.mark.dielectricfunction
@pytest.mark.response
@pytest.mark.slow
@pytest.mark.xfail(reason='https://gitlab.com/gpaw/gpaw/-/jobs/5215834173')
def test_response_na_plasmons_tetrahedron(in_tmp_dir, scalapack):
    a = 4.23 / 2.0
    a1 = Atoms('Na',
               scaled_positions=[[0, 0, 0]],
               cell=(a, a, a),
               pbc=True)

    a1.calc = GPAW(mode=PW(250),
                   kpts={'size': (10, 10, 10), 'gamma': True},
                   parallel={'band': 1},
                   txt='small.txt')

    a1.get_potential_energy()
    a1.calc.diagonalize_full_hamiltonian(nbands=20)
    a1.calc.write('gs_Na.gpw', 'all')

    kwargs = {'integrationmode': 'tetrahedron integration',
              'ecut': 40}

    # Calculate the dielectric functions: tetrahedral 1 block
    df1 = DielectricFunction('gs_Na.gpw',
                             rate=0.001,
                             nblocks=1,
                             txt='1block.txt',
                             **kwargs)
    df1NLFCx, df1LFCx = df1.get_dielectric_function(direction='x')

    # tetrahedral 4 blocks
    df2 = DielectricFunction('gs_Na.gpw',
                             rate=0.001,
                             nblocks=4,
                             txt='4block.txt',
                             **kwargs)
    df2NLFCx, df2LFCx = df2.get_dielectric_function(direction='x')

    # tetrahedron integration 4 blocks with large eta
    kwargs.update({'eta': 4.25})
    df3 = DielectricFunction('gs_Na.gpw',
                             rate=0.001,
                             nblocks=4,
                             txt='4block.txt',
                             **kwargs)
    df3NLFCx, df3LFCx = df3.get_dielectric_function(direction='x')

    # point integration 4 blocks with large eta (smearing)
    kwargs.update({'integrationmode': 'point integration', 'eta': 4.25})
    df4 = DielectricFunction('gs_Na.gpw',
                             rate=0.001,
                             nblocks=4,
                             txt='4block.txt',
                             **kwargs)
    df4NLFCx, df4LFCx = df4.get_dielectric_function(direction='x')

    # Compare plasmon frequencies and intensities
    w_w = df1.chi0calc.wd.omega_w

    w1, I1 = findpeak(w_w, -(1. / df1LFCx).imag)
    w2, I2 = findpeak(w_w, -(1. / df2LFCx).imag)
    w3, I3 = findpeak(w_w, -(1. / df3LFCx).imag)
    w4, I4 = findpeak(w_w, -(1. / df4LFCx).imag)

    # omegas don't change
    assert [w3, w4] == pytest.approx([0.283057, 0.300520], abs=1e-2)
    assert w1 == pytest.approx(w2, 1e-2)  # omega: serial vs parallel
    # omega: PI & TI w/ large eta are aprx equal
    assert w4 == pytest.approx(w3, 2e-2, abs=True)
    assert I1 == pytest.approx(I2, 1e-3)  # intensity: serial vs parallel
