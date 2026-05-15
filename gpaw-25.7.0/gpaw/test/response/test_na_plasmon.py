import pytest
import numpy as np

from ase import Atoms
from gpaw import GPAW, PW
from gpaw.mpi import world
from gpaw.test import findpeak
from gpaw.utilities import compiled_with_sl
from gpaw.response.df import DielectricFunction
from gpaw.response.symmetry import QSymmetryAnalyzer

# Comparing the plasmon peaks found in bulk sodium for two different
# atomic structures. Testing for identical plasmon peaks. Not using
# physical sodium cell.


@pytest.mark.dielectricfunction
@pytest.mark.response
def test_response_na_plasmon(in_tmp_dir):
    a = 4.23 / 2.0
    a1 = Atoms('Na',
               scaled_positions=[[0, 0.1, 0]],
               cell=(a, a, a),
               pbc=True)

    # parallel calculations must have domain = 1
    parallel = {'band': 1}
    if world.size > 1 and compiled_with_sl():
        parallel.update({'domain': 1})

    # Expanding along x-direction
    a2 = Atoms('Na2',
               scaled_positions=[[0, 0.1, 0], [0.5, 0.1, 0]],
               cell=(2 * a, a, a),
               pbc=True)

    a1.calc = GPAW(mode=PW(250),
                   kpts={'size': (4, 4, 4), 'gamma': True},
                   parallel=parallel,
                   # txt='small.txt',
                   )

    # Kpoint sampling should be halved in the expanded direction.
    a2.calc = GPAW(mode=PW(250),
                   kpts={'size': (2, 4, 4), 'gamma': True},
                   parallel=parallel,
                   # txt='large.txt',
                   )

    a1.get_potential_energy()
    a2.get_potential_energy()

    # Use twice as many bands for expanded structure
    a1.calc.diagonalize_full_hamiltonian(nbands=20)
    a2.calc.diagonalize_full_hamiltonian(nbands=40)

    a1.calc.write('gs_Na_small.gpw', 'all')
    a2.calc.write('gs_Na_large.gpw', 'all')

    # Settings that should yield the same result
    settings = [
        {'qsymmetry': QSymmetryAnalyzer(pointgroup, timerev)}
        for pointgroup in [False, True]
        for timerev in [False, True]]

    # Test block parallelization (needs scalapack)
    if world.size > 1 and compiled_with_sl():
        settings.append({'qsymmetry': True, 'nblocks': 2})

    # Calculate the dielectric functions
    dfs0 = []  # Arrays to check for self-consistency
    dfs1 = []
    dfs2 = []
    dfs3 = []
    dfs4 = []
    dfs5 = []

    # list of intensities to compare against. Intensity values matched
    # to 10-3 w/ higher tol. Speeding up test degraded the agreement to 10-2
    # Added additional intensity difference check test with tol 10-3
    I_diffs = {'x': [0.008999, 0.007558, 0.008999, 0.007558, 0.006101],
               'y': [0.004063, 0.004063, 0.004063, 0.004063, 0.004063],
               'z': [0.004063, 0.005689, 0.004244, 0.005689, 0.005689]}
    for idx, kwargs in enumerate(settings):
        df1 = DielectricFunction('gs_Na_small.gpw',
                                 ecut=40,
                                 rate=0.001,
                                 **kwargs)

        df1NLFCx, df1LFCx = df1.get_dielectric_function(direction='x')
        df1NLFCy, df1LFCy = df1.get_dielectric_function(direction='y')
        df1NLFCz, df1LFCz = df1.get_dielectric_function(direction='z')

        df2 = DielectricFunction('gs_Na_large.gpw',
                                 ecut=40,
                                 rate=0.001,
                                 **kwargs)

        df2NLFCx, df2LFCx = df2.get_dielectric_function(direction='x')
        df2NLFCy, df2LFCy = df2.get_dielectric_function(direction='y')
        df2NLFCz, df2LFCz = df2.get_dielectric_function(direction='z')

        dfs0.append(df1NLFCx)
        dfs1.append(df1LFCx)
        dfs2.append(df1NLFCy)
        dfs3.append(df1LFCy)
        dfs4.append(df1NLFCz)
        dfs5.append(df1LFCz)

        # Compare plasmon frequencies and intensities: x, y, z
        # x values
        w_w = df1.chi0calc.wd.omega_w
        w1, I1 = findpeak(w_w, -(1. / df1LFCx).imag)
        w2, I2 = findpeak(w_w, -(1. / df2LFCx).imag)
        I_diff = abs(I1 - I2)
        # test that the frequency for 2 settings are aprx equal
        assert w1 == pytest.approx(w2, abs=1e-2)
        # test that the intensity difference is within some tol
        assert I_diff == pytest.approx(I_diffs['x'][idx], abs=5e-3)
        # test that the intensities are aprx equal
        assert I1 == pytest.approx(I2, abs=1e-2)

        # y values
        w1, I1 = findpeak(w_w, -(1. / df1LFCy).imag)
        w2, I2 = findpeak(w_w, -(1. / df2LFCy).imag)
        I_diff = abs(I1 - I2)
        assert w1 == pytest.approx(w2, abs=1e-2)
        assert I_diff == pytest.approx(I_diffs['y'][idx], abs=5e-3)
        assert I1 == pytest.approx(I2, abs=1e-2)

        # z values
        w1, I1 = findpeak(w_w, -(1. / df1LFCz).imag)
        w2, I2 = findpeak(w_w, -(1. / df2LFCz).imag)
        I_diff = abs(I1 - I2)
        assert w1 == pytest.approx(w2, abs=1e-2)
        assert I_diff == pytest.approx(I_diffs['z'][idx], abs=5e-3)
        assert I1 == pytest.approx(I2, abs=1e-2)

    # Check for self-consistency
    for i, dfs in enumerate([dfs0, dfs1, dfs2, dfs3, dfs4, dfs5]):
        while len(dfs):
            df = dfs.pop()
            for df2 in dfs:
                assert np.max(np.abs((df - df2) / df)) < 2e-3
