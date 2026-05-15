"""Test GW band-gaps for Si."""

import pytest
from ase.build import bulk

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response.g0w0 import G0W0
from gpaw.response.screened_interaction import GammaIntegrationMode


def generate_si_systems():
    a = 5.43
    si1 = bulk('Si', 'diamond', a=a)
    si2 = si1.copy()
    si2.positions -= a / 8

    return [si1, si2]


def run(gpw_filename, nblocks, integrate_gamma, qpt=False):
    # This tests checks the actual numerical accuracy which is asserted below
    calc = GPAW(gpw_filename)
    e = calc.get_potential_energy()

    integrate_gamma = GammaIntegrationMode(integrate_gamma)
    # The numerical integration default is too slow, so overriding
    integrate_gamma._N = 30

    kwargs = dict(nbands=8, integrate_gamma=integrate_gamma,
                  kpts=[(0, 0, 0), (0.5, 0.5, 0)],  # Gamma, X
                  ecut=40, nblocks=nblocks,
                  frequencies={'type': 'nonlinear',
                               'domega0': 0.1, 'omegamax': None},
                  eta=0.2, relbands=(-1, 2))

    if qpt:
        # This part of the code is testing for separate calculation of qpoints
        # which would help in trivial parallelization of GW
        gw = G0W0(gpw_filename, 'gw_None', **kwargs)
        for q in range(gw.nqpts):
            gw.calculate(qpoints=[q])

    gw = G0W0(gpw_filename, 'gw_None', **kwargs)
    results = gw.calculate()

    G, X = results['eps'][0]
    output = [e, G[0], G[1] - G[0], X[1] - G[0], X[2] - X[1]]
    G, X = results['qp'][0]
    output += [G[0], G[1] - G[0], X[1] - G[0], X[2] - X[1]]
    return output


reference = {'sphere': pytest.approx([-9.253, 5.442, 2.389, 0.403, 0.000,
                                      6.261, 3.570, 1.323, 0.001], abs=0.0035),
             'WS': pytest.approx([-9.253, 5.442, 2.389, 0.403, 0.000,
                                  6.284, 3.551, 1.285, 0.001], abs=0.0035),
             '1BZ': pytest.approx([-9.252, 5.441, 2.389, 0.403, 0.000,
                                   6.337, 3.450, 1.193, 0.002], abs=0.0035),
             'reciprocal': pytest.approx([-9.252, 5.441, 2.389, 0.403, 0.000,
                                          6.110, 3.86, 1.624, 0.002],
                                         abs=0.0035)}

# The systems are not 2D, thus, the reciprocal2D will yield same results as
# reciprocal. This is tested in test_integrate_gamma_modes.
reference['reciprocal2D'] = reference['reciprocal']
reference['1BZ2D'] = reference['1BZ']


@pytest.mark.response
@pytest.mark.slow
@pytest.mark.parametrize('si', [0, 1])
@pytest.mark.parametrize('integrate_gamma', ['sphere', 'WS'])
@pytest.mark.parametrize('symm', ['all', 'no', 'tr', 'pg'])
@pytest.mark.parametrize('nblocks',
                         [x for x in [1, 2, 4, 8] if x <= world.size])
def test_response_gwsi(in_tmp_dir, si, symm, nblocks, integrate_gamma,
                       scalapack, gpw_files):
    filename = gpw_files[f'si_gw_a{si}_{symm}']
    assert run(filename, nblocks, integrate_gamma) ==\
           reference[integrate_gamma]


@pytest.mark.parametrize('integrate_gamma', ['sphere', 'WS', '1BZ',
                                             'reciprocal',
                                             'reciprocal2D',
                                             '1BZ2D'])
@pytest.mark.response
def test_integrate_gamma_modes(in_tmp_dir, integrate_gamma, gpw_files):
    assert run(gpw_files['si_gw_a0_all'], 1, integrate_gamma) == \
           reference[integrate_gamma]


@pytest.mark.response
@pytest.mark.ci
@pytest.mark.parametrize('si', [0, 1])
@pytest.mark.parametrize('symm', ['all'])
def test_small_response_gwsi(in_tmp_dir, si, symm, scalapack,
                             gpw_files):
    filename = gpw_files[f'si_gw_a{si}_{symm}']
    assert run(filename, 1, 'sphere') == reference['sphere']


@pytest.mark.response
@pytest.mark.ci
def test_few_freq_response_gwsi(in_tmp_dir, scalapack,
                                gpw_files):
    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # This test has very few frequencies and tests that the code doesn't crash.
    filename = gpw_files['si_gw_a0_all']
    gw = G0W0(filename, 'gw_0.2',
              nbands=8, integrate_gamma='sphere',
              kpts=[(0, 0, 0), (0.5, 0.5, 0)],  # Gamma, X
              ecut=40, nblocks=nblocks,
              frequencies={'type': 'nonlinear',
                           'domega0': 0.1, 'omegamax': 0.2},
              eta=0.2, relbands=(-1, 2))
    gw.calculate()
