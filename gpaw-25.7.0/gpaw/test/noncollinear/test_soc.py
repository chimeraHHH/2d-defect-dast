"""Test HOMO and LUMO band-splitting for MoS2.

See:

  https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.155433
"""

import numpy as np
import pytest
from ase.build import mx2
from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.spinorbit import soc_eigenstates
from gpaw.berryphase import polarization_phase
from pathlib import Path


def check(E, hsplit, lsplit):
    print(E)
    h1, h2, l1, l2 = E[24:28]  # HOMO-1, HOMO, LUMO, LUMO+1
    print(h2 - h1)
    print(l2 - l1)
    assert abs(h2 - h1 - hsplit) < 0.01
    assert abs(l2 - l1 - lsplit) < 0.002


def check_pol(phi_c):
    pol_c = (phi_c / (2 * np.pi)) % 1
    assert abs(pol_c[0] - 2 / 3) < 0.01
    assert abs(pol_c[1] - 1 / 3) < 0.01


params = dict(mode={'name': 'pw', 'ecut': 350},
              kpts={'size': (3, 3, 1), 'gamma': True})


@pytest.mark.soc
@pytest.mark.skipif(mpi.size > 2, reason='May not work in parallel')
def test_soc_self_consistent(gpaw_new, in_tmp_dir):
    """Self-consistent SOC."""
    gpw_wfs = Path('mos2.gpw')
    a = mx2('MoS2')
    a.center(vacuum=3, axis=2)

    if gpaw_new:
        kwargs = {**params, 'symmetry': 'off',
                  'magmoms': np.zeros((3, 3)), 'soc': True}
    else:
        kwargs = {**params, 'symmetry': 'off',
                  'experimental': {'magmoms': np.zeros((3, 3)), 'soc': True}}

    a.calc = GPAW(convergence={'bands': 28}, **kwargs)
    a.get_potential_energy()
    eigs = a.calc.get_eigenvalues(kpt=0)
    check(eigs, 0.15, 0.002)

    a.calc.write(gpw_wfs, 'all')
    GPAW(gpw_wfs)

    if mpi.size == 1:
        phases_c = polarization_phase(gpw_wfs, comm=mpi.world)
        phi_c = phases_c['electronic_phase_c']
        check_pol(phi_c)


@pytest.mark.soc
@pytest.mark.skipif(mpi.size > 2,
                    reason='Does not work with more than 2 cores')
def test_non_collinear_plus_soc():
    a = mx2('MoS2')
    a.center(vacuum=3, axis=2)

    a.calc = GPAW(experimental={'magmoms': np.zeros((3, 3)), 'soc': False},
                  convergence={'bands': 28}, symmetry='off',
                  parallel={'domain': 1}, **params)
    a.get_potential_energy()

    bzwfs = soc_eigenstates(a.calc, n2=28)
    eigs = bzwfs.eigenvalues()[8]
    check(eigs, 0.15, 0.007)


@pytest.mark.soc
def test_soc_non_self_consistent():
    """Non self-consistent SOC."""
    a = mx2('MoS2')
    a.center(vacuum=3, axis=2)

    a.calc = GPAW(convergence={'bands': 14}, **params)
    a.get_potential_energy()

    bzwfs = soc_eigenstates(a.calc, n2=14)
    eigs = bzwfs.eigenvalues()[8]
    check(eigs, 0.15, 0.007)
