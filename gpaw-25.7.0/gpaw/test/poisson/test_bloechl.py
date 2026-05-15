from math import pi

import numpy as np
import pytest
from ase import Atoms
from scipy.special import erf

from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD
from gpaw.core import PWDesc
from gpaw.new.ase_interface import GPAW
from gpaw.new.pw.bloechl_poisson import BloechlPAWPoissonSolver
from gpaw.new.pw.paw_poisson import (SimplePAWPoissonSolver,
                                     SlowPAWPoissonSolver)
from gpaw.new.pw.poisson import PWPoissonSolver
from gpaw.mpi import world
from gpaw.gpu import cupy as cp
from gpaw.gpu.mpi import CuPyMPI


def g(rc, rgd):
    """Gaussian."""
    return rgd.spline(4 / rc**3 / np.pi**0.5 * np.exp(-(rgd.r_g / rc)**2),
                      l=0)


def c(r, rc1, rc2):
    """Coulomb interaction between 2 gaussians."""
    a1 = 1 / rc1**2
    a2 = 1 / rc2**2
    f = 2 * (pi**5 / (a1 + a2))**0.5 / (a1 * a2)
    f *= 16 / pi / rc1**3 / rc2**3
    if r == 0.0:
        return f
    T = a1 * a2 / (a1 + a2) * r**2
    y = 0.5 * f * erf(T**0.5) * (pi / T)**0.5
    return y


def energy(charges):
    e0 = 0.0
    for q1, r1, p1 in charges:
        for q2, r2, p2 in charges:
            d = abs(p1 - p2)
            e12 = 0.5 * q1 * q2 * c(d, r1, r2) / (4 * np.pi)**2
            # print(q1, q2, rc1, rc2, d, e12)
            e0 += e12
    return e0


def force(charges, a):
    eps = 1e-5
    charges[a + 2, 2] += eps
    ep = energy(charges)
    charges[a + 2, 2] -= 2 * eps
    em = energy(charges)
    charges[a + 2, 2] += eps
    return (em - ep) / (2 * eps)


@pytest.mark.parametrize('xp',
                         [np,
                          pytest.param(cp, marks=pytest.mark.gpu)])
def test_psolve(xp):
    """Unit-test for Blöchl's fast Poisson-solver."""
    comm = CuPyMPI(world)
    rgd = RGD(0.01, 500)
    rc1 = 0.6
    rc2 = 0.7
    d12 = 1.35
    g_ai = [[g(rc1, rgd)], [g(rc2, rgd)]]
    v = 7.5
    gcut = 25.0
    pw = PWDesc(gcut=gcut, cell=[2 * v, 2 * v, 2 * v + d12], comm=comm)
    relpos_ac = np.array([[0.5, 0.5, v / (2 * v + d12)],
                          [0.5, 0.5, (v + d12) / (2 * v + d12)]])
    g_aig = pw.atom_centered_functions(g_ai, positions=relpos_ac, xp=xp)
    nt_g = pw.zeros(xp=xp)
    C_ai = g_aig.empty()
    if 0 in C_ai:
        C_ai[0] = 0.9
    if 1 in C_ai:
        C_ai[1] = 0.7
    C_ai.data *= 1.0 / (4.0 * np.pi)**0.5
    g_aig.add_to(nt_g, C_ai)

    charges = np.array(
        [(0.9, rc1, 0.0),
         (0.7, rc2, d12),
         (-0.9, 0.3, 0.0),
         (-0.7, 0.4, d12)])
    e0 = energy(charges)
    f0 = force(charges, 0)
    f1 = force(charges, 1)
    print(e0, f0, f1)

    ps = PWPoissonSolver(pw)
    spps = SimplePAWPoissonSolver(
        pw, [0.3, 0.4], ps, relpos_ac, g_aig.atomdist, xp=xp)
    Q_aL = spps.ghat_aLg.empty()
    Q_aL.data[:] = 0.0
    for a, C_i in C_ai.items():
        Q_aL[a][0] = -C_i[0]
    nt0_g = nt_g.gather()
    vt10_g = pw.zeros(xp=xp).gather()
    e1, vHt_g, V1_aL = spps.solve(nt0_g, Q_aL, vt10_g)
    F1_av = spps.force_contribution(Q_aL, vHt_g, nt_g)
    comm.sum(F1_av)
    assert e1 == pytest.approx(e0, abs=1e-9)
    print('simple', e1, e1 - e0)
    print(F1_av)
    assert xp.allclose(F1_av, [[0, 0, f0],
                               [0, 0, f1]])

    pps = BloechlPAWPoissonSolver(
        pw, [0.3, 0.4], ps, relpos_ac, g_aig.atomdist, xp=xp)
    vt20_g = pw.zeros(xp=xp).gather()
    e2, vHt_g, V2_aL = pps.solve(nt0_g, Q_aL, vt20_g)
    F2_av = pps.force_contribution(Q_aL, vHt_g, nt_g)
    comm.sum(F2_av)
    assert e2 == pytest.approx(e0, abs=1e-8)
    print('\nfast  ', e2, e2 - e0)
    assert xp.allclose(V2_aL.data[::9], V1_aL.data[::9])
    if comm.rank == 0:
        vt10_g = vt10_g.to_xp(np)
        vt20_g = vt20_g.to_xp(np)
        assert vt20_g.data[:5] == pytest.approx(vt10_g.data[:5], abs=1e-10)
    assert xp.allclose(F1_av, F2_av, atol=3e-6)

    if 0:
        ps = PWPoissonSolver(pw.new(gcut=2 * gcut))
        opps = SlowPAWPoissonSolver(
            pw, [0.3, 0.4], ps, relpos_ac, g_aig.atomdist)
        vt_g = pw.zeros()
        e3, vHt_h, V_aL = opps.solve(nt_g, Q_aL, vt_g)
        print('old   ', e3, e3 - e0)
        print(V_aL.data[::9])
        print(vt_g.data[:5])


def fast_slow(fast):
    atoms = Atoms('H2', [[0, 0, 0], [0.1, 0.2, 0.8]], pbc=True)
    atoms.center(vacuum=3.5)
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 600},
                      poissonsolver={'fast': fast},
                      convergence={'forces': 1e-3},
                      # txt=None,
                      symmetry='off')
    atoms.get_potential_energy()
    f = atoms.get_forces()
    eps = 0.001 / 2
    atoms.positions[1, 2] += eps
    ep = atoms.get_potential_energy()
    atoms.positions[1, 2] -= 2 * eps
    em = atoms.get_potential_energy()
    print(f[1, 2], (em - ep) / (2 * eps))


if __name__ == '__main__':
    # test_psolve()
    import sys
    fast_slow(int(sys.argv[1]))
