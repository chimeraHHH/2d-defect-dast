import numpy as np
import pytest
from ase import Atoms

from gpaw.grid_descriptor import GridDescriptor
from gpaw.hybrids.kpts import PWKPoint
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import world
from gpaw.projections import Projections
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.lfc import PWLFC
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb as WSTC
from gpaw.spline import Spline
from gpaw.symmetry import Symmetry
from gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions


class AP:
    my_indices = [0]
    comm = world
    rank_a = [0]


r2 = np.linspace(0, 1, 51)**2


class Setup:
    Delta_iiL = np.zeros((1, 1, 1)) + 0.1
    X_p = np.zeros(1) + 0.3
    ExxC = -10.0
    ghat_l = [Spline.from_data(0, 1.0, 1 - r2 * (1 - 2 * r2))]


@pytest.mark.xfail  # XXX: reason=...
def test_exx_derivs():
    if world.size > 1:
        from unittest import SkipTest
        raise SkipTest

    N = 20
    L = 2.5
    nb = 2
    spos_ac = np.zeros((1, 3)) + 0.25

    gd = GridDescriptor([N, N, N], np.eye(3) * L)
    sym = Symmetry([], gd.cell_cv)
    kd = KPointDescriptor(None)
    kd.set_symmetry(Atoms(pbc=True), sym)
    coulomb = WSTC(gd.cell_cv, kd.N_c)
    pd = PWDescriptor(10, gd, complex, kd)

    data = pd.zeros(nb)
    data[0, 1] = 3.0
    data[1, 2] = -2.5
    psit = PlaneWaveExpansionWaveFunctions(nb, pd, data=data)

    proj = Projections(nb, [1], AP(), world, dtype=complex)

    pt = PWLFC([[Spline.from_data(0, 1.0, 1 - r2 * (1 - 2 * r2))]], pd)
    pt.set_positions(spos_ac)

    f_n = np.array([1.0, 0.5])
    kpt = PWKPoint(psit, proj, f_n, np.array([0.0, 0.0, 0.0]), 1.0)

    # xx = EXX(kd, [Setup()], pt, coulomb, spos_ac)
    xx = (kd, [Setup()], pt, coulomb, spos_ac)

    psit.matrix_elements(pt, out=proj)
    C = 0.79
    VV_aii = {a: np.einsum('n, ni, nj -> ij', f_n, P_ni, P_ni.conj()) * C
              for a, P_ni in proj.items()}

    x = xx.calculate([kpt], [kpt], VV_aii, derivatives=True)
    v = x[3][0][0, 1]

    eps = 0.00001

    data[0, 1] = 3 + eps
    psit.matrix_elements(pt, out=proj)
    VV_aii = {a: np.einsum('n, ni, nj -> ij', f_n, P_ni, P_ni.conj()) * C
              for a, P_ni in proj.items()}
    xp = xx.calculate([kpt], [kpt], VV_aii)

    data[0, 1] = 3 - eps
    psit.matrix_elements(pt, out=proj)
    VV_aii = {a: np.einsum('n, ni, nj -> ij', f_n, P_ni, P_ni.conj()) * C
              for a, P_ni in proj.items()}
    xm = xx.calculate([kpt], [kpt], VV_aii)

    d = (xp[0] + xp[1] - xm[0] - xm[1]) / (2 * eps) * N**6 / L**3 / 2
    assert abs(v - d) < 1e-10, (v, d)
