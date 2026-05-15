import pytest
from ase import Atoms
from gpaw import GPAW
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.poisson import ChargedReciprocalSpacePoissonSolver as CRSPC


def test_charged_pw_poisson():
    """Calculate Coulomb energy of Gaussian charge and compare to analytic
    result.
    """
    n = 50
    L = n / 40 * 8.0
    charge = 1.5
    gd = GridDescriptor((n, n, n), (L, L, L))
    pd = PWDescriptor(None, gd)
    ps = CRSPC(pd, charge, alpha=1.1)
    a = 0.8
    C = gd.cell_cv.sum(0) / 2
    rho = -np.exp(-1 / (4 * a) * pd.G2_qG[0] +
                  1j * (pd.get_reciprocal_vectors() @ C)) / gd.dv
    v = np.empty_like(rho)
    e = ps._solve(v, rho * charge)
    w = gd.collect(pd.ifft(v), broadcast=True)
    for pot, d in [(w[n // 2, n // 2, 0], L / 2),
                   (w[n // 2, 0, 0], L / 2 * 2**0.5),
                   (w[0, 0, 0], L / 2 * 3**0.5)]:
        assert pot == pytest.approx(-charge / d, abs=0.002)
    assert e == pytest.approx((a / 2 / np.pi)**0.5 * charge**2)


def test_pw_proton():
    """Check that the energy of a proton is 0.0."""
    proton = Atoms('H')
    proton.center(vacuum=2.0)
    proton.calc = GPAW(mode='pw', charge=1)
    e = proton.get_potential_energy()
    e += proton.calc.get_reference_energy()
    assert e == pytest.approx(0.0, abs=0.004)
