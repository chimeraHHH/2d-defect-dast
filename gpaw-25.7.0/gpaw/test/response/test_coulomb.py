import pytest
from gpaw.grid_descriptor import GridDescriptor
from gpaw import PoissonSolver
from gpaw.spline import Spline
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.response.qpd import SingleQPWDescriptor
import numpy as np
from gpaw.response.coulomb_kernels import (get_coulomb_kernel,
                                           get_integrated_kernel)
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.poisson_extravacuum import ExtraVacuumPoissonSolver
from gpaw.poisson_moment import MomentCorrectionPoissonSolver


class ExtraVacuum2DPoisson:
    def __init__(self, poisson):
        self.poisson = poisson

    def set_grid_descriptor(self, gd):
        self.N_c = gd.N_c
        self.gd = GridDescriptor(gd.N_c * np.array([1, 1, 9]),
                                 gd.cell_cv @ np.diag([1, 1, 9]),
                                 pbc_c=gd.pbc_c)
        self.poisson.set_grid_descriptor(self.gd)

    def solve(self, v, n):
        myv = self.gd.zeros()
        myn = self.gd.zeros()
        beg, end = self.N_c[2] * 4, self.N_c[2] * 5 - 1
        myn[:, :, beg:end] = n
        self.poisson.solve(myv, myn)
        v[:] = myv[:, :, beg:end]


class Grid:
    def __init__(self, L, N, q, truncation):
        if q == 0:
            supercell = 1
        else:
            supercell = round(1 / q)
            assert np.allclose(supercell * q, 1.0)

        # First we create a unit cell quantities
        pbc_c = [True, True, True]
        if truncation is None:
            pass
        elif truncation == '0D':
            pbc_c = [False, False, False]
        elif truncation == '2D':
            pbc_c = [True, True, False]
        else:
            raise NotImplementedError
        pbc_c = np.array(pbc_c)
        self.pbc_c = pbc_c
        gd = GridDescriptor((N, N, N), [L, L, L], pbc_c=pbc_c)
        self.periodicgd = GridDescriptor((N, N, N), [L, L, L])
        kd = KPointDescriptor([[q, 0, 0]])
        # Create some charge distribution to the unit cell
        if truncation == '0D':
            cut = L / 9
        else:
            cut = L / 4
        spline = Spline.from_data(l=0, rmax=cut, f_g=np.array([1, 0.5, 0.0]))
        c = LFC(gd, [[spline]] * 2, kd=kd, dtype=complex)

        if truncation == '0D':
            c.set_positions([[0.5 - 0.1, 0.5, 0.5], [0.5 + 0.1, 0.5, 0.5]])
        else:
            c.set_positions([[0.11, 0.0, 0.5], [0.5, 0.39, 0.5]])
        n_R = gd.zeros(dtype=complex)
        c.add(n_R, {0: np.array([-1], dtype=complex),
                    1: np.array([1], dtype=complex)}, 0)

        # Now we replicate to supercell, to calculate using grid based Poisson
        # solvers which cannot handle Bloch-phases at the moment
        supergd = GridDescriptor((N * supercell, N, N),
                                 [L * supercell, L, L],
                                 pbc_c=pbc_c)

        # Avoid going over 79 characters
        EVPS = ExtraVacuumPoissonSolver
        MCPS = MomentCorrectionPoissonSolver
        PS = PoissonSolver
        if truncation is None:
            superpoisson = PS(nn=3)
        elif truncation == '0D':
            mcps = MCPS(PS(nn=3), moment_corrections=9)
            superpoisson = EVPS(gpts=(N * 4, N * 4, N * 4),
                                poissonsolver_large=mcps,
                                coarses=1,
                                poissonsolver_small=PS(nn=3))
        elif truncation == '2D':
            superpoisson = ExtraVacuum2DPoisson(PS(nn=3))
        else:
            raise NotImplementedError(truncation)
        superpoisson.set_grid_descriptor(supergd)

        if q != 0:
            supern_R = supergd.zeros(dtype=complex)
            for i in range(supercell):
                pn_R = n_R * np.exp(2j * np.pi * i / supercell)
                supern_R[(i * N):((i + 1) * N), :, :] = pn_R

        else:
            supergd = gd
            supern_R = n_R

        superv_R = supergd.zeros(dtype=complex)
        superpoisson.solve(superv_R.real, supern_R.real)
        superpoisson.solve(superv_R.imag, supern_R.imag)

        self.Ec = supergd.integrate(supern_R, superv_R) / supercell

        self.gd = gd
        self.n_R = gd.zero_pad(n_R)


@pytest.mark.serial
@pytest.mark.response
@pytest.mark.parametrize('gridparam', [(32, 1e-5), (64, 1e-7), (96, 1e-8)])
@pytest.mark.parametrize('qtrunc', [
    (0, '2D', 20.7454594963, 506.01293778),
    (1 / 3, '2D', 13.174804153, 190.916278817),
    (0, None, 20.228908696, 578.42826785),
    (1 / 3, None, 13.5467930334, 214.823201910),
    (0, '0D', 14.3596834829, 206.74182299)])
def test_coulomb(gridparam, qtrunc):
    N, maxdev = gridparam
    q, truncation, sqrtV0_dev, V0_dev = qtrunc
    L = 10
    print()
    # Slightly lower tolerance for 0D system, because it uses so localized
    # charges to avoid crosstalk
    if truncation == '0D':
        maxdev *= 1e2
    grid = Grid(L, N, q, truncation=truncation)
    # Use maximum ecut
    ecut0 = 0.5 * np.pi**2 / ((L / N)**2)
    qpd = SingleQPWDescriptor.from_q([q, 0, 0], ecut0, grid.periodicgd,
                                     gammacentered=False)
    # XXX It is silly that get_coulomb_kernel uses k-point numbers
    # per dim to determine non-periodic direction. It should just get
    # non_per_dir=2, etc.
    N_c = np.array([2, 2, 1])
    v_G = get_coulomb_kernel(qpd, N_c, truncation=truncation, pbc_c=grid.pbc_c)
    V0, sqrtV0 = get_integrated_kernel(
        qpd=qpd, N_c=N_c, truncation=truncation, pbc_c=grid.pbc_c, N=100)
    assert V0 == pytest.approx(V0_dev, maxdev)
    assert sqrtV0 == pytest.approx(sqrtV0_dev, maxdev)

    n_G = qpd.fft(grid.n_R * grid.periodicgd.plane_wave([-q, 0, 0]))
    Ec2 = n_G.conj() @ (v_G * n_G) / N**6 * L**3
    dev = np.abs(grid.Ec / Ec2 - 1)
    print(f'N={N:4d} q=[{q:.3f},0,0] truncation: {str(truncation):5s}'
          f'rel. log10 deviation: {np.log10(dev):7.4f}'
          f'coulomb:{Ec2:9.5f}')
    assert np.abs(dev) < maxdev
