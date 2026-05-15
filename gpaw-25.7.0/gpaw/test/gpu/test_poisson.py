from gpaw.grid_descriptor import GridDescriptor
from gpaw.poisson import FDPoissonSolver
import pytest
import numpy as np
from gpaw.mpi import size
from gpaw.gpu import cupy_is_fake


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.skipif(size == 8, reason='Fails at the moment for size=8')
def test_poisson():
    import cupy
    phis = []
    for xp in [np, cupy]:
        lat = 8.0
        gd = GridDescriptor((8, 6, 8), (lat, lat, lat),
                            pbc_c=[False, False, False])
        # Use Gaussian as input
        x, y, z = gd.get_grid_point_coordinates()
        x, y, z = xp.asarray(x), xp.asarray(y), xp.asarray(z)
        sigma = 1.5
        mu = lat / 2.0

        rho = gd.zeros(xp=xp)
        rho[:] = xp.exp(
            -((x - mu)**2 + (y - mu)**2 + (z - mu)**2) / (2.0 * sigma))
        charge = gd.integrate(rho)
        rho -= charge * gd.dv

        phi = gd.zeros(xp=xp)

        poisson = FDPoissonSolver(xp=xp)
        poisson.set_grid_descriptor(gd)
        poisson.solve(phi, rho)
        phis.append(phi)
    cupy.allclose(phis[0], phis[1], rtol=1e-10)
