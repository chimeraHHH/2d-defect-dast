import numpy as np
import pytest

from gpaw.fd_operators import Laplace
from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import world


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize('pbc', [True, False])
def test_fd_laplace(pbc):
    if world.size > 4:
        # Grid is so small that domain decomposition cannot exceed 4 domains
        assert world.size % 4 == 0
        group, other = divmod(world.rank, 4)
        ranks = np.arange(4 * group, 4 * (group + 1))
        domain_comm = world.new_communicator(ranks)
    else:
        domain_comm = world

    lat = 8.0
    gd = GridDescriptor((32, 32, 32), (lat, lat, lat),
                        pbc_c=pbc, comm=domain_comm)

    if pbc:
        dtype = complex
        phase = np.ones((3, 2), complex)
    else:
        dtype = float
        phase = None

    # Use Gaussian as input
    x, y, z = gd.get_grid_point_coordinates()
    sigma = 1.5
    mu = lat / 2.0

    a = gd.zeros(dtype=dtype)
    a[:] = np.exp(-((x - mu)**2 + (y - mu)**2 + (z - mu)**2) / (2.0 * sigma))
    # analytic solution
    b_analytic = np.zeros_like(a)
    b_analytic[:] = (((x - mu)**2 + (y - mu)**2 + (z - mu)**2) / sigma**2
                     - 3.0 / sigma) * a

    b = np.zeros_like(a)
    a_gpu = cp.asarray(a)
    b_gpu = cp.zeros_like(a_gpu)

    # Laplace
    Laplace(gd, 1.0, 3, dtype=dtype).apply(a, b, phase_cd=phase)
    Laplace(gd, 1.0, 3, dtype=dtype, xp=cp).apply(a_gpu, b_gpu, phase_cd=phase)
    b_ref = b_gpu.get()

    assert b == pytest.approx(b_ref, abs=1e-12)
    # Neglect boundaries in check to analytic solution
    assert (b_analytic[2:-2, 2:-2, 2:-2]
            == pytest.approx(b_ref[2:-2, 2:-2, 2:-2], abs=1e-2))
