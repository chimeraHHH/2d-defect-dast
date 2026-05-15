import pytest
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.transformers import Transformer
from gpaw.mpi import world
from gpaw.gpu import cupy as cp, cupy_is_fake


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize('pbc', [True, False])
@pytest.mark.parametrize('nn', [1, 2, 3, 4])
def test_fd_transformers(pbc, nn):
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

    a_gpu = cp.asarray(a)

    # Transformers
    coarsegd = gd.coarsen()
    a_coarse = coarsegd.zeros(dtype=dtype)
    a_coarse_gpu = cp.zeros_like(a_coarse)

    # Restrict
    Transformer(gd, coarsegd, nn, dtype=dtype).apply(a, a_coarse, phases=phase)
    Transformer(gd, coarsegd, nn, dtype=dtype, xp=cp).apply(
        a_gpu, a_coarse_gpu, phases=phase)
    a_coarse_ref = a_coarse_gpu.get()
    assert a_coarse == pytest.approx(a_coarse_ref, abs=1e-14)

    # Interpolate
    Transformer(coarsegd, gd, nn, dtype=dtype).apply(a_coarse, a, phases=phase)
    Transformer(coarsegd, gd, nn, dtype=dtype, xp=cp).apply(
        a_coarse_gpu, a_gpu, phases=phase)
    a_ref = a_gpu.get()
    assert a == pytest.approx(a_ref, abs=1e-14)
