import numpy as np

import pytest

from gpaw.mpi import world
from gpaw.response.pw_parallelization import Blocks1D


@pytest.mark.response
def test_blocks1d_collect():
    """Test the ability to collect an array distributed over the first
    dimension."""
    dat_i = np.arange(150)
    dat_ij = dat_i.reshape((10, 15))
    dat_ijk = dat_i.reshape((5, 3, 10))

    for array in [dat_i, dat_ij, dat_ijk]:
        blocks = Blocks1D(world, array.shape[0])
        local_array = array[blocks.myslice]

        # Test all-gather
        collected_array = blocks.all_gather(local_array)
        assert np.all(array == collected_array)

        # Test gather
        collected_array = blocks.gather(local_array)
        if world.rank == 0:
            assert np.all(array == collected_array)
        else:
            assert collected_array is None
