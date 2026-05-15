from gpaw.mpi import broadcast_exception, world
from gpaw.response.bse import parallel_delete
import numpy as np
from gpaw.blacs import BlacsGrid, Redistributor


def test_bse_parallel_delete(in_tmp_dir):
    rank = world.rank
    n = 17
    N = n * world.size
    my_range = range(rank * n, (rank + 1) * n)

    A_NN = np.diag(np.arange(N, dtype=float))
    deleteN = 2 * np.arange(N // 2)
    A_NR = np.delete(A_NN, deleteN, axis=1)
    A_RR = np.delete(A_NR, deleteN, axis=0)
    R, _ = A_RR.shape
    A_nN = np.zeros((n, N), dtype=float)

    for j, J in enumerate(my_range):
        A_nN[j, J] = J

    grid_nN = BlacsGrid(world, world.size, 1)
    desc_nN = grid_nN.new_descriptor(N, N, n, N)
    if world.size == 1:
        grid_nn = BlacsGrid(world, 1, 1)
    else:
        grid_nn = BlacsGrid(world, world.size // 2, 2)
    desc_nn = grid_nn.new_descriptor(N, N, 2, 2)

    serial_desc = grid_nN.new_descriptor(R, R, R, R).as_serial()

    A_nn = desc_nn.zeros(dtype=float)
    Redistributor(world, desc_nN,
                  desc_nn).redistribute(A_nN, A_nn)
    A_rr, desc_rr = parallel_delete(A_nn, deleteN,
                                    grid_desc=desc_nn)

    ArR_RR, _ = parallel_delete(A_nN, deleteN,
                                grid_desc=desc_nN, new_desc=serial_desc)

    Arr_RR = desc_rr.as_serial().zeros(dtype=float)
    desc_rr.collect_on_master(A_rr, Arr_RR)
    with broadcast_exception(world):
        if rank == 0:
            assert np.allclose(Arr_RR, A_RR)
            assert np.allclose(ArR_RR, A_RR)
