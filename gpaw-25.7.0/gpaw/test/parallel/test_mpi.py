import pytest
import numpy as np
from gpaw.mpi import world, send, receive, broadcast_array
from gpaw.mpi4pywrapper import MPI4PYWrapper

worlds = [world]

try:
    from mpi4py.MPI import COMM_WORLD
except ImportError:
    pass
else:
    worlds.append(MPI4PYWrapper(COMM_WORLD))


@pytest.mark.parametrize('world', worlds)
@pytest.mark.ci
def test_send_receive_object(world):
    if world.size == 1:
        return
    obj = (42, 'hello')
    if world.rank == 0:
        send(obj, 1, world)
    elif world.rank == 1:
        assert obj == receive(0, world)


@pytest.mark.ci
def test_scalar_reduce():
    assert world.sum_scalar(world.rank + 1) == world.size * \
        (world.size + 1) // 2
    assert np.allclose(world.sum_scalar(world.rank + 1.0),
                       world.size * (world.size + 1.0) / 2)
    assert world.min_scalar(world.rank + 1) == 1
    assert world.max_scalar(world.rank + 1) == world.size
    assert world.min_scalar(world.rank + 1.0) == 1.0
    assert world.max_scalar(world.rank + 1.0) == world.size * 1.0


def test_bcast_array():
    new = world.new_communicator

    if world.size == 2:
        comms = [world]
    elif world.size == 4:
        ranks = np.array([[0, 1], [2, 3]])
        comms = [new(ranks[world.rank // 2]),
                 new(ranks[:, world.rank % 2])]
    elif world.size == 8:
        ranks = [[[0, 1], [0, 2], [0, 4]],
                 [[0, 1], [1, 3], [1, 5]],
                 [[2, 3], [0, 2], [2, 6]],
                 [[2, 3], [1, 3], [3, 7]],
                 [[4, 5], [4, 6], [0, 4]],
                 [[4, 5], [5, 7], [1, 5]],
                 [[6, 7], [4, 6], [2, 6]],
                 [[6, 7], [5, 7], [3, 7]]]
        comms = [new(r) for r in ranks[world.rank]]
    else:
        return

    array = np.zeros(3, int)
    if world.rank == 0:
        array[:] = 42

    out = broadcast_array(array, *comms)
    assert (out == 42).all()
