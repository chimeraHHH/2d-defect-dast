# *** An error occurred in MPI_Recv
# *** on communicator MPI COMMUNICATOR 36 DUP FROM 28
# *** MPI_ERR_TRUNCATE: message truncated
# *** MPI_ERRORS_ARE_FATAL (your MPI job will now abort)

# works with 'sl_default': (2, 2, 32)

import pytest
from gpaw.mpi import world
from ase.build import fcc100, add_adsorbate
from gpaw import GPAW, ConvergenceError
from gpaw.utilities import compiled_with_sl

pytestmark = pytest.mark.skipif(world.size != 4,
                                reason='world.size != 4')


def test_parallel_scalapack_mpirecv_crash():
    assert world.size == 4

    slab = fcc100('Cu', size=(2, 2, 4))
    add_adsorbate(slab, 'O', 1.1, 'hollow')
    slab.center(vacuum=3.0, axis=2)

    kwargs = {}
    if compiled_with_sl():
        kwargs['parallel'] = {'domain': (1, 1, 4), 'sl_default': (2, 2, 64)}

    calc = GPAW(mode='lcao',
                kpts=(2, 2, 1),
                txt='-',
                maxiter=1,
                **kwargs)

    slab.calc = calc
    try:
        slab.get_potential_energy()
    except ConvergenceError:
        pass
