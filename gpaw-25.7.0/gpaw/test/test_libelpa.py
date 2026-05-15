import pytest
from gpaw.utilities.elpa import LibElpa
import numpy as np
import scipy as sp
from gpaw.blacs import BlacsGrid
from gpaw.mpi import world

pytestmark = pytest.mark.skipif(not LibElpa.have_elpa(),
                                reason='not LibElpa.have_elpa()')


@pytest.mark.ci
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('eigensolver', ['elpa', 'scalapack'])
@pytest.mark.parametrize('eigentype', ['normal', 'general'])
def test_libelpa(dtype, eigensolver, eigentype):
    rng = np.random.RandomState(87878787)

    if world.size == 1:
        shape = 1, 1
    else:
        shape = world.size // 2, 2
    bg = BlacsGrid(world, *shape)

    M = 8
    blocksize = 2

    desc = bg.new_descriptor(M, M, blocksize, blocksize)
    sdesc = desc.as_serial()

    Aserial = sdesc.zeros(dtype=dtype)
    if world.rank == 0:
        Aserial[:] = rng.rand(*Aserial.shape)
        if dtype == complex:
            Aserial.imag += rng.rand(*Aserial.shape)
        Aserial += Aserial.T.copy().conj()
    A = desc.distribute_from_master(Aserial)
    C2 = desc.zeros(dtype=dtype)
    eps2 = np.zeros(M)

    if eigentype == 'normal':
        if world.rank == 0:
            eps1, C1 = np.linalg.eigh(Aserial)

        if eigensolver == 'elpa':
            elpa = LibElpa(desc)
            elpa.diagonalize(A.copy(), C2, eps2)
        elif eigensolver == 'scalapack':
            desc.diagonalize_dc(A.copy(), C2, eps2)
    elif eigentype == 'general':
        Sserial = sdesc.zeros(dtype=dtype)
        if world.rank == 0:
            Sserial[:] = np.eye(M)
            Sserial[3, 1] += 0.5
            if dtype == complex:
                Sserial[2, 4] += 0.2j
        S = desc.distribute_from_master(Sserial)

        if world.rank == 0:
            eps1, C1 = sp.linalg.eigh(Aserial, Sserial)

        if eigensolver == 'elpa':
            elpa = LibElpa(desc)
            elpa.general_diagonalize(A.copy(), S.copy(), C2, eps2)
        elif eigensolver == 'scalapack':
            desc.general_diagonalize_dc(A.copy(), S.copy(), C2, eps2)

    if world.rank == 0:
        print(eps1)
        print(eps2)
        err = np.abs(eps1 - eps2).max()
        assert err < 1e-13, err
