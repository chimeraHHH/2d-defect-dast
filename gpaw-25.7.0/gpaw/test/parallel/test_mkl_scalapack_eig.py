from gpaw.utilities.scalapack import mkl_scalapack_diagonalize_non_symmetric
from gpaw.utilities.scalapack import have_mkl
from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.mpi import world
from gpaw.matrix import suggest_blocking

import numpy as np

import pytest


def test_mkl_eig():
    if not have_mkl():
        pytest.skip('MKL Scalapack functions only testable with '
                    'the intelMKL toolchain active and '
                    'intelmkl = True in siteconfig.py')

    gsize = 200

    scgrid = BlacsGrid(world, 1, 1)
    scdesc = scgrid.new_descriptor(gsize, gsize, gsize, gsize)
    asc = scdesc.empty(dtype=complex)

    if world.rank == 0:
        np.random.seed(1337)
        asc[:] = np.random.rand(gsize, gsize)

    ga, gb, bsize = suggest_blocking(gsize, world.size)
    grid = BlacsGrid(world, ga, gb)
    desc = grid.new_descriptor(gsize, gsize, bsize, bsize)
    redist_sc_to_full = Redistributor(world, scdesc, desc)
    a = redist_sc_to_full.redistribute(asc)
    z = desc.empty(dtype=complex)

    eps = np.empty(gsize, dtype=complex)
    mkl_scalapack_diagonalize_non_symmetric(desc, a, z, eps)

    redist_full_to_sc = Redistributor(world, desc, scdesc)
    zsc = redist_full_to_sc.redistribute(z)

    if world.rank == 0:
        assert np.linalg.norm(
            asc - zsc @ np.diag(eps) @ np.linalg.inv(zsc)
        ) == pytest.approx(0, abs=1e-8)

        # Test against lapack
        [eps2, z2] = np.linalg.eig(asc)

        assert np.sum(np.abs(np.subtract.outer(eps, eps2)) < 1e-10) == gsize
