"""Parallelization scheme for frequency–planewave–planewave arrays."""
from gpaw.mpi import world
from gpaw.response.pw_parallelization import block_partition
from gpaw.utilities.scalapack import scalapack_set, scalapack_solve
from gpaw.blacs import BlacsGrid
import numpy as np


def get_blocksize(length, commsize):
    return -(-length // commsize)


def get_strides(cpugrid):
    return np.array([cpugrid[1] * cpugrid[2], cpugrid[2], 1], int)


class Grid:
    def __init__(self, comm, shape, cpugrid=None, blocksize=None):
        self.comm = comm
        self.shape = shape

        if cpugrid is None:
            cpugrid = choose_parallelization(shape[0], shape[1],
                                             comm.size)

        self.cpugrid = cpugrid

        if blocksize is None:
            blocksize = [get_blocksize(size, commsize)
                         for size, commsize in zip(shape, cpugrid)]
            # XXX scalapack blocksize hack
            blocksize[1] = blocksize[2] = max(blocksize[1:])

        # XXX Our scalapack interface does NOT like it when blocksizes
        # are not the same.  There must be a bug.
        assert blocksize[1] == blocksize[2]

        self.blocksize = tuple(blocksize)
        self.cpugrid = cpugrid

        self.myparpos = self.rank2parpos(self.comm.rank)

        n_cp = self.get_domains()

        shape = []
        myslice_c = []
        for i in range(3):
            n_p = n_cp[i]
            parpos = self.myparpos[i]
            myend = n_p[parpos + 1]
            mystart = n_p[parpos]
            myslice_c.append(slice(mystart, myend))
            size = myend - mystart
            shape.append(size)

        self.myshape = tuple(shape)
        self.myslice = tuple(myslice_c)

    # TODO inherit these from array descriptor
    def zeros(self, dtype=float):
        return np.zeros(self.myshape, dtype=dtype)

    def get_domains(self):
        """Get definition of domains.

        Returns domains_cp where domains_cp[c][r + 1] - domains_cp[c][r]
        is the number of points in domain r along direction c.

        The second axis contains the "fencepost" locations
        of the grid: [0, blocksize, 2 * blocksize, ...]
        """
        domains_cp = []

        for i in range(3):
            n_p = np.empty(self.cpugrid[i] + 1, int)
            n_p[0] = 0
            n_p[1:] = self.blocksize[i]
            n_p[:] = n_p.cumsum().clip(0, self.shape[i])
            domains_cp.append(n_p)

        return domains_cp

    def rank2parpos(self, rank):
        # XXX Borrowing from gd -- we should eliminate this duplication.

        strides = get_strides(self.cpugrid)
        cpugrid_coord = np.array(
            [rank // strides[0],
             (rank % strides[0]) // strides[1],
             rank % strides[1]])

        return cpugrid_coord

    def redistribute(self, dstgrid, srcarray, dstarray):
        from gpaw.utilities.grid_redistribute import general_redistribute
        domains1 = self.get_domains()
        domains2 = dstgrid.get_domains()
        general_redistribute(self.comm, domains1, domains2,
                             self.rank2parpos, dstgrid.rank2parpos,
                             srcarray, dstarray, behavior='overwrite')

    def invert_inplace(self, x_wgg):
        # Build wgg grid choosing scalapack
        nscalapack_cores = np.prod(self.cpugrid[1:])
        blacs_comm, wcomm = block_partition(self.comm, nscalapack_cores)
        assert wcomm.size == self.cpugrid[0]
        assert blacs_comm.size * wcomm.size == self.comm.size
        for iw, x_gg in enumerate(x_wgg):
            bg = BlacsGrid(blacs_comm, *self.cpugrid[1:][::-1])
            desc = bg.new_descriptor(
                *self.shape[1:],
                *self.blocksize[1:])

            xtmp_gg = desc.empty(dtype=x_wgg.dtype)
            xtmp_gg[:] = x_gg.T

            righthand = desc.zeros(dtype=complex)
            scalapack_set(desc, righthand, alpha=0.0, beta=1.0, uplo='U')

            scalapack_solve(desc, desc, xtmp_gg, righthand)
            x_gg[:] = righthand.T


def get_x_WGG(WGG_grid):
    x_WGG = WGG_grid.zeros(dtype=complex)
    rng = np.random.RandomState(42)

    x_WGG.flat[:] = rng.random(x_WGG.size)
    x_WGG.flat[:] += rng.random(x_WGG.size) * 1j
    # XXX write also to imaginary parts

    nG = x_WGG.shape[1]

    xinv_WGG = np.zeros_like(x_WGG)
    if WGG_grid.comm.rank == 0:
        assert x_WGG.shape == WGG_grid.myshape
        for iw, x_GG in enumerate(x_WGG):
            x_GG += x_GG.T.conj().copy()
            x_GG += np.identity(nG) * 5
            eigs = np.linalg.eigvals(x_GG)
            assert all(eigs.real) > 0
            xinv_WGG[iw] = np.linalg.inv(x_GG)
    else:
        assert np.prod(x_WGG.shape) == 0
    return x_WGG, xinv_WGG


def factorize(N):
    for n in range(1, N + 1):
        if N % n == 0:
            yield N // n, n


def get_products(N):
    for a1, a2 in factorize(N):
        for a2p, a3 in factorize(a2):
            yield a1, a2p, a3


def choose_parallelization(nW, nG, commsize):
    min_badness = 10000000

    for wGG in get_products(commsize):
        wsize, gsize1, gsize2 = wGG
        nw = (nW + wsize - 1) // wsize

        if nw > nW:
            continue

        number_of_cores_with_zeros = (wsize * nw - nW) // nw
        scalapack_skew = (gsize1 - gsize2)**2
        scalapack_size = gsize1 * gsize2
        badness = (number_of_cores_with_zeros * 1000
                   + 10 * scalapack_skew + scalapack_size)

        # print(wsize, gsize1, gsize2, nw, number_of_cores_with_zeros, badness)
        if badness < min_badness:
            wGG_min = wGG
            min_badness = badness
    return wGG_min


def main(comm=world):
    nW = 3
    nG = 31

    cpugrid = choose_parallelization(nW, nG, comm.size)

    WGG = (nW, nG, nG)
    dtype = complex

    # Build serial grid (data only on rank 0)
    # and establish matrix and its inverse
    WGG_grid = Grid(comm, WGG, cpugrid, blocksize=WGG)
    x_WGG, xinv_WGG = get_x_WGG(WGG_grid)

    # Distribute to WgG grid:
    WgG_grid = Grid(comm, WGG, (1, comm.size, 1))
    x_WgG = np.zeros(WgG_grid.myshape, dtype=dtype)
    WGG_grid.redistribute(WgG_grid, x_WGG, x_WgG)

    wgg_grid = Grid(comm, WGG, cpugrid)
    print(f'cpugrid={cpugrid} blocksize={wgg_grid.blocksize} '
          f'shape={wgg_grid.shape} myshape={wgg_grid.myshape}')

    x_wgg = wgg_grid.zeros(dtype=dtype)
    WgG_grid.redistribute(wgg_grid, x_WgG, x_wgg)

    # By now let's distribute wgg back to WgG to check that numbers
    # are the same:
    x1_WgG = WgG_grid.zeros(dtype=dtype)
    wgg_grid.redistribute(WgG_grid, x_wgg, x1_WgG)
    assert np.allclose(x_WgG, x1_WgG)

    wgg_grid.invert_inplace(x_wgg)

    # Distribute the inverse wgg back to WGG:
    inv_x_WGG = WGG_grid.zeros(dtype=dtype)
    wgg_grid.redistribute(WGG_grid, x_wgg, inv_x_WGG)

    from gpaw.utilities.tools import tri2full
    if comm.rank == 0:
        for inv_x_GG in inv_x_WGG:
            tri2full(inv_x_GG, 'L')

        for x_GG, inv_x_GG in zip(x_WGG, inv_x_WGG):
            assert np.allclose(x_GG @ inv_x_GG, np.identity(nG))


if __name__ == '__main__':
    main()
