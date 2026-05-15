import numpy as np
from gpaw.blacs import BlacsDescriptor, BlacsGrid, Redistributor


class Blocks1D:
    def __init__(self, blockcomm, N):
        self.blockcomm = blockcomm
        self.N = N  # Global number of points

        self.blocksize = (N + blockcomm.size - 1) // blockcomm.size
        self.a = min(blockcomm.rank * self.blocksize, N)
        self.b = min(self.a + self.blocksize, N)
        self.nlocal = self.b - self.a

        self.myslice = slice(self.a, self.b)

    def all_gather(self, in_myix):
        """All-gather array where the first dimension is block distributed.

        Here, myi is understood as the distributed index, whereas x are the
        remaining (global) dimensions."""
        # Set up buffers
        buf_myix = self.local_communication_buffer(in_myix)
        buf_ix = self.global_communication_buffer(in_myix)

        # Excecute all-gather
        self.blockcomm.all_gather(buf_myix, buf_ix)
        out_ix = buf_ix[:self.N]

        return out_ix

    def gather(self, in_myix, root=0):
        """Gather array to root where the first dimension is block distributed.
        """
        assert root in range(self.blockcomm.size)

        # Set up buffers
        buf_myix = self.local_communication_buffer(in_myix)
        if self.blockcomm.rank == root:
            buf_ix = self.global_communication_buffer(in_myix)
        else:
            buf_ix = None

        # Excecute gather
        self.blockcomm.gather(buf_myix, root, buf_ix)
        if self.blockcomm.rank == root:
            out_ix = buf_ix[:self.N]
        else:
            out_ix = None

        return out_ix

    def local_communication_buffer(self, in_myix):
        """Set up local communication buffer."""
        if in_myix.shape[0] == self.blocksize and in_myix.flags.contiguous:
            buf_myix = in_myix  # Use input array as communication buffer
        else:
            assert in_myix.shape[0] == self.nlocal
            buf_myix = np.empty(
                (self.blocksize,) + in_myix.shape[1:], in_myix.dtype)
            buf_myix[:self.nlocal] = in_myix

        return buf_myix

    def global_communication_buffer(self, in_myix):
        """Set up global communication buffer."""
        buf_ix = np.empty(
            (self.blockcomm.size * self.blocksize,) + in_myix.shape[1:],
            dtype=in_myix.dtype)
        return buf_ix

    def find_global_index(self, i):
        """Find rank and local index of the global index i"""
        rank = i // self.blocksize
        li = i % self.blocksize

        return rank, li


def block_partition(comm, nblocks):
    r"""Partition the communicator into a 2D array with horizontal
    and vertical communication.

         Communication between blocks (blockcomm)
    <----------------------------------------------->
     _______________________________________________
    |     |     |     |     |     |     |     |     | ⋀
    |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  | |
    |_____|_____|_____|_____|_____|_____|_____|_____| |
    |     |     |     |     |     |     |     |     | | Communication inside
    |  8  |  9  | 10  | 11  | 12  | 13  | 14  | 15  | | blocks
    |_____|_____|_____|_____|_____|_____|_____|_____| | (intrablockcomm)
    |     |     |     |     |     |     |     |     | |
    | 16  | 17  | 18  | 19  | 20  | 21  | 22  | 23  | |
    |_____|_____|_____|_____|_____|_____|_____|_____| ⋁

    """
    if nblocks == 'max':
        # Maximize the number of blocks
        nblocks = comm.size
    assert isinstance(nblocks, int)
    assert nblocks > 0 and nblocks <= comm.size, comm.size
    assert comm.size % nblocks == 0, comm.size

    # Communicator between different blocks
    if nblocks == comm.size:
        blockcomm = comm
    else:
        rank1 = comm.rank // nblocks * nblocks
        rank2 = rank1 + nblocks
        blockcomm = comm.new_communicator(range(rank1, rank2))

    # Communicator inside each block
    ranks = range(comm.rank % nblocks, comm.size, nblocks)
    if nblocks == 1:
        assert len(ranks) == comm.size
        intrablockcomm = comm
    else:
        intrablockcomm = comm.new_communicator(ranks)

    assert blockcomm.size * intrablockcomm.size == comm.size

    return blockcomm, intrablockcomm


class PlaneWaveBlockDistributor:
    """Functionality to shuffle block distribution of pair functions
    in the plane wave basis."""

    def __init__(self, world, blockcomm, intrablockcomm):
        self.world = world
        self.blockcomm = blockcomm
        self.intrablockcomm = intrablockcomm

    @property
    def fully_block_distributed(self):
        return self.world.compare(self.blockcomm) == 'ident'

    def new_distributor(self, *, nblocks):
        """Set up a new PlaneWaveBlockDistributor."""
        world = self.world
        blockcomm, intrablockcomm = block_partition(comm=world,
                                                    nblocks=nblocks)
        blockdist = PlaneWaveBlockDistributor(world, blockcomm, intrablockcomm)

        return blockdist

    def _redistribute(self, in_wGG, nw):
        """Redistribute array.

        Switch between two kinds of parallel distributions:

        1) parallel over G-vectors (second dimension of in_wGG)
        2) parallel over frequency (first dimension of in_wGG)

        Returns new array using the memory in the 1-d array out_x.
        """

        comm = self.blockcomm

        if comm.size == 1:
            return in_wGG

        mynw = (nw + comm.size - 1) // comm.size
        nG = in_wGG.shape[2]
        mynG = (nG + comm.size - 1) // comm.size

        bg1 = BlacsGrid(comm, comm.size, 1)
        bg2 = BlacsGrid(comm, 1, comm.size)
        md1 = BlacsDescriptor(bg1, nw, nG**2, mynw, nG**2)
        md2 = BlacsDescriptor(bg2, nw, nG**2, nw, mynG * nG)

        if len(in_wGG) == nw:
            mdin = md2
            mdout = md1
        else:
            mdin = md1
            mdout = md2

        r = Redistributor(comm, mdin, mdout)

        # mdout.shape[1] is always divisible by nG because
        # every block starts at a multiple of nG, and the last block
        # ends at nG² which of course also is divisible.  Nevertheless:
        assert mdout.shape[1] % nG == 0
        # (If it were not divisible, we would "lose" some numbers and the
        #  redistribution would be corrupted.)

        inbuf = in_wGG.reshape(mdin.shape)
        # numpy.reshape does not *guarantee* that the reshaped view will
        # be contiguous. To support redistribution of input arrays with an
        # arbitrary allocation layout, we make sure that the corresponding
        # input BLACS buffer in contiguous
        inbuf = np.ascontiguousarray(inbuf)

        outbuf = np.empty(mdout.shape, complex)

        r.redistribute(inbuf, outbuf)

        outshape = (mdout.shape[0], mdout.shape[1] // nG, nG)
        out_wGG = outbuf.reshape(outshape)
        assert out_wGG.flags.contiguous  # Since mdout.shape[1] % nG == 0

        return out_wGG

    def has_distribution(self, array, nw, distribution):
        """Check if array 'array' has distribution 'distribution'."""
        if distribution not in ['wGG', 'WgG', 'zGG', 'ZgG']:
            raise ValueError(f'Invalid dist_type: {distribution}')

        comm = self.blockcomm
        if comm.size == 1:
            # In serial, all distributions are equivalent
            return True

        # At the moment, only wGG and WgG distributions are supported. zGG and
        # ZgG are complex frequency aliases for wGG and WgG respectively
        assert len(array.shape) == 3
        nG = array.shape[2]
        if array.shape[1] < array.shape[2]:
            # Looks like array is WgG/ZgG distributed
            gblocks = Blocks1D(comm, nG)
            assert array.shape == (nw, gblocks.nlocal, nG)
            return distribution in ['WgG', 'ZgG']
        else:
            # Looks like array is wGG/zGG distributed
            wblocks = Blocks1D(comm, nw)
            assert array.shape == (wblocks.nlocal, nG, nG)
            return distribution in ['wGG', 'zGG']

    def distribute_as(self, array, nw, distribution):
        """Redistribute array.

        Switch between two kinds of parallel distributions:

        1) parallel over G-vectors (distribution in ['WgG', 'ZgG'])
        2) parallel over frequency (distribution in ['wGG', 'zGG'])
        """
        if self.has_distribution(array, nw, distribution):
            # If the array already has the requested distribution, do nothing
            return array
        else:
            return self._redistribute(array, nw)

    def distribute_frequencies(self, in_wGG, nw):
        """Distribute frequencies to all cores."""

        world = self.world
        comm = self.blockcomm

        if world.size == 1:
            return in_wGG

        mynw = (nw + world.size - 1) // world.size
        nG = in_wGG.shape[2]
        mynG = (nG + comm.size - 1) // comm.size

        wa = min(world.rank * mynw, nw)
        wb = min(wa + mynw, nw)

        if self.blockcomm.size == 1:
            return in_wGG[wa:wb].copy()

        if self.intrablockcomm.rank == 0:
            bg1 = BlacsGrid(comm, 1, comm.size)
            in_wGG = in_wGG.reshape((nw, -1))
        else:
            bg1 = BlacsGrid(None, 1, 1)
            # bg1 = DryRunBlacsGrid(mpi.serial_comm, 1, 1)
            in_wGG = np.zeros((0, 0), complex)
        md1 = BlacsDescriptor(bg1, nw, nG**2, nw, mynG * nG)

        bg2 = BlacsGrid(world, world.size, 1)
        md2 = BlacsDescriptor(bg2, nw, nG**2, mynw, nG**2)

        r = Redistributor(world, md1, md2)
        shape = (wb - wa, nG, nG)
        out_wGG = np.empty(shape, complex)
        r.redistribute(in_wGG, out_wGG.reshape((wb - wa, nG**2)))

        return out_wGG
