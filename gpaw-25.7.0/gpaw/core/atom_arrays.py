from __future__ import annotations

import numbers
from typing import Sequence, overload, Literal

import numpy as np
from gpaw.core.matrix import Matrix
from gpaw.gpu import cupy as cp, XP
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import prod, zips
from gpaw.typing import Array1D, ArrayLike1D
from gpaw.new.c import dH_aii_times_P_ani_gpu
from gpaw.utilities import as_real_dtype


class AtomArraysLayout(XP):
    def __init__(self,
                 shapes: Sequence[int | tuple[int, ...]],
                 atomdist: AtomDistribution | MPIComm = serial_comm,
                 dtype=float,
                 xp=None):
        """Description of layout of atom arrays.

        Parameters
        ----------
        shapes:
            Shapse of arrays - one for each atom.
        atomdist:
            Distribution of atoms.
        dtype:
            Data-type (float or complex).
        """
        self.shape_a = [shape if isinstance(shape, tuple) else (shape,)
                        for shape in shapes]
        if not isinstance(atomdist, AtomDistribution):
            atomdist = AtomDistribution(np.zeros(len(shapes), int), atomdist)
        self.atomdist = atomdist
        self.dtype = np.dtype(dtype)
        XP.__init__(self, xp or np)

        self.size = sum(prod(shape) for shape in self.shape_a)

        self.myindices = []
        self.mysize = 0
        I1 = 0
        for a in atomdist.indices:
            I2 = I1 + prod(self.shape_a[a])
            self.myindices.append((a, I1, I2))
            self.mysize += I2 - I1
            I1 = I2

    def __len__(self):
        return len(self.shape_a)

    def __repr__(self):
        return (f'AtomArraysLayout({self.shape_a}, {self.atomdist}, '
                f'{self.dtype}, xp={self.xp.__name__})')

    def new(self, atomdist=None, dtype=None, xp=None):
        """Create new AtomsArrayLayout object with new atomdist."""
        return AtomArraysLayout(self.shape_a,
                                atomdist or self.atomdist,
                                dtype or self.dtype,
                                xp or self.xp)

    def empty(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> AtomArrays:
        """Create new AtomArrays object.

        parameters
        ----------
        dims:
            Extra dimensions.
        comm:
            Distribute dimensions along this communicator.
        """
        return AtomArrays(self, dims, comm)

    def zeros(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> AtomArrays:
        aa = self.empty(dims, comm)
        aa.data[:] = 0.0
        return aa

    def sizes(self) -> tuple[list[dict[int, int]], Array1D]:
        """Compute array sizes for all ranks.

        >>> AtomArraysLayout([3, 4]).sizes()
        ([{0: 3, 1: 4}], array([7]))
        """
        comm = self.atomdist.comm
        size_ra: list[dict[int, int]] = [{} for _ in range(comm.size)]
        size_r = np.zeros(comm.size, int)
        for a, (rank, shape) in enumerate(zips(self.atomdist.rank_a,
                                               self.shape_a)):
            size = prod(shape)
            size_ra[rank][a] = size
            size_r[rank] += size
        return size_ra, size_r


class AtomDistribution:
    def __init__(self, ranks: ArrayLike1D, comm: MPIComm = serial_comm):
        """Atom-distribution.

        Parameters
        ----------
        ranks:
            List of ranks, one rank per atom.
        comm:
            MPI-communicator.
        """
        self.comm = comm
        self.rank_a = np.array(ranks)
        # convert from np.int64 -> int:
        self.indices = [int(a) for a in np.where(self.rank_a == comm.rank)[0]]

    def __len__(self) -> int:
        return len(self.rank_a)

    @classmethod
    def from_number_of_atoms(cls,
                             natoms: int,
                             comm: MPIComm = serial_comm) -> AtomDistribution:
        """Distribute atoms evenly.

        >>> AtomDistribution.from_number_of_atoms(3).rank_a
        array([0, 0, 0])
        """
        blocksize = (natoms + comm.size - 1) // comm.size
        rank_a = np.empty(natoms, int)
        a1 = 0
        for rank in range(comm.size):
            a2 = a1 + blocksize
            rank_a[a1:a2] = rank
            if a2 >= natoms:
                break
            a1 = a2
        return cls(rank_a, comm)

    @classmethod
    def from_atom_indices(cls,
                          atom_indices: Sequence[int],
                          comm: MPIComm = serial_comm,
                          *,
                          natoms: int | None = None) -> AtomDistribution:
        """Create distribution from atom indices.

        >>> AtomDistribution.from_atom_indices([0, 1, 2]).rank_a
        array([0, 0, 0])
        """
        if natoms is None:
            natoms = comm.max_scalar(max(atom_indices)) + 1
        rank_a = np.zeros(natoms, int)  # type: ignore
        rank_a[atom_indices] = comm.rank
        comm.sum(rank_a)
        return cls(rank_a, comm)

    def __repr__(self):
        return (f'AtomDistribution(ranks={self.rank_a}, '
                f'comm={self.comm.rank}/{self.comm.size})')

    def gather(self):
        return AtomDistribution(np.zeros(len(self.rank_a), int))


class AtomArrays:
    def __init__(self,
                 layout: AtomArraysLayout,
                 dims: int | Sequence[int] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray | None = None):
        """AtomArrays object.

        parameters
        ----------
        layout:
            Layout-description.
        dims:
            Extra dimensions.
        comm:
            Distribute dimensions along this communicator.
        data:
            Data array for storage.
        """
        myshape = (layout.mysize,)
        domain_comm = layout.atomdist.comm
        dtype = layout.dtype

        self.myshape = myshape
        self.comm = comm
        self.domain_comm = domain_comm

        # convert int to tuple:
        self.dims = tuple(dims) if not isinstance(dims, int) else (dims,)

        if self.dims:
            d1, d2 = self.my_slice()
            mydims0 = d2 - d1
            self.mydims = (mydims0,) + self.dims[1:]
        else:
            self.mydims = ()

        fullshape = self.mydims + self.myshape

        if data is not None:
            if data.shape != fullshape:
                raise ValueError(
                    f'Bad shape for data: {data.shape} != {fullshape}')
            if data.dtype != dtype:
                raise ValueError(
                    f'Bad dtype for data: {data.dtype} != {dtype}')
        else:
            data = layout.xp.empty(fullshape, dtype)

        self.data = data
        self._matrix: Matrix | None = None  # matrix view

        self.layout = layout
        self._arrays = {}
        for a, I1, I2 in layout.myindices:
            self._arrays[a] = self.data[..., I1:I2].reshape(
                self.mydims + layout.shape_a[a])

    def __len__(self) -> int:
        return len(self.layout)

    def my_slice(self) -> tuple[int, int]:
        mydims0 = (self.dims[0] + self.comm.size - 1) // self.comm.size
        d1 = min(self.comm.rank * mydims0, self.dims[0])
        d2 = min((self.comm.rank + 1) * mydims0, self.dims[0])
        return d1, d2

    def __repr__(self):
        txt = f'AtomArrays({self.layout}, dims={self.dims}'
        if self.comm.size > 1:
            txt += f', comm={self.comm.rank}/{self.comm.size}'
        return txt + ')'

    @property
    def matrix(self) -> Matrix:
        if self._matrix is not None:
            return self._matrix

        shape = (self.dims[0], prod(self.dims[1:]) * prod(self.myshape))
        myshape = (self.mydims[0], prod(self.mydims[1:]) * prod(self.myshape))
        dist = (self.comm, -1, 1)

        data = self.data.reshape(myshape)
        self._matrix = Matrix(*shape, data=data, dist=dist)

        return self._matrix

    def new(self, *, layout=None, data=None, xp=None):
        """Create new AtomArrays object of same kind.

        Parameters
        ----------
        layout:
            Layout-description.
        data:
            Array to use for storage.
        """
        if xp is not None:
            assert layout is None
            assert data is None
            if self.layout.xp is not xp:
                layout = self.layout.new(xp=xp)
        return AtomArrays(layout or self.layout,
                          self.dims,
                          self.comm,
                          data=data)

    def to_cpu(self):
        if self.layout.xp is np:
            return self
        return self.new(layout=self.layout.new(xp=np),
                        data=cp.asnumpy(self.data))

    def to_xp(self, xp):
        if self.layout.xp is xp:
            assert xp is np, 'cp -> cp should not be needed!'
            return self
        if xp is np:
            return self.new(layout=self.layout.new(xp=np),
                            data=cp.asnumpy(self.data))
        return self.new(layout=self.layout.new(xp=cp),
                        data=cp.asarray(self.data))

    @overload
    def __getitem__(self, a: int) -> np.ndarray:
        ...

    @overload
    def __getitem__(self, a: tuple) -> AtomArrays:
        ...

    def __getitem__(self, a):
        if isinstance(a, numbers.Integral):
            return self._arrays[a]
        assert len(self.dims) >= 1
        a0, a1 = a
        assert a0 == slice(None)
        data = self.data[a1]
        a_ai = AtomArrays(self.layout, dims=data.shape[:-1], data=data)
        return a_ai

    def copy(self):
        return self.new(data=self.data.copy())

    def get(self, a):
        return self._arrays.get(a)

    def __setitem__(self, a, value):
        self._arrays[a][:] = value

    def __contains__(self, a):
        return a in self._arrays

    def items(self):
        return self._arrays.items()

    def keys(self):
        return self._arrays.keys()

    def values(self):
        return self._arrays.values()

    @overload
    def gather(self, broadcast: Literal[False] = False, copy: bool = False
               ) -> AtomArrays | None:
        ...

    @overload
    def gather(self, broadcast: Literal[True], copy: bool = False
               ) -> AtomArrays:
        ...

    def gather(self, broadcast=False, copy=False):
        """Gather all atoms on master."""
        comm = self.layout.atomdist.comm
        if comm.size == 1:
            if copy:
                aa = self.new()
                aa.data[:] = self.data
                return aa
            return self

        if comm.rank == 0 or broadcast:
            aa = self.new(layout=self.layout.new(atomdist=serial_comm))
        else:
            aa = None

        if comm.rank == 0:
            size_ra, size_r = self.layout.sizes()
            n = prod(self.mydims)
            m = size_r.max()
            buffer = self.layout.xp.empty(n * m, self.layout.dtype)
            for rank in range(1, comm.size):
                buf = buffer[:n * size_r[rank]].reshape((n, size_r[rank]))
                comm.receive(buf, rank)
                b1 = 0
                for a, size in size_ra[rank].items():
                    b2 = b1 + size
                    A = aa[a]
                    A[:] = buf[:, b1:b2].reshape(A.shape)
                    b1 = b2
            for a, array in self._arrays.items():
                aa[a] = array
        else:
            comm.send(self.data, 0)

        if broadcast:
            comm.broadcast(aa.data, 0)

        return aa

    def scatter_from(self,
                     data: np.ndarray | AtomArrays | None = None) -> None:
        """Scatter atoms."""
        if isinstance(data, AtomArrays):
            data = data.data
        comm = self.layout.atomdist.comm
        xp = self.layout.xp
        if comm.size == 1:
            assert data is not None
            self.data[:] = data
            return

        if comm.rank != 0:
            comm.receive(self.data, 0, 42)
            return

        size_ra, size_r = self.layout.sizes()
        aa = self.new(layout=self.layout.new(atomdist=serial_comm),
                      data=data)
        requests = []
        for rank, (totsize, size_a) in enumerate(zips(size_r, size_ra)):
            if rank != 0:
                buf = xp.empty(self.mydims + (totsize,), self.layout.dtype)
                b1 = 0
                for a, size in size_a.items():
                    b2 = b1 + size
                    buf[..., b1:b2] = aa[a].reshape(self.mydims + (size,))
                    b1 = b2
                request = comm.send(buf, rank, 42, False)
                # Remember to store a reference to the
                # send buffer (buf) so that is isn't
                # deallocated
                requests.append((request, buf))
            else:
                for a in size_a:
                    self[a] = aa[a]

        for request, _ in requests:
            comm.wait(request)

    def to_lower_triangle(self):
        """Convert `N*N` matrices to `N*(N+1)/2` vectors.

        >>> a = AtomArraysLayout([(3, 3)]).empty()
        >>> a[0][:] = [[11, 12, 13],
        ...            [12, 22, 23],
        ...            [13, 23, 33]]
        >>> a.to_lower_triangle()[0]
        array([11., 12., 22., 13., 23., 33.])
        """
        shape_a = []
        for i1, i2 in self.layout.shape_a:
            assert i1 == i2
            shape_a.append((i1 * (i1 + 1) // 2,))
        xp = self.layout.xp
        layout = AtomArraysLayout(shape_a,
                                  self.layout.atomdist,
                                  dtype=self.layout.dtype,
                                  xp=xp)
        a_axp = layout.empty(self.dims)
        for a_xii, a_xp in zips(self.values(), a_axp.values()):
            i = a_xii.shape[-1]
            L = xp.tril_indices(i)
            for a_p, a_ii in zips(a_xp.reshape((-1, i * (i + 1) // 2)),
                                  a_xii.reshape((-1, i, i))):
                a_p[:] = a_ii[L]

        return a_axp

    def to_full(self):
        r"""Convert `N(N+1)/2` vectors to `N\times N` matrices.

        >>> a = AtomArraysLayout([6]).empty()
        >>> a[0][:] = [1, 2, 3, 4, 5, 6]
        >>> a.to_full()[0]
        array([[1., 2., 4.],
               [2., 3., 5.],
               [4., 5., 6.]])
        """
        shape_a = []
        for (p,) in self.layout.shape_a:
            i = int((2 * p + 0.25)**0.5)
            shape_a.append((i, i))
        layout = AtomArraysLayout(shape_a,
                                  self.layout.atomdist,
                                  self.layout.dtype)
        a_axii = layout.empty(self.dims)
        for a_xp, a_xii in zips(self.values(), a_axii.values()):
            i = a_xii.shape[-1]
            a_xii[(...,) + np.tril_indices(i)] = a_xp
            u = (...,) + np.triu_indices(i, 1)
            a_xii[u] = np.swapaxes(a_xii, -1, -2)[u].conj()
        return a_axii

    def moved(self, atomdist):
        if (self.layout.atomdist.rank_a == atomdist.rank_a).all():
            return self
        assert self.comm.size == 1
        layout = self.layout.new(atomdist=atomdist)
        new = layout.empty(self.dims)
        comm = atomdist.comm
        xp = self.layout.xp
        requests = []
        for a, I1, I2 in self.layout.myindices:
            r = layout.atomdist.rank_a[a]
            if r == comm.rank:
                new[a][:] = self[a]
            else:
                requests.append(comm.send(xp.ascontiguousarray(self[a]),
                                          r, block=False))

        for a, I1, I2 in layout.myindices:
            r = self.layout.atomdist.rank_a[a]
            if r != comm.rank:
                target = new[a]
                buf = xp.empty_like(target)
                comm.receive(buf, r)
                target[:] = buf

        comm.waitall(requests)
        return new

    def redist(self,
               atomdist: AtomDistribution,
               comm1: MPIComm,
               comm2: MPIComm) -> AtomArrays:
        layout = self.layout.new(atomdist=atomdist)
        result = layout.empty(dims=self.dims)
        if comm1.rank == 0:
            a = self.gather()
        else:
            a = None
        if comm2.rank == 0:
            result.scatter_from(a)
        comm2.broadcast(result.data, 0)
        return result

    def block_diag_multiply(self,
                            block_diag_matrix_axii: AtomArrays,
                            out_ani: AtomArrays,
                            index: int | None = None) -> None:
        """Multiply by block diagonal matrix.

        with A, B and C refering to ``self``, ``block_diag_matrix_axii`` and
        ``out_ani``:::

            --  a   a      a
            >  A   B   -> C
            --  ni  ij     nj
            i

        If index is not None, ``block_diag_matrix_axii`` must have an extra
        dimension: :math:`B_{ij}^{ax}` and x=index is used.
        """
        xp = self.layout.xp
        if xp is np:
            if index is not None:
                block_diag_matrix_axii = block_diag_matrix_axii[:, index]
            for P_ni, dX_ii, out_ni in zips(self.values(),
                                            block_diag_matrix_axii.values(),
                                            out_ani.values()):
                out_ni[:] = P_ni @ dX_ii
            return

        ni_a = xp.array(
            [I2 - I1 for a, I1, I2 in self.layout.myindices],
            dtype=np.int32)
        data = block_diag_matrix_axii.data
        if index is not None:
            data = data[index]
        if self.data.size > 0:
            realdtype = as_real_dtype(self.data.dtype)
            dH_aii_times_P_ani_gpu(xp.asarray(data, dtype=realdtype),
                                   ni_a, self.data, out_ani.data)
