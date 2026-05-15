from typing import Any, Optional, Dict
from collections.abc import Mapping

import numpy as np

from gpaw.matrix import Matrix
from gpaw.mpi import serial_comm
from gpaw.utilities.partition import AtomPartition
from gpaw.typing import Array2D, ArrayLike1D

MPIComm = Any


class Projections(Mapping):
    def __init__(self,
                 nbands: int,
                 nproj_a: ArrayLike1D,
                 atom_partition: AtomPartition,
                 bcomm: MPIComm = None,
                 collinear=True,
                 spin=0,
                 dtype=None,
                 data=None,
                 bdist=None):
        if bdist is None:
            self.bcomm = bcomm or serial_comm
            bdist = (self.bcomm, self.bcomm.size, 1)
        else:
            assert bcomm is None
            self.bdist = bdist  # For calling "new"
            self.bcomm = bdist[0]

        self.nproj_a = np.asarray(nproj_a)
        self.atom_partition = atom_partition
        self.collinear = collinear
        self.spin = spin
        self.nbands = nbands

        self.indices = []
        self.map = {}
        I1 = 0

        for a in self.atom_partition.my_indices:
            ni = self.nproj_a[a]
            I2 = I1 + ni
            self.indices.append((a, I1, I2))
            self.map[a] = (I1, I2)
            I1 = I2

        if not collinear:
            I1 *= 2

        if dtype is None and data is None:
            dtype = float if collinear else complex

        self.matrix = Matrix(nbands, I1, dtype, data, dist=bdist)

        if collinear:
            self.myshape = self.matrix.array.shape
        else:
            self.myshape = (len(self.matrix.array), 2, I1 // 2)

    @property
    def array(self):
        if self.collinear:
            return self.matrix.array
        else:
            return self.matrix.array.reshape(self.myshape)

    def new(self, bcomm='inherit', nbands=None, atom_partition=None):
        if bcomm == 'inherit':
            if hasattr(self, 'bdist'):
                return Projections(
                    nbands or self.nbands, self.nproj_a,
                    self.atom_partition if atom_partition is None
                    else atom_partition,
                    collinear=self.collinear,
                    spin=self.spin,
                    dtype=self.matrix.dtype,
                    bdist=self.bdist)
            else:
                bcomm = self.bcomm
        elif bcomm is None:
            bcomm = serial_comm

        return Projections(
            nbands or self.nbands, self.nproj_a,
            self.atom_partition if atom_partition is None else atom_partition,
            bcomm, self.collinear, self.spin, self.matrix.dtype)

    def view(self, n1: int, n2: int) -> 'Projections':
        return Projections(n2 - n1, self.nproj_a,
                           self.atom_partition,
                           self.bcomm, self.collinear, self.spin,
                           self.matrix.dtype, self.matrix.array[n1:n2])

    def __getitem__(self, a):
        I1, I2 = self.map[a]
        return self.array[..., I1:I2]

    def __iter__(self):
        return iter(self.map)

    def __len__(self):
        return len(self.map)

    def broadcast(self) -> 'Projections':
        ap = AtomPartition(serial_comm, np.zeros(len(self.nproj_a), int))
        P = self.new(atom_partition=ap)
        comm = self.atom_partition.comm
        for a, rank in enumerate(self.atom_partition.rank_a):
            P1_ni = P[a]
            if comm.rank == rank:
                P_ni = self[a].copy()
            else:
                P_ni = np.empty_like(P1_ni)
            comm.broadcast(P_ni, rank)
            P1_ni[:] = P_ni
        return P

    def redist(self, atom_partition) -> 'Projections':
        """Redistribute atoms."""
        P = self.new(atom_partition=atom_partition)
        arraydict = self.toarraydict()
        arraydict.redistribute(atom_partition)
        P.fromarraydict(arraydict)
        return P

    def collect(self) -> Optional[Array2D]:
        """Collect all bands and atoms to master."""
        if self.bcomm.size == 1:
            P = self.matrix
        else:
            P = self.matrix.new(dist=(self.bcomm, 1, 1))
            self.matrix.redist(P)

        if self.bcomm.rank > 0:
            return None

        if self.atom_partition.comm.size == 1:
            return P.array

        P_In = self.collect_atoms(P)
        if P_In is not None:
            return P_In.T

        return None

    def toarraydict(self):
        shape = self.myshape[:-1]
        shapes = [shape + (nproj,) for nproj in self.nproj_a]

        d = self.atom_partition.arraydict(shapes, self.matrix.array.dtype)
        for a, I1, I2 in self.indices:
            d[a][:] = self.array[..., I1:I2]  # Blocks will be contiguous
        return d

    def fromarraydict(self, d):
        assert d.partition == self.atom_partition
        for a, I1, I2 in self.indices:
            self.array[..., I1:I2] = d[a]

    def collect_atoms(self, P):
        if self.atom_partition.comm.rank == 0:
            nproj = sum(self.nproj_a)
            P_In = np.empty((nproj, P.array.shape[0]), dtype=P.array.dtype)

            I1 = 0
            myI1 = 0
            for nproj, rank in zip(self.nproj_a, self.atom_partition.rank_a):
                I2 = I1 + nproj
                if rank == 0:
                    myI2 = myI1 + nproj
                    P_In[I1:I2] = P.array[:, myI1:myI2].T
                    myI1 = myI2
                else:
                    self.atom_partition.comm.receive(P_In[I1:I2], rank)
                I1 = I2
            return P_In
        else:
            for a, I1, I2 in self.indices:
                self.atom_partition.comm.send(P.array[:, I1:I2].T.copy(), 0)
            return None

    def as_dict_on_master(self, n1: int, n2: int) -> Dict[int, Array2D]:
        P_nI = self.collect()
        if P_nI is None:
            return {}
        I1 = 0
        P_ani = {}
        for a, ni in enumerate(self.nproj_a):
            I2 = I1 + ni
            P_ani[a] = P_nI[n1:n2, I1:I2]
            I1 = I2
        return P_ani
