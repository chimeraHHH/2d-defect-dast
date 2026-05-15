from __future__ import annotations

from functools import cached_property
from math import pi
from typing import Sequence, Literal, TYPE_CHECKING
import numpy as np

import gpaw.fftw as fftw
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_centered_functions import UGAtomCenteredFunctions
from gpaw.core.domain import Domain
from gpaw.gpu import as_np, cupy_is_fake
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import zips
from gpaw.typing import (Array1D, Array2D, Array3D, Array4D, ArrayLike1D,
                         ArrayLike2D, Vector)
from gpaw.new.c import add_to_density, add_to_density_gpu, symmetrize_ft
from gpaw.fd_operators import Gradient

if TYPE_CHECKING:
    import plotly.graph_objects as go


class UGDesc(Domain['UGArray']):
    def __init__(self,
                 *,
                 cell: ArrayLike1D | ArrayLike2D,  # bohr
                 size: ArrayLike1D,
                 pbc=(True, True, True),
                 zerobc=(False, False, False),
                 kpt: Vector | None = None,  # in units of reciprocal cell
                 comm: MPIComm = serial_comm,
                 decomp: Sequence[Sequence[int]] | None = None,
                 dtype=None):
        """Description of 3D uniform grid.

        parameters
        ----------
        cell:
            Unit cell given as three floats (orthorhombic grid), six floats
            (three lengths and the angles in degrees) or a 3x3 matrix
            (units: bohr).
        size:
            Number of grid points along axes.
        pbc:
            Periodic boundary conditions flag(s).
        zerobc:
            Zero-boundary conditions flag(s).  Skip first grid-point
            (assumed to be zero).
        comm:
            Communicator for domain-decomposition.
        kpt:
            K-point for Block-boundary conditions specified in units of the
            reciprocal cell.
        decomp:
            Decomposition of the domain.
        dtype:
            Data-type (float or complex).
        """
        self.size_c = np.array(size, int)
        if isinstance(zerobc, int):
            zerobc = (zerobc,) * 3
        self.zerobc_c = np.array(zerobc, bool)

        if decomp is None:
            gd = GridDescriptor(size, pbc_c=~self.zerobc_c, comm=comm)
            decomp = gd.n_cp
        self.decomp_cp = [np.asarray(d) for d in decomp]

        self.parsize_c = np.array([len(d_p) - 1 for d_p in self.decomp_cp])
        self.mypos_c = np.unravel_index(comm.rank, self.parsize_c)

        self.start_c = np.array([d_p[p]
                                 for d_p, p
                                 in zips(self.decomp_cp, self.mypos_c)])
        self.end_c = np.array([d_p[p + 1]
                               for d_p, p
                               in zips(self.decomp_cp, self.mypos_c)])
        self.mysize_c = self.end_c - self.start_c

        Domain.__init__(self, cell, pbc, kpt, comm, dtype)
        self.myshape = tuple(self.mysize_c)

        self.dv = self.volume / self.size_c.prod()

        self.itemsize = 8 if self.dtype == float else 16

        if (self.zerobc_c & self.pbc_c).any():
            raise ValueError('Bad boundary conditions')

    @property
    def size(self):
        """Size of uniform grid."""
        return self.size_c.copy()

    def global_shape(self) -> tuple[int, ...]:
        """Actual size of uniform grid."""
        return tuple(self.size_c - self.zerobc_c)

    def __repr__(self):
        return Domain.__repr__(self).replace(
            'Domain(',
            f'UGDesc(size={self.size_c.tolist()}, ')

    def _short_string(self, global_shape):
        return f'uniform wave function grid shape: {global_shape}'

    @cached_property
    def phase_factor_cd(self):
        """Phase factor for block-boundary conditions."""
        delta_d = np.array([-1, 1])
        disp_cd = np.empty((3, 2))
        for pos, pbc, size, disp_d in zips(self.mypos_c, self.pbc_c,
                                           self.parsize_c, disp_cd):
            disp_d[:] = -((pos + delta_d) // size)
        return np.exp(2j * np.pi *
                      disp_cd *
                      self.kpt_c[:, np.newaxis])

    def new(self,
            *,
            kpt=None,
            dtype=None,
            comm: MPIComm | Literal['inherit'] | None = 'inherit',
            size=None,
            pbc=None,
            zerobc=None,
            decomp=None) -> UGDesc:
        """Create new uniform grid description."""
        reuse_decomp = (decomp is None and comm == 'inherit' and
                        size is None and pbc is None and zerobc is None)
        if reuse_decomp:
            decomp = self.decomp_cp
        comm = self.comm if comm == 'inherit' else comm
        return UGDesc(cell=self.cell_cv,
                      size=self.size_c if size is None else size,
                      pbc=self.pbc_c if pbc is None else pbc,
                      zerobc=self.zerobc_c if zerobc is None else zerobc,
                      kpt=(self.kpt_c if self.kpt_c.any() else None)
                      if kpt is None else kpt,
                      comm=comm or serial_comm,
                      decomp=decomp,
                      dtype=self.dtype if dtype is None else dtype)

    def empty(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm,
              xp=np) -> UGArray:
        """Create new UGArray object.

        parameters
        ----------
        dims:
            Extra dimensions.
        comm:
            Distribute dimensions along this communicator.
        """
        return UGArray(self, dims, comm, xp=xp)

    def from_data(self, data: np.ndarray) -> UGArray:
        return UGArray(self, data.shape[:-3], data=data)

    def blocks(self, data: np.ndarray):
        """Yield views of blocks of data."""
        s0, s1, s2 = self.parsize_c
        d0_p, d1_p, d2_p = (d_p - d_p[0] for d_p in self.decomp_cp)
        for p0 in range(s0):
            b0, e0 = d0_p[p0:p0 + 2]
            for p1 in range(s1):
                b1, e1 = d1_p[p1:p1 + 2]
                for p2 in range(s2):
                    b2, e2 = d2_p[p2:p2 + 2]
                    yield data[..., b0:e0, b1:e1, b2:e2]

    def xyz(self) -> Array4D:
        """Create array of (x, y, z) coordinates."""
        indices_Rc = np.indices(self.mysize_c).transpose((1, 2, 3, 0))
        indices_Rc += self.start_c
        return indices_Rc @ (self.cell_cv.T / self.size_c).T

    def atom_centered_functions(self,
                                functions,
                                positions,
                                *,
                                qspiral_v=None,
                                atomdist=None,
                                integrals=None,
                                cut=False,
                                xp=None):
        """Create UGAtomCenteredFunctions object."""
        assert qspiral_v is None
        return UGAtomCenteredFunctions(functions,
                                       positions,
                                       self,
                                       atomdist=atomdist,
                                       integrals=integrals,
                                       cut=cut,
                                       xp=xp)

    def transformer(self, other: UGDesc, stencil_range=3, xp=np):
        """Create transformer from one grid to another.

        (for interpolation and restriction).
        """
        from gpaw.transformers import Transformer

        apply = Transformer(self._gd, other._gd, nn=stencil_range, xp=xp).apply

        def transform(functions, out=None):
            if out is None:
                out = other.empty(functions.dims, functions.comm, xp=xp)
            for input, output in zips(functions._arrays(), out._arrays()):
                apply(input, output)
            return out

        return transform

    def eikr(self, kpt_c: Vector | None = None) -> Array3D:
        """Plane wave.

        :::
               _ _
              ik.r
             e

        Parameters
        ----------
        kpt_c:
            k-point in units of the reciprocal cell.  Defaults to the
            UGDesc objects own k-point.
        """
        if kpt_c is None:
            kpt_c = self.kpt_c
        index_Rc = np.indices(self.mysize_c).T + self.start_c
        return np.exp(2j * pi * (index_Rc @ (kpt_c / self.size_c))).T

    @property
    def _gd(self):
        # Make sure gd can be pickled (in serial):
        comm = self.comm if self.comm.size > 1 else serial_comm

        return GridDescriptor(self.size_c,
                              cell_cv=self.cell_cv,
                              pbc_c=~self.zerobc_c,
                              comm=comm,
                              parsize_c=[len(d_p) - 1
                                         for d_p in self.decomp_cp])

    @classmethod
    def from_cell_and_grid_spacing(cls,
                                   cell: ArrayLike1D | ArrayLike2D,
                                   grid_spacing: float,
                                   pbc=(True, True, True),
                                   kpt: Vector | None = None,
                                   comm: MPIComm = serial_comm,
                                   dtype=None) -> UGDesc:
        """Create UGDesc from grid-spacing."""
        domain: Domain = Domain(cell, pbc, kpt, comm, dtype)
        return domain.uniform_grid_with_grid_spacing(grid_spacing)

    def fft_plans(self,
                  flags: int = fftw.MEASURE,
                  xp=np,
                  dtype=None) -> fftw.FFTPlans:
        """Create FFTW-plans."""
        if dtype is None:
            dtype = self.dtype
        if self.comm.rank == 0:
            return fftw.create_plans(self.size_c, dtype, flags, xp)
        else:
            return fftw.create_plans([0, 0, 0], dtype)

    def ranks_from_fractional_positions(self,
                                        relpos_ac: Array2D) -> Array1D:
        rank_ac = np.floor(relpos_ac * self.parsize_c).astype(int)
        if (rank_ac < 0).any() or (rank_ac >= self.parsize_c).any():
            raise ValueError('Positions outside cell!')
        return np.ravel_multi_index(rank_ac.T, self.parsize_c)  # type: ignore

    def ekin_max(self) -> float:
        """Maximum value of ekin so that all 0.5 * G^2 < ekin.

        In 1D, this will be 0.5*(pi/h)^2 where h is the grid-spacing.
        """
        # Height of reciprocal cell (squared):
        b2_c = np.pi**2 / (self.cell_cv**2).sum(1)
        return 0.5 * (self.size_c**2 * b2_c).min()


class UGArray(DistributedArrays[UGDesc]):
    def __init__(self,
                 grid: UGDesc,
                 dims: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray | None = None,
                 xp=None):
        """Object for storing function(s) on a uniform grid.

        parameters
        ----------
        grid:
            Description of uniform grid.
        dims:
            Extra dimensions.
        comm:
            Distribute dimensions along this communicator.
        data:
            Data array for storage.
        """
        DistributedArrays. __init__(self, dims, grid.myshape,
                                    comm, grid.comm, data, grid.dv,
                                    grid.dtype, xp)
        self.desc = grid

    def __repr__(self):
        txt = f'UGArray(grid={self.desc}, dims={self.dims}'
        if self.comm.size > 1:
            txt += f', comm={self.comm.rank}/{self.comm.size}'
        if self.xp is not np:
            txt += ', xp=cp'
        return txt + ')'

    def new(self, data=None, zeroed=False, dims=None):
        """Create new UniforGridFunctions object of same kind.

        Parameters
        ----------
        data:
            Array to use for storage.
        zeroed:
            If True, set data to zero.
        dims:
            Extra dimensions (bands, spin, etc.), required if
            data does not fit the full array.
        """
        if dims:
            assert data is not None
        else:
            dims = self.dims
        if data is None:
            data = self.xp.empty_like(self.data)

        f_xR = UGArray(self.desc, dims, self.comm, data)
        if zeroed:
            f_xR.data[:] = 0.0
        return f_xR

    def __getitem__(self, index):
        data = self.data[index]
        return UGArray(data=data,
                       dims=data.shape[:-3],
                       grid=self.desc)

    def __imul__(self,
                 other: float | np.ndarray | UGArray
                 ) -> UGArray:
        if isinstance(other, float):
            self.data *= other
            return self
        if isinstance(other, UGArray):
            other = other.data
        assert other.shape[-3:] == self.data.shape[-3:]
        self.data *= other
        return self

    def __mul__(self,
                other: float | np.ndarray | UGArray
                ) -> UGArray:
        result = self.new(data=self.data.copy())
        result *= other
        return result

    def _arrays(self):
        return self.data.reshape((-1,) + self.data.shape[-3:])

    def xy(self, *axes: int | None) -> tuple[Array1D, Array1D]:
        """Extract x, y values along line.

        Useful for plotting::

          x, y = grid.xy(0, ..., 0)
          plt.plot(x, y)
        """
        assert len(axes) == 3 + len(self.dims)
        index = tuple([slice(0, None) if axis is None else axis
                       for axis in axes])
        y = self.data[index]  # type: ignore
        c = axes[-3:].index(...)
        grid = self.desc
        dx = (grid.cell_cv[c]**2).sum()**0.5 / grid.size_c[c]
        x = np.arange(grid.start_c[c], grid.end_c[c]) * dx
        return x, as_np(y)

    def to_complex(self) -> UGArray:
        """Return a copy with dtype=complex."""
        c = self.desc.new(dtype=complex).empty()
        c.data[:] = self.data
        return c

    def scatter_from(self, data: np.ndarray | UGArray | None = None) -> None:
        """Scatter data from rank-0 to all ranks."""
        if isinstance(data, UGArray):
            data = data.data

        comm = self.desc.comm
        if comm.size == 1:
            self.data[:] = data
            return

        if comm.rank != 0:
            comm.receive(self.data, 0, 42)
            return

        requests = []
        assert isinstance(data, self.xp.ndarray)
        for rank, block in enumerate(self.desc.blocks(data)):
            if rank != 0:
                block = block.copy()
                request = comm.send(block, rank, 42, False)
                # Remember to store a reference to the
                # send buffer (block) so that is isn't
                # deallocated:
                requests.append((request, block))
            else:
                self.data[:] = block

        for request, _ in requests:
            comm.wait(request)

    def gather(self, out=None, broadcast=False):
        """Gather data from all ranks to rank-0."""
        assert out is None
        comm = self.desc.comm
        if comm.size == 1:
            return self

        if broadcast or comm.rank == 0:
            grid = self.desc.new(comm=serial_comm)
            out = grid.empty(self.dims, comm=self.comm, xp=self.xp)

        if comm.rank != 0:
            # There can be several sends before the corresponding receives
            # are posted, so use synchronous send here
            comm.ssend(self.data, 0, 301)
            if broadcast:
                comm.broadcast(out.data, 0)
                return out
            return

        # Put the subdomains from the slaves into the big array
        # for the whole domain:
        for rank, block in enumerate(self.desc.blocks(out.data)):
            if rank != 0:
                buf = self.xp.empty_like(block)
                comm.receive(buf, rank, 301)
                block[:] = buf
            else:
                block[:] = self.data

        if broadcast:
            comm.broadcast(out.data, 0)

        return out

    def fft(self, plan=None, pw=None, out=None):
        r"""Do FFT.

        Returns:
            PWArray with values
        :::
                          _ _
           _    1  / _  -iG.r   _
         C(G) = -- |dr e      f(r),
                Ω  /

        where `C(\bG)` are the plane wave coefficients and Ω is the cell
        volume.

        Parameters
        ----------
        plan:
            Plan for FFT.
        pw:
            Target PW description.
        out:
            Target PWArray object.
        """
        assert not self.desc.zerobc_c.any()
        if out is None:
            assert pw is not None
            out = pw.empty(dims=self.dims, xp=self.xp)
        if pw is None:
            pw = out.desc
        if pw.dtype != self.desc.dtype:
            raise TypeError(
                f'Type mismatch: {self.desc.dtype} -> {pw.dtype}')
        input = self
        if self.desc.comm.size > 1:
            input = input.gather()
        if self.desc.comm.rank == 0:
            plan = plan or self.desc.fft_plans(xp=self.xp)
            for i, o in zip(input.flat(), out.flat()):
                coefs = plan.fft_sphere(i.data, pw)
                o.scatter_from(coefs)
        else:
            for o in out.flat():
                o.scatter_from(None)

        return out

    def norm2(self):
        """Calculate integral over cell of absolute value squared.

        :::

         /   _  2 _
         ||a(r)| dr
         /
        """
        norm_x = []
        arrays_xR = self._arrays()
        for a_R in arrays_xR:
            norm_x.append(self.xp.vdot(a_R, a_R).real * self.desc.dv)
        result = self.xp.array(norm_x).reshape(self.mydims)
        self.desc.comm.sum(result)
        return result

    def integrate(self, other=None, skip_sum=False):
        """Integral of self or self times cc(other)."""
        if other is not None:
            assert self.desc.dtype == other.desc.dtype
            a_xR = self._arrays()
            b_yR = other._arrays()
            a_xR = a_xR.reshape((len(a_xR), -1))
            b_yR = b_yR.reshape((len(b_yR), -1))
            result = (a_xR @ b_yR.T.conj()).reshape(self.dims + other.dims)
        else:
            # Make sure we have an array and not a scalar!
            result = self.xp.asarray(self.data.sum(axis=(-3, -2, -1)))

        if not skip_sum:
            self.desc.comm.sum(result)
        if result.ndim == 0:
            result = result.item()  # convert to scalar
        return result * self.desc.dv

    def to_pbc_grid(self):
        """Convert to UniformGrid with ``pbc=(True, True, True)``."""
        if not self.desc.zerobc_c.any():
            return self
        grid = self.desc.new(zerobc=False)
        new = grid.empty(self.dims)
        new.data[:] = 0.0
        *_, i, j, k = self.data.shape
        new.data[..., -i:, -j:, -k:] = self.data
        return new

    def multiply_by_eikr(self, kpt_c: Vector | None = None) -> None:
        """Multiply by `exp(ik.r)`."""
        if kpt_c is None:
            kpt_c = self.desc.kpt_c
        else:
            kpt_c = np.asarray(kpt_c)
        if kpt_c.any():
            self.data *= self.desc.eikr(kpt_c)

    def interpolate(self,
                    plan1: fftw.FFTPlans | None = None,
                    plan2: fftw.FFTPlans | None = None,
                    grid: UGDesc | None = None,
                    out: UGArray | None = None) -> UGArray:
        """Interpolate to finer grid.

        Parameters
        ----------
        plan1:
            Plan for FFT (course grid).
        plan2:
            Plan for inverse FFT (fine grid).
        grid:
            Target grid.
        out:
            Target UGArray object.
        """
        if out is None:
            if grid is None:
                raise ValueError('Please specify "grid" or "out".')
            out = grid.empty(self.dims, xp=self.xp)

        if out.desc.zerobc_c.any() or self.desc.zerobc_c.any():
            raise ValueError('Grids must have zerobc=False!')

        if self.desc.comm.size > 1:
            input = self.gather()
            if input is not None:
                output = input.interpolate(plan1, plan2,
                                           out.desc.new(comm=None))
                out.scatter_from(output.data)
            else:
                out.scatter_from()
            return out

        size1_c = self.desc.size_c
        size2_c = out.desc.size_c
        if (size2_c <= size1_c).any():
            raise ValueError('Too few points in target grid!')

        plan1 = plan1 or self.desc.fft_plans(xp=self.xp)
        plan2 = plan2 or out.desc.fft_plans(xp=self.xp)

        if self.dims:
            for input, output in zips(self.flat(), out.flat()):
                input.interpolate(plan1, plan2, grid, output)
            return out

        plan1.tmp_R[:] = self.data
        kpt_c = self.desc.kpt_c
        if kpt_c.any():
            plan1.tmp_R *= self.desc.eikr(-kpt_c)
        plan1.fft()

        a_Q = plan1.tmp_Q
        b_Q = plan2.tmp_Q

        e0, e1, e2 = 1 - size1_c % 2  # even or odd size
        a0, a1, a2 = size2_c // 2 - size1_c // 2
        b0, b1, b2 = size1_c + (a0, a1, a2)

        if self.desc.dtype == float:
            b2 = (b2 - a2) // 2 + 1
            a2 = 0
            axes = [0, 1]
        else:
            axes = [0, 1, 2]

        b_Q[:] = 0.0
        b_Q[a0:b0, a1:b1, a2:b2] = self.xp.fft.fftshift(a_Q, axes=axes)

        if e0:
            b_Q[a0, a1:b1, a2:b2] *= 0.5
            b_Q[b0, a1:b1, a2:b2] = b_Q[a0, a1:b1, a2:b2]
            b0 += 1
        if e1:
            b_Q[a0:b0, a1, a2:b2] *= 0.5
            b_Q[a0:b0, b1, a2:b2] = b_Q[a0:b0, a1, a2:b2]
            b1 += 1
        if self.desc.dtype == complex:
            if e2:
                b_Q[a0:b0, a1:b1, a2] *= 0.5
                b_Q[a0:b0, a1:b1, b2] = b_Q[a0:b0, a1:b1, a2]
        else:
            if e2:
                b_Q[a0:b0, a1:b1, b2 - 1] *= 0.5

        b_Q[:] = self.xp.fft.ifftshift(b_Q, axes=axes)
        plan2.ifft()
        out.data[:] = plan2.tmp_R
        out.data *= (1.0 / self.data.size)
        out.multiply_by_eikr()
        return out

    def fft_restrict(self,
                     plan1: fftw.FFTPlans | None = None,
                     plan2: fftw.FFTPlans | None = None,
                     grid: UGDesc | None = None,
                     out: UGArray | None = None) -> UGArray:
        """Restrict to coarser grid.

        Parameters
        ----------
        plan1:
            Plan for FFT.
        plan2:
            Plan for inverse FFT.
        grid:
            Target grid.
        out:
            Target UGArray object.
        """
        if out is None:
            if grid is None:
                raise ValueError('Please specify "grid" or "out".')
            out = grid.empty(self.dims, xp=self.xp)

        if out.desc.zerobc_c.any() or self.desc.zerobc_c.any():
            raise ValueError('Grids must have zerobc=False!')

        if self.desc.comm.size > 1:
            input = self.gather()
            if input is not None:
                output = input.fft_restrict(plan1, plan2,
                                            out.desc.new(comm=None))
                out.scatter_from(output.data)
            else:
                out.scatter_from()
            return out

        size1_c = self.desc.size_c
        size2_c = out.desc.size_c

        plan1 = plan1 or self.desc.fft_plans()
        plan2 = plan2 or out.desc.fft_plans()

        if self.dims:
            for input, output in zips(self.flat(), out.flat()):
                input.fft_restrict(plan1, plan2, grid, output)
            return out

        plan1.tmp_R[:] = self.data
        a_Q = plan2.tmp_Q
        b_Q = plan1.tmp_Q

        e0, e1, e2 = 1 - size2_c % 2  # even or odd size
        a0, a1, a2 = size1_c // 2 - size2_c // 2
        b0, b1, b2 = size2_c // 2 + size1_c // 2 + 1

        if self.desc.dtype == float:
            b2 = size2_c[2] // 2 + 1
            a2 = 0
            axes = [0, 1]
        else:
            axes = [0, 1, 2]

        plan1.fft()
        b_Q[:] = self.xp.fft.fftshift(b_Q, axes=axes)

        if e0:
            b_Q[a0, a1:b1, a2:b2] += b_Q[b0 - 1, a1:b1, a2:b2]
            b_Q[a0, a1:b1, a2:b2] *= 0.5
            b0 -= 1
        if e1:
            b_Q[a0:b0, a1, a2:b2] += b_Q[a0:b0, b1 - 1, a2:b2]
            b_Q[a0:b0, a1, a2:b2] *= 0.5
            b1 -= 1
        if self.desc.dtype == complex and e2:
            b_Q[a0:b0, a1:b1, a2] += b_Q[a0:b0, a1:b1, b2 - 1]
            b_Q[a0:b0, a1:b1, a2] *= 0.5
            b2 -= 1

        a_Q[:] = b_Q[a0:b0, a1:b1, a2:b2]
        a_Q[:] = self.xp.fft.ifftshift(a_Q, axes=axes)
        plan2.ifft()
        out.data[:] = plan2.tmp_R
        out.data *= (1.0 / self.data.size)
        return out

    def abs_square(self,
                   weights: Array1D,
                   out: UGArray | None = None) -> None:
        """Add weighted absolute square of data to output array."""
        assert out is not None

        if self.xp is np:
            for f, psit_R in zips(weights, self.data):
                add_to_density(f, psit_R, out.data)
        elif cupy_is_fake:
            for f, psit_R in zips(weights, self.data):
                add_to_density(f, psit_R._data, out.data._data)  # type: ignore
        else:
            add_to_density_gpu(self.xp.asarray(weights), self.data, out.data)

    def symmetrize(self, rotation_scc, translation_sc):
        """Make data symmetric."""
        if len(rotation_scc) == 1:
            return

        a_xR = self.gather()

        if a_xR is None:
            b_xR = None
        else:
            if self.xp is not np:
                a_xR = a_xR.to_xp(np)
            b_xR = a_xR.new()
            t_sc = (translation_sc * self.desc.size_c).round().astype(int)
            offset_c = np.array(self.desc.zerobc_c, dtype=int)
            for a_R, b_R in zips(a_xR._arrays(), b_xR._arrays()):
                b_R[:] = 0.0
                for r_cc, t_c in zips(rotation_scc, t_sc):
                    symmetrize_ft(a_R, b_R, r_cc, t_c, offset_c)
            if self.xp is not np:
                b_xR = b_xR.to_xp(self.xp)
        self.scatter_from(b_xR)

        self.data *= 1.0 / len(rotation_scc)

    def randomize(self, seed: int | None = None) -> None:
        """Insert random numbers between -0.5 and 0.5 into data."""
        if seed is None:
            seed = self.comm.rank + self.desc.comm.rank * self.comm.size
        rng = self.xp.random.default_rng(seed)
        a = self.data.view(float)
        rng.random(a.shape, out=a)
        a -= 0.5

    def moment(self):
        """Calculate moment of data."""
        assert self.dims == ()
        ug = self.desc

        index_cr = [np.arange(ug.start_c[c], ug.end_c[c], dtype=float)
                    for c in range(3)]
        for index_r, size in zip(index_cr, ug.size_c):
            if index_r[0] == 0:
                # We have periodic bc's, so index 0 is the same as index
                # size (= last + 1).  Include both points with 0.5 weight:
                index_r[0] = 0.5 * size

        rho_ijk = self.data
        rho_ij = rho_ijk.sum(axis=2)
        rho_ik = rho_ijk.sum(axis=1)
        rho_cr = [rho_ij.sum(axis=1), rho_ij.sum(axis=0), rho_ik.sum(axis=0)]
        if self.xp is not np:
            rho_cr = [rho_r.get() for rho_r in rho_cr]

        d_c = [index_r @ rho_r for index_r, rho_r in zips(index_cr, rho_cr)]
        d_v = (d_c / ug.size_c) @ ug.cell_cv * self.dv
        self.desc.comm.sum(d_v)
        return d_v

    def scaled(self, cell: float, values: float = 1.0) -> UGArray:
        """Create new scaled UGArray object.

        Unit cell axes are multiplied by `cell` and data by `values`.
        """
        grid = self.desc
        grid = UGDesc(cell=grid.cell_cv * cell,
                      size=grid.size_c,
                      pbc=grid.pbc_c,
                      zerobc=grid.zerobc_c,
                      kpt=(grid.kpt_c if grid.kpt_c.any() else None),
                      dtype=grid.dtype,
                      comm=grid.comm)
        return UGArray(grid, self.dims, self.comm, self.data * values)

    def add_ked(self,
                occ_n: Array1D,
                taut_R: UGArray) -> None:
        grad_v = [
            Gradient(self.desc._gd, v, n=3, dtype=self.desc.dtype)
            for v in range(3)]
        tmp_R = self.desc.empty()
        for f, psit_R in zips(occ_n, self):
            for grad in grad_v:
                grad(psit_R, tmp_R)
                add_to_density(0.5 * f, tmp_R.data, taut_R.data)

    def redist(self,
               domain: UGDesc,
               comm1: MPIComm, comm2: MPIComm) -> UGArray:
        a = super().redist(domain, comm1, comm2)
        assert isinstance(a, UGArray)
        return a

    def isosurface(self, show=True, **kwargs) -> go.Isosurface:
        import plotly.graph_objects as go
        values = self.data
        assert values.ndim == 3
        if values.dtype == complex:
            values = abs(values)
        x, y, z = (c.T.flatten() for c in self.desc.xyz().T)
        vmin = values.min()
        vmax = values.max()
        kwargs = {
            'isomin': vmin + (vmax - vmin) * 0.1,
            'isomax': vmax - (vmax - vmin) * 0.1,
            'caps': dict(x_show=False,
                         y_show=False,
                         z_show=False),
            **kwargs}
        surf = go.Isosurface(x=x, y=y, z=z, value=values.flatten(),
                             **kwargs)
        if show:
            go.Figure(data=[surf]).show()
        return surf
