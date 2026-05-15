from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np

from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_centered_functions import AtomArrays
from gpaw.mpi import broadcast_exception
from gpaw.new import trace, zips
from gpaw.new.c import calculate_residuals_gpu
from gpaw.new.eigensolver import Eigensolver, calculate_weights
from gpaw.new.energies import DFTEnergies
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.utilities.blas import axpy
from gpaw.utilities import as_real_dtype
from gpaw.new.ibzwfs import IBZWaveFunctions


class PWFDEigensolver(Eigensolver):
    def __init__(self,
                 hamiltonian,
                 converge_bands: int | str = 'occupied',
                 blocksize: int = 10,
                 max_buffer_mem: int | None = 200 * 1024 ** 2):
        self.converge_bands = converge_bands
        self.blocksize = blocksize
        self.preconditioner: Callable
        self.preconditioner_factory = hamiltonian.create_preconditioner
        self.work_arrays: np.ndarray
        self.data_buffers: np.ndarray

        # Maximal memory to be used for the eigensolver
        # should be infinite if hamiltonian is not band-local (hybrids)
        self.max_buffer_mem = (
            max_buffer_mem if hamiltonian.band_local else None)

    def _initialize(self, ibzwfs):
        # First time: allocate work-arrays
        self.preconditioner = self.preconditioner_factory(self.blocksize,
                                                          xp=ibzwfs.xp)

    def _allocate_buffer_arrays(self, ibzwfs, shape):
        G_max = np.prod(ibzwfs.get_max_shape())
        b = max(wfs.n2 - wfs.n1 for wfs in ibzwfs)
        nbands = ibzwfs.nbands
        dtype_size = ibzwfs.wfs_qs[0][0].psit_nX.data.dtype.itemsize
        domain_size = ibzwfs.domain_comm.size

        if self.max_buffer_mem is not None:
            # Buffer size needs to ensure that the number of bands
            # of the buffer is a multiple of domain_size.
            buffer_size_per_domain = max(self.max_buffer_mem,
                                         domain_size * G_max * dtype_size,
                                         nbands * dtype_size) \
                // (domain_size * G_max * dtype_size)
            buffer_size = min(buffer_size_per_domain * domain_size
                              * G_max * dtype_size,
                              b * G_max * dtype_size)
        else:
            buffer_size = max(b * G_max * dtype_size,
                              nbands * dtype_size)

        self.data_buffers = ibzwfs.xp.empty(shape + (buffer_size,),
                                            np.byte)

    def _allocate_work_arrays(self, ibzwfs, shape):
        b = max(wfs.n2 - wfs.n1 for wfs in ibzwfs)
        shape += (b,) + ibzwfs.get_max_shape()
        dtype = ibzwfs.wfs_qs[0][0].psit_nX.data.dtype
        self.work_arrays = ibzwfs.xp.empty(shape, dtype)

    @trace
    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density,
                potential,
                hamiltonian: Hamiltonian,
                pot_calc,
                energies: DFTEnergies) -> tuple[float, float, DFTEnergies]:
        """Iterate on state given fixed hamiltonian.

        Returns
        -------
        float:
            Weighted error of residuals:::

                           ~
              R = (H - ε S)ψ
               n        n   n
        """

        if not hasattr(self, 'preconditioner'):
            self._initialize(ibzwfs)

        wfs = ibzwfs.wfs_qs[0][0]
        dS_aii = wfs.setups.get_overlap_corrections(wfs.P_ani.layout.atomdist,
                                                    wfs.xp)

        ibzwfs.orthonormalize()
        hamiltonian.update_wave_functions(ibzwfs)

        apply = partial(hamiltonian.apply,
                        potential.vt_sR,
                        potential.dedtaut_sR,
                        ibzwfs, density.D_asii)  # used by hybrids

        weight_un = calculate_weights(self.converge_bands, ibzwfs)

        wfs_error = 0.0
        eig_error = 0.0
        # Loop over k-points:
        with broadcast_exception(ibzwfs.kpt_comm):
            for wfs, weight_n in zips(ibzwfs, weight_un):
                dH = partial(potential.dH, spin=wfs.spin)
                Ht = partial(apply, spin=wfs.spin)
                temp_wfs_error, temp_eig_error = \
                    self.iterate_kpt(wfs, weight_n, self.iterate1,
                                     Ht=Ht, dH=dH, dS_aii=dS_aii)
                wfs_error += wfs.weight * temp_wfs_error
                if eig_error < temp_eig_error:
                    eig_error = temp_eig_error

        wfs_error = ibzwfs.kpt_band_comm.sum_scalar(
            float(wfs_error)) * ibzwfs.spin_degeneracy
        eig_error = ibzwfs.kpt_band_comm.max_scalar(eig_error)

        return eig_error, wfs_error, energies

    def iterate1(self, wfs, Ht, dH, dS_aii, weight_n):
        raise NotImplementedError


@trace
def calculate_residuals(psit_nX,
                        residual_nX: XArray,
                        pt_aiX,
                        P_ani,
                        eig_n,
                        dH: Callable[[AtomArrays, AtomArrays], AtomArrays],
                        dS_aii: AtomArrays,
                        P1_ani: AtomArrays,
                        P2_ani: AtomArrays) -> None:
    """Complete the calculation of resuduals.

    Starting from residual_nX having the values:::

       ^   ~  ~
      (T + v) ψ
               n

    add the following:::

      --   a       a   ~a ~   ~a    ~
      > (ΔH  - ε ΔS  )<p |ψ > p - ε ψ .
      --   ij   n  ij   j  n   i     n
      ij

    (P1_ani and P2_ani are work buffers).
    """
    xp = residual_nX.xp
    if xp is np:
        for r, e, p in zips(residual_nX.data, eig_n, psit_nX.data):
            axpy(-e, p, r)
    else:
        eig_n = xp.asarray(eig_n, dtype=as_real_dtype(residual_nX.data.dtype))
        calculate_residuals_gpu(residual_nX.data, eig_n, psit_nX.data)

    dH(P_ani, P1_ani)
    P_ani.block_diag_multiply(dS_aii, out_ani=P2_ani)

    if P_ani.data.ndim == 2:
        subscripts = 'nI, n -> nI'
    else:
        subscripts = 'nsI, n -> nsI'
    if xp is np:
        np.einsum(subscripts, P2_ani.data, eig_n, out=P2_ani.data,
                  dtype=P2_ani.data.dtype, casting='same_kind')
    else:
        P2_ani.data[:] = xp.einsum(subscripts, P2_ani.data, eig_n)
    P1_ani.data -= P2_ani.data
    pt_aiX.add_to(residual_nX, P1_ani)
