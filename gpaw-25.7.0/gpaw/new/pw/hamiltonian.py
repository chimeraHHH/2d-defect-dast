from __future__ import annotations
from typing import Callable

import numpy as np

from gpaw.core.plane_waves import PWArray
from gpaw.core.uniform_grid import UGArray
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.gpu import cupy as cp
from gpaw.new import trace, zips
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.c import pw_precond, pw_insert_gpu
from gpaw.utilities import as_complex_dtype


class PWHamiltonian(Hamiltonian):
    def __init__(self, grid, pw, xp=np):
        self.grid_local = grid.new(comm=None, dtype=pw.dtype)
        self.plan = self.grid_local.fft_plans(xp=xp)
        # It's a bit too expensive to create all the local PW-descriptors
        # for all the k-points every time we apply the Hamiltonian, so we
        # cache them:
        self.pw_cache = {}

    @trace
    def apply_local_potential(self,
                              vt_R: UGArray,
                              psit_nG: XArray,
                              out: XArray) -> None:
        assert isinstance(psit_nG, PWArray)
        assert isinstance(out, PWArray)
        out_nG = out
        xp = psit_nG.xp
        pw = psit_nG.desc
        if xp is not np and pw.comm.size == 1:
            return apply_local_potential_gpu(vt_R, psit_nG, out_nG)
        vt_R = vt_R.gather(broadcast=True)
        tmp_R = self.grid_local.empty(xp=xp)
        if pw.comm.size == 1:
            pw_local = pw
        else:
            key = tuple(pw.kpt_c)
            pw_local = self.pw_cache.get(key)
            if pw_local is None:
                pw_local = pw.new(comm=None)
                self.pw_cache[key] = pw_local
        psit_G = pw_local.empty(xp=xp)
        e_kin_G = xp.asarray(psit_G.desc.ekin_G)
        domain_comm = psit_nG.desc.comm
        mynbands = psit_nG.mydims[0]
        vtpsit_G = pw_local.empty(xp=xp)

        for n1 in range(0, mynbands, domain_comm.size):
            n2 = min(n1 + domain_comm.size, mynbands)
            psit_nG[n1:n2].gather_all(psit_G)
            if domain_comm.rank < n2 - n1:
                psit_G.ifft(out=tmp_R, plan=self.plan)
                tmp_R.data *= vt_R.data
                tmp_R.fft(out=vtpsit_G, plan=self.plan)
                psit_G.data *= e_kin_G
                vtpsit_G.data += psit_G.data
            out_nG[n1:n2].scatter_from_all(vtpsit_G)

    def apply_mgga(self,
                   dedtaut_R: UGArray,
                   psit_nG: XArray,
                   vt_nG: XArray) -> None:
        pw = psit_nG.desc
        dpsit_R = dedtaut_R.desc.new(dtype=pw.dtype).empty()
        Gplusk1_Gv = pw.reciprocal_vectors()
        tmp_G = pw.empty()

        for psit_G, vt_G in zips(psit_nG, vt_nG):
            for v in range(3):
                tmp_G.data[:] = psit_G.data
                tmp_G.data *= 1j * Gplusk1_Gv[:, v]
                tmp_G.ifft(out=dpsit_R)
                dpsit_R.data *= dedtaut_R.data
                dpsit_R.fft(out=tmp_G)
                vt_G.data -= 0.5j * Gplusk1_Gv[:, v] * tmp_G.data

    def create_preconditioner(self,
                              blocksize: int,
                              xp=np
                              ) -> Callable[[PWArray,
                                             PWArray,
                                             PWArray], None]:
        return precondition

    def calculate_kinetic_energy(self, wfs, skip_sum=False):
        e_kin = 0.0
        for f, psit_G in zip(wfs.myocc_n, wfs.psit_nX):
            if f > 1.0e-10:
                e_kin += f * psit_G.norm2('kinetic', skip_sum=skip_sum)
        if not skip_sum:
            e_kin = psit_G.desc.comm.sum_scalar(e_kin)
            e_kin = wfs.band_comm.sum_scalar(e_kin)
        return e_kin * wfs.spin_degeneracy


@trace
def precondition(psit_nG: PWArray,
                 residual_nG: PWArray,
                 out: PWArray,
                 ekin_n=None) -> None:
    """Preconditioner for KS equation.

    From:

      Teter, Payne and Allen, Phys. Rev. B 40, 12255 (1989)

    as modified by:

      Kresse and Furthmüller, Phys. Rev. B 54, 11169 (1996)
    """
    xp = psit_nG.xp
    G2_G = xp.asarray(psit_nG.desc.ekin_G * 2)
    if ekin_n is None:
        ekin_n = psit_nG.norm2('kinetic')

    if xp is np:
        for r_G, o_G, ekin in zips(residual_nG.data,
                                   out.data,
                                   ekin_n):
            pw_precond(G2_G, r_G, ekin, o_G)
    else:
        out.data[:] = gpu_prec(ekin_n[:, np.newaxis],
                               G2_G[np.newaxis],
                               residual_nG.data)
    return ekin_n


@trace(gpu=True)
@cp.fuse()
def gpu_prec(ekin, G2, residual):
    x = 1 / ekin / 3 * G2
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    return -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * residual


def spinor_precondition(psit_nsG, residual_nsG, out):
    G2_G = psit_nsG.desc.ekin_G * 2
    for r_sG, o_sG, ekin in zips(residual_nsG.data,
                                 out.data,
                                 psit_nsG.norm2('kinetic').sum(1)):
        for r_G, o_G in zips(r_sG, o_sG):
            pw_precond(G2_G, r_G, ekin, o_G)


class SpinorPWHamiltonian(Hamiltonian):
    def __init__(self, qspiral_v):
        super().__init__()
        self.qspiral_v = qspiral_v

    def apply(self,
              vt_xR: UGArray,
              dedtaut_xR: UGArray | None,
              ibzwfs,
              D_asii,
              psit_nsG: XArray,
              out: XArray,
              spin: int,
              calculate_energy: bool = False) -> XArray:
        assert dedtaut_xR is None
        out_nsG = out
        pw = psit_nsG.desc

        if self.qspiral_v is None:
            np.multiply(pw.ekin_G, psit_nsG.data, out_nsG.data)
        else:
            for s, sign in enumerate([1, -1]):
                ekin_G = 0.5 * ((pw.G_plus_k_Gv +
                                 0.5 * sign * self.qspiral_v)**2).sum(1)
                np.multiply(ekin_G, psit_nsG.data[:, s], out_nsG.data[:, s])

        grid = vt_xR.desc.new(dtype=complex)

        v, x, y, z = vt_xR.data
        iy = y * 1j

        f_sR = grid.empty(2)
        g_R = grid.empty()

        for p_sG, o_sG in zips(psit_nsG, out_nsG):
            p_sG.ifft(out=f_sR)
            a, b = f_sR.data
            g_R.data = a * (v + z) + b * (x - iy)
            o_sG.data[0] += g_R.fft(pw=pw).data
            g_R.data = a * (x + iy) + b * (v - z)
            o_sG.data[1] += g_R.fft(pw=pw).data

        return out_nsG

    def create_preconditioner(self, blocksize, xp):
        return spinor_precondition


@trace
def apply_local_potential_gpu(vt_R,
                              psit_nG,
                              out_nG,
                              blocksize=10):
    from gpaw.gpu import cupyx
    pw = psit_nG.desc
    e_kin_G = cp.asarray(pw.ekin_G)
    mynbands = psit_nG.mydims[0]
    size_c = vt_R.desc.size_c
    w = trace(gpu=True)
    if np.issubdtype(pw.dtype, np.floating):
        shape = (size_c[0], size_c[1], size_c[2] // 2 + 1)
        ifftn = w(cupyx.scipy.fft.irfftn)
        fftn = w(cupyx.scipy.fft.rfftn)
    else:
        shape = tuple(size_c)
        ifftn = w(cupyx.scipy.fft.ifftn)
        fftn = w(cupyx.scipy.fft.fftn)
    Q_G = cp.asarray(pw.indices(shape))
    psit_bQ = None
    for b1 in range(0, mynbands, blocksize):
        b2 = min(b1 + blocksize, mynbands)
        nb = b2 - b1
        if psit_bQ is None:
            psit_bQ = cp.empty((nb,) + shape, as_complex_dtype(pw.dtype))
        elif nb < blocksize:
            psit_bQ = psit_bQ[:nb]
        psit_bQ[:] = 0.0
        pw_insert_gpu(psit_nG.data[b1:b2],
                      Q_G,
                      1.0,
                      psit_bQ.reshape((nb, -1)),
                      *size_c)
        psit_bR = ifftn(
            psit_bQ,
            tuple(size_c),
            norm='forward',
            overwrite_x=True)
        psit_bR *= vt_R.data
        vtpsit_bQ = fftn(
            psit_bR,
            tuple(size_c),
            norm='forward',
            overwrite_x=True)
        out_nG.data[b1:b2] = psit_nG.data[b1:b2]
        out_nG.data[b1:b2] *= e_kin_G
        out_nG.data[b1:b2] += vtpsit_bQ.reshape((nb, -1))[:, Q_G]
