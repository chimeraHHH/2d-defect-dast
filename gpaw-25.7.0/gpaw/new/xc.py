""""""
from __future__ import annotations

from typing import Callable

import numpy as np

from gpaw.core import UGArray, UGDesc
from gpaw.gpu import einsum
from gpaw.new import zips
from gpaw.new.c import (add_to_density, add_to_density_gpu, evaluate_lda_gpu,
                        evaluate_pbe_gpu)
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array2D
from gpaw.xc import XC
from gpaw.xc.functional import XCFunctional as OldXCFunctional
from gpaw.xc.gga import add_gradient_correction
from gpaw.xc.libvdwxc import VDWXC
from gpaw.xc.mgga import MGGA
from gpaw.xc.vdw import VDWFunctionalBase
from gpaw.hybrids import HybridXC


def create_functional(xc: OldXCFunctional | str | dict,
                      grid: UGDesc,
                      xp=np) -> Functional:
    exx_fraction = 0.0
    exx_omega = 0.0
    exx_yukawa = False
    if isinstance(xc, (str, dict)):
        xc = XC(xc)

    if xc.type == 'HYB':
        assert isinstance(xc, HybridXC)
        exx_fraction = xc.exx_fraction
        exx_omega = xc.omega
        exx_yukawa = xc.yukawa
        xc = xc.xc

    if xc.type == 'LDA':
        functional = LDAFunctional(xc, grid, xp)
    elif xc.type == 'GGA':
        functional = GGAFunctional(xc, grid, xp)
    elif xc.type == 'MGGA':
        functional = MGGAFunctional(xc, grid)
    else:
        raise ValueError(f'{xc.type} not supported')

    functional.exx_fraction = exx_fraction
    functional.exx_omega = exx_omega
    functional.exx_yukawa = exx_yukawa

    return functional


class Functional:
    def __init__(self,
                 xc: OldXCFunctional,
                 grid: UGDesc,
                 xp=np):
        self.xc = xc
        self.grid = grid
        self.xp = xp
        self.setup_name = self.xc.get_setup_name()
        self.name = self.xc.name
        self.type = self.xc.type
        self.xc.xp = xp
        self.xc.set_grid_descriptor(grid._gd)
        self.exx_fraction = 0.0
        self.exx_omega = 0.0
        self.exx_yukawa = False
        self.energies: dict[str, float] = {}

    def __str__(self):
        return f'name: {self.xc.get_description()}'

    def calculate(self,
                  nt_sr: UGArray,
                  taut_sr: UGArray | None = None) -> tuple[float,
                                                           UGArray,
                                                           UGArray | None]:
        raise NotImplementedError

    def calculate_paw_correction(self, setup, d, h=None):
        return self.xc.calculate_paw_correction(setup, d, h)

    def get_setup_name(self) -> str:
        return self.name

    def stress_contribution(self,
                            ibzwfs, density,
                            interpolate: Callable[[UGArray], UGArray]
                            ) -> Array2D:
        args, kwargs = self._args(ibzwfs, density, interpolate)
        if ibzwfs.kpt_band_comm.rank == 0:
            if self.xp is np:
                if isinstance(self.xc, VDWXC):
                    xck = self.xc.semilocal_xc.kernel
                    xck.calculate(*[array.data for array in args])
                    wrapper = self.xc.redist_wrapper
                    stress_vv = wrapper.calculate(args[1].data, args[3].data,
                                                  args[2].data, args[4].data,
                                                  get_stress=True)[1]
                    return stress_vv + self._stress(*args, **kwargs)
                else:
                    self.xc.kernel.calculate(*[array.data for array in args])
            # Special GPU cases:
            elif self.name == 'LDA':
                e_r, nt_sr, vt_sr = args
                evaluate_lda_gpu(nt_sr.data, vt_sr.data, e_r.data)
            elif self.name == 'PBE':
                e_r, nt_sr, vt_sr, sigma_xr, dedsigma_xr = args
                evaluate_pbe_gpu(nt_sr.data, vt_sr.data, e_r.data,
                                 sigma_xr.data, dedsigma_xr.data)
            else:
                1 / 0
            return self._stress(*args, **kwargs)
        return self.xp.zeros((3, 3))

    def _args(self,
              ibzwfs,
              density,
              interpolate: Callable[[UGArray], UGArray]
              ) -> tuple[tuple[UGArray, ...], dict]:
        raise NotImplementedError

    def _stress(self, *args: UGArray, **kwargs) -> Array2D:
        raise NotImplementedError


class LDAFunctional(Functional):
    def calculate(self,
                  nt_sr: UGArray,
                  taut_sr: UGArray | None = None) -> tuple[float,
                                                           UGArray,
                                                           UGArray | None]:
        vxct_sr = nt_sr.new()
        vxct_sr.data[:] = 0.0
        e_r = nt_sr.desc.empty(xp=self.xp)
        if self.xp is np:
            self.xc.kernel.calculate(e_r.data, nt_sr.data, vxct_sr.data)
        else:
            if self.name != 'LDA':
                raise ValueError(f'{self.name} not supported on GPU')
            evaluate_lda_gpu(nt_sr.data, vxct_sr.data, e_r.data)
        exc = e_r.integrate()
        return exc, vxct_sr, None

    def _args(self, ibzwfs, density, interpolate):
        if ibzwfs.kpt_band_comm.rank != 0:
            return (), {}
        nt_sR = density.nt_sR
        e_r = self.grid.empty(xp=self.xp)
        nt_sr = interpolate(nt_sR)
        vt_sr = nt_sr.new(zeroed=True)
        return (e_r, nt_sr, vt_sr), {}

    def _stress(self,  # type: ignore
                e_r: UGArray,
                nt_sr: UGArray,
                vt_sr: UGArray) -> Array2D:
        P = e_r.integrate(skip_sum=True)
        for vt_r, nt_r in zip(vt_sr, nt_sr):
            P -= vt_r.integrate(nt_r, skip_sum=True)
        return float(P) * self.xp.eye(3)


class GGAFunctional(LDAFunctional):
    def __init__(self,
                 xc: OldXCFunctional,
                 grid: UGDesc,
                 xp=np):
        super().__init__(xc, grid, xp)
        # xc already has Gradient.apply bound methods!!!
        self.grad_v = [grad.__self__ for grad in xc.grad_v]  # type: ignore

    def _evaluate_xc_cpu(self, args):
        if 'vdW' not in self.name:
            self.xc.kernel.calculate(*args)
        elif isinstance(self.xc, VDWXC):
            libvdwxc = self.xc.libvdwxc
            nspins = args[1].shape[0]
            if libvdwxc is None or self.xc._nspins != nspins:
                self.xc._nspins = nspins
                self.xc.initialize_backend(self.xc.gd)
            self.xc.semilocal_xc.kernel.calculate(*args)
            self.xc.calculate_vdw(*args[:5])
        else:
            assert isinstance(self.xc, VDWFunctionalBase)
            self.xc.kernel.calculate(*args)
            self.xc.calculate_correlation(*args[:5])

    def calculate(self,
                  nt_sr: UGArray,
                  taut_sr: UGArray | None = None) -> tuple[float,
                                                           UGArray,
                                                           UGArray | None]:
        gradn_svr, sigma_xr = gradient_and_sigma(self.grad_v, nt_sr)

        vxct_sr = nt_sr.new(zeroed=True)
        dedsigma_xr = sigma_xr.new()
        e_r = self.grid.empty(xp=self.xp)

        if self.xp is np:
            args = [a.data
                    for a in [e_r, nt_sr, vxct_sr, sigma_xr, dedsigma_xr]]
            self._evaluate_xc_cpu(args)
        else:
            if self.name != 'PBE':
                raise ValueError(f'{self.name} not supported on GPU')
            evaluate_pbe_gpu(nt_sr.data, vxct_sr.data, e_r.data,
                             sigma_xr.data, dedsigma_xr.data)

        add_gradient_correction([grad.apply for grad in self.grad_v],
                                gradn_svr.data, sigma_xr.data,
                                dedsigma_xr.data, vxct_sr.data)
        exc = e_r.integrate()
        return exc, vxct_sr, None

    def _args(self,
              ibzwfs,
              density,
              interpolate: Callable[[UGArray], UGArray]
              ) -> tuple[tuple[UGArray, ...], dict]:
        args, kwargs = LDAFunctional._args(self, ibzwfs, density, interpolate)
        if args:
            e_r, nt_sr, vt_sr = args
            gradn_svr, sigma_xr = gradient_and_sigma(self.grad_v, nt_sr)
            dedsigma_xr = sigma_xr.new()
            args += (sigma_xr, dedsigma_xr)
            kwargs = {'gradn_svr': gradn_svr}
        return args, kwargs

    def _stress(self,  # type: ignore
                e_r, nt_sr, vt_sr, sigma_xr, dedsigma_xr,
                gradn_svr,
                ) -> Array2D:
        stress_vv = LDAFunctional._stress(self, e_r, nt_sr, vt_sr)
        P = 0.0
        for sigma_r, dedsigma_r in zip(sigma_xr, dedsigma_xr):
            P -= 2 * sigma_r.integrate(dedsigma_r, skip_sum=True)
        stress_vv += float(P) * self.xp.eye(3)

        nspins = len(nt_sr)
        for v1 in range(3):
            for v2 in range(3):
                stress_vv[v1, v2] -= 2 * dedsigma_xr[0].integrate(
                    gradn_svr[0, v1] * gradn_svr[0, v2], skip_sum=True)
                if nspins == 2:
                    stress_vv[v1, v2] -= 2 * dedsigma_xr[1].integrate(
                        gradn_svr[0, v1] * gradn_svr[1, v2], skip_sum=True)
                    stress_vv[v1, v2] -= 2 * dedsigma_xr[2].integrate(
                        gradn_svr[1, v1] * gradn_svr[1, v2], skip_sum=True)

        return stress_vv


def gradient_and_sigma(grad_v, n_sr: UGArray) -> tuple[UGArray, UGArray]:
    """Calculate gradient of density and sigma.

    Returns:::

      _    _
      ∇ n (r)
         s

    and:::

         _     _   2     _    _          _   2
      σ (r) = |∇n |, σ = ∇n . ∇n ,  σ = |∇n |
       0         0    1    0    1    2     1
    """
    nspins = n_sr.dims[0]
    xp = n_sr.xp

    gradn_svr = n_sr.desc.empty((nspins, 3), xp=xp)
    for v, grad in enumerate(grad_v):
        for s in range(nspins):
            grad(n_sr[s], gradn_svr[s, v])

    sigma_xr = n_sr.desc.empty(nspins * 2 - 1, xp=xp)
    dot_product(gradn_svr[0], None, sigma_xr[0])
    if nspins == 2:
        dot_product(gradn_svr[0], gradn_svr[1], sigma_xr[1])
        dot_product(gradn_svr[1], None, sigma_xr[2])

    return gradn_svr, sigma_xr


def dot_product(a_vr, b_vr, out_r):
    xp = a_vr.xp
    if b_vr is None:
        out_r.data[:] = 0.0
        if xp is np:
            for a_r in a_vr.data:
                add_to_density(1.0, a_r, out_r.data)
        else:
            add_to_density_gpu(xp.ones(3), a_vr.data, out_r.data)
    else:
        einsum('vabc, vabc -> abc', a_vr.data, b_vr.data, out=out_r.data)


class MGGAFunctional(GGAFunctional):
    def get_setup_name(self):
        return 'PBE'

    def calculate(self,
                  nt_sr: UGArray,
                  taut_sr: UGArray | None = None) -> tuple[float,
                                                           UGArray,
                                                           UGArray | None]:
        gradn_svr, sigma_xr = gradient_and_sigma(self.grad_v, nt_sr)
        if isinstance(self.xc, VDWXC):
            assert isinstance(self.xc.semilocal_xc, MGGA), self.xc.semilocal_xc
        else:
            assert isinstance(self.xc, MGGA), self.xc
        e_r = self.grid.empty()
        if taut_sr is None:
            taut_sr = nt_sr.new(zeroed=True)
        dedtaut_sr = taut_sr.new()
        vxct_sr = taut_sr.new()
        vxct_sr.data[:] = 0.0
        dedsigma_xr = sigma_xr.new()
        args = [e_r, nt_sr, vxct_sr, sigma_xr, dedsigma_xr,
                taut_sr, dedtaut_sr]
        args = [array.data for array in args]
        self._evaluate_xc_cpu(args)
        add_gradient_correction([grad.apply for grad in self.grad_v],
                                gradn_svr.data, sigma_xr.data,
                                dedsigma_xr.data, vxct_sr.data)
        return e_r.integrate(), vxct_sr, dedtaut_sr

    def _args(self,
              ibzwfs,
              density,
              interpolate: Callable[[UGArray], UGArray]):
        args, kwargs = GGAFunctional._args(self, ibzwfs, density, interpolate)
        taut_swR = _taut(ibzwfs, density.nt_sR.desc)

        if not args:
            return (), {}

        e_r, nt_sr, vt_sr, sigma_xr, dedsigma_xr = args
        taut_sR = density.taut_sR
        assert taut_sR is not None
        taut_sr = interpolate(taut_sR)
        dedtaut_sr = taut_sr.new()
        args += (taut_sr, dedtaut_sr)
        kwargs['taut_swR'] = taut_swR
        kwargs['interpolate'] = interpolate

        return args, kwargs

    def _stress(self,  # type: ignore
                e_r,
                nt_sr, vt_sr,
                sigma_xr, dedsigma_xr,
                taut_sr, dedtaut_sr,
                gradn_svr,
                taut_swR,
                interpolate
                ) -> Array2D:  # type: ignore
        stress_vv = GGAFunctional._stress(
            self, e_r, nt_sr, vt_sr, sigma_xr, dedsigma_xr,
            gradn_svr=gradn_svr)
        for taut_wR, dedtaut_r in zips(taut_swR, dedtaut_sr):
            w = 0
            for v1 in range(3):
                for v2 in range(v1, 3):
                    taut_r = interpolate(taut_wR[w])
                    s = taut_r.integrate(dedtaut_r, skip_sum=True)
                    stress_vv[v1, v2] -= s
                    if v2 != v1:
                        stress_vv[v2, v1] -= s
                    w += 1

        P = 0.0
        for taut_r, dedtaut_r in zip(taut_sr, dedtaut_sr):
            P -= taut_r.integrate(dedtaut_r, skip_sum=True)
        stress_vv += P * np.eye(3)

        return stress_vv


def _taut(ibzwfs: IBZWaveFunctions, grid: UGDesc) -> UGArray | None:
    """Calculate upper half of symmetric ked tensor.

    Returns ``taut_swR=taut_svvR``.  Mapping from ``w`` to ``vv``::

        0 1 2
        . 3 4
        . . 5

    Only cores with ``kpt_comm.rank==0`` and ``band_comm.rank==0``
    return the uniform grids - other cores return None.
    """
    # "1" refers to undistributed arrays
    dpsit1_vR = grid.new(comm=None, dtype=ibzwfs.dtype).empty(3)
    taut1_swR = grid.new(comm=None).zeros((ibzwfs.nspins, 6))
    assert isinstance(taut1_swR, UGArray)  # Argggghhh!
    domain_comm = grid.comm

    for wfs in ibzwfs:
        assert isinstance(wfs, PWFDWaveFunctions)
        psit_nG = wfs.psit_nX
        pw = psit_nG.desc

        pw1 = pw.new(comm=None)
        psit1_G = pw1.empty()
        iGpsit1_G = pw1.empty()
        Gplusk1_Gv = pw1.reciprocal_vectors()

        occ_n = wfs.weight * wfs.spin_degeneracy * wfs.myocc_n

        (N,) = psit_nG.mydims
        for n1 in range(0, N, domain_comm.size):
            n2 = min(n1 + domain_comm.size, N)
            psit_nG[n1:n2].gather_all(psit1_G)
            n = n1 + domain_comm.rank
            if n >= N:
                continue
            f = occ_n[n]
            if f == 0.0:
                continue
            for Gplusk1_G, dpsit1_R in zips(Gplusk1_Gv.T, dpsit1_vR):
                iGpsit1_G.data[:] = psit1_G.data
                iGpsit1_G.data *= 1j * Gplusk1_G
                iGpsit1_G.ifft(out=dpsit1_R)
            w = 0
            for v1 in range(3):
                for v2 in range(v1, 3):
                    taut1_swR[wfs.spin, w].data += (
                        f * (dpsit1_vR[v1].data.conj() *
                             dpsit1_vR[v2].data).real)
                    w += 1

    ibzwfs.kpt_comm.sum(taut1_swR.data, 0)
    if ibzwfs.kpt_comm.rank == 0:
        ibzwfs.band_comm.sum(taut1_swR.data, 0)
        if ibzwfs.band_comm.rank == 0:
            domain_comm.sum(taut1_swR.data, 0)
            if domain_comm.rank == 0:
                symmetries = ibzwfs.ibz.symmetries
                taut1_swR.symmetrize(symmetries.rotation_scc,
                                     symmetries.translation_sc)
            taut_swR = grid.empty((ibzwfs.nspins, 6))
            taut_swR.scatter_from(taut1_swR)
            return taut_swR
    return None
