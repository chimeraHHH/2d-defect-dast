from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import pi, nan

import numpy as np
from gpaw.core import PWArray, PWDesc, UGArray, UGDesc
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.hybrids.paw import pawexxvv
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.new import zips as zip
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.typing import Array1D
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.blas import mmm


def coulomb(pw: PWDesc,
            grid: UGDesc,
            omega: float,
            yukawa: bool = False) -> PWArray:
    if omega == 0.0:
        wstc = WignerSeitzTruncatedCoulomb(
            pw.cell_cv, np.array([1, 1, 1]))
        return wstc.get_potential_new(pw, grid)
    return truncated_coulomb(pw, omega, yukawa)


def truncated_coulomb(pw: PWDesc,
                      omega: float = 0.11,
                      yukawa: bool = False) -> PWArray:
    """Fourier transform of truncated Coulomb.

    Real space:::

        erfc(ωr)
        --------.
           r

    Reciprocal space:::

        4π             _ _ 2     2
      ------(1 - exp(-(G+k) /(4 ω )))
       _ _ 2
      (G+k)

    (G+k=0 limit is pi/ω^2).
    """
    v_G = pw.empty()
    G2_G = pw.ekin_G * 2
    if yukawa:
        v_G.data[:] = 4 * pi / (G2_G + omega**2)
    else:
        v_G.data[:] = 4 * pi * (1 - np.exp(-G2_G / (4 * omega**2)))
        ok_G = G2_G > 1e-10
        v_G.data[ok_G] /= G2_G[ok_G]
        v_G.data[~ok_G] = pi / omega**2
    return v_G


@dataclass
class Psi:
    psit_nG: PWArray
    P_ani: AtomArrays
    f_n: Array1D | None = None
    psit_nR: UGArray | None = None

    def empty(self):
        return Psi(self.psit_nG.new(),
                   self.P_ani.new(),
                   np.empty_like(self.f_n))

    @cached_property
    def comm(self):
        return self.psit_nG.comm

    def send(self, rank):
        self.requests = [self.comm.send(self.psit_nG.data, rank, block=False),
                         self.comm.send(self.P_ani.data, rank, block=False),
                         self.comm.send(self.f_n, rank, block=False)]

    def receive(self, rank):
        self.requests = [
            self.comm.receive(self.psit_nG.data, rank, block=False),
            self.comm.receive(self.P_ani.data, rank, block=False),
            self.comm.receive(self.f_n, rank, block=False)]

    def wait(self):
        comm = self.psit_nG.comm
        comm.waitall(self.requests)


class PWHybridHamiltonian(PWHamiltonian):
    band_local = False

    def __init__(self,
                 grid: UGDesc,
                 pw: PWDesc,
                 xc,
                 setups,
                 relpos_ac,
                 atomdist,
                 comp_charge_in_real_space: bool = False):
        super().__init__(grid, pw)
        self.comp_charge_in_real_space = comp_charge_in_real_space
        self.pw = pw
        self.exx_fraction = xc.exx_fraction
        self.exx_omega = xc.exx_omega
        self.exx_yukawa = xc.exx_yukawa
        self.xc = xc

        # Stuff for PAW core-core, core-valence and valence-valence correctios:
        self.exx_cc = sum(setup.ExxC for setup in setups) * self.exx_fraction
        self.VC_aii = [unpack_hermitian(setup.X_p * self.exx_fraction)
                       for setup in setups]
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.VV_app = [setup.M_pp * self.exx_fraction for setup in setups]

        self.v_G = coulomb(pw, grid, self.exx_omega)
        self.v_G.data *= self.exx_fraction

        desc = grid if comp_charge_in_real_space else pw

        self.ghat_aLX = setups.create_compensation_charges(
            desc, relpos_ac, atomdist)
        if not comp_charge_in_real_space:
            self.ghat_aLX._lazy_init()
            self.ghat_GA = self.ghat_aLX._lfc.expand()
        else:
            self.ghat_GA = None
        # self.plan = grid.fft_plans()

    def apply_orbital_dependent(self,
                                ibzwfs: IBZWaveFunctions,
                                D_asii,
                                psit2_nG: XArray,
                                spin: int,
                                Htpsit2_nG: XArray,
                                calculate_energy: bool = False) -> None:
        assert isinstance(psit2_nG, PWArray)
        assert isinstance(Htpsit2_nG, PWArray)
        wfs = ibzwfs.wfs_qs[0][spin]
        D_aii = D_asii[:, spin].copy()
        if ibzwfs.nspins == 1:
            D_aii = D_aii.copy()
            D_aii.data *= 0.5
        psi1 = Psi(wfs.psit_nX, wfs.P_ani, wfs.myocc_n)
        pt_aiG = wfs.pt_aiX

        if calculate_energy:
            # We are doing a subspace diagonalization ...
            evv, evc, ekin = self.apply1(D_aii, pt_aiG,
                                         psi1, psi1, Htpsit2_nG)
            for name, e in [('hybrid_xc', evv + evc),
                            ('hybrid_kinetic_correction', ekin)]:
                e *= ibzwfs.spin_degeneracy
                if spin == 0:
                    self.xc.energies[name] = e
                else:
                    self.xc.energies[name] += e
            self.xc.energies['hybrid_xc'] += self.exx_cc
            return

        # We are applying the exchange operator (defined by psit1_nG,
        # P1_ani, f1_n and D_aii) to another set of wave functions
        # (psit2_nG):
        psi2 = Psi(psit2_nG, pt_aiG.integrate(psit2_nG))
        self.apply1(D_aii, pt_aiG, psi1, psi2, Htpsit2_nG)

    def apply1(self,
               D_aii,
               pt_aiG,
               psi1: Psi,
               psi2: Psi,
               Htpsit_nG: PWArray) -> tuple[float, float, float]:
        comm = Htpsit_nG.comm
        mynbands1 = psi1.psit_nG.mydims[0]
        mynbands2 = psi2.psit_nG.mydims[0]
        same = psi1 is psi2
        evv = 0.0
        evc = 0.0
        ekin = 0.0
        B_ani = {}
        for a, D_ii in D_aii.items():
            VV_ii = pawexxvv(self.VV_app[a], D_ii)
            VC_ii = self.VC_aii[a]
            V_ii = -VC_ii - 2 * VV_ii
            B_ani[a] = psi2.P_ani[a] @ V_ii
            if same:
                ec = (D_ii * VC_ii).sum()
                ev = (D_ii * VV_ii).sum()
                ekin += ec + 2 * ev
                evv -= ev
                evc -= ec

        Q_anL = self.ghat_aLX.empty(mynbands1)
        Q_nA = Q_anL.data
        assert Q_nA.shape == (mynbands1,
                              sum(delta_iiL.shape[2]
                                  for delta_iiL in self.delta_aiiL))
        assert Q_nA.dtype == self.pw.dtype

        rhot_nR = self.grid_local.empty(mynbands1)
        rhot_nG = self.pw.empty(mynbands1)
        vrhot_G = self.pw.empty()

        if not same or comm.size > 1:
            psit1_nR = self.grid_local.empty(mynbands1)
        else:
            psit1_nR = None

        e = 0.0
        for p in range(comm.size):
            if p < comm.size - 1:
                psi1.send((comm.rank + 1) % comm.size)
                if p == 0:
                    psi = psi1.empty()
                psi.receive((comm.rank - 1) % comm.size)
            if p == 0:
                psi2.psit_nR = self.grid_local.empty(mynbands2)
                ifft(psi2.psit_nG, psi2.psit_nR, self.plan)
            e += self.inner(psi1, psi2,
                            Q_anL,
                            psit1_nR,
                            rhot_nG, rhot_nR, vrhot_G,
                            Htpsit_nG, B_ani)
            if p < comm.size - 1:
                psi.wait()
                psi1.wait()
                if p == 0:
                    psi1 = psi
                    psi = psi1.empty()
                else:
                    psi1, psi = psi, psi1

        pt_aiG.add_to(Htpsit_nG, B_ani)

        if same:
            e = comm.sum_scalar(e)
            evv -= 0.5 * e
            ekin += e
            return evv, evc, ekin

        return nan, nan, nan

    def inner(self, psi1, psi2,
              Q_anL,
              psit1_nR,
              rhot_nG, rhot_nR, vrhot_G,
              Htpsit_nG, B_ani):
        Q1_aniL = {a: np.einsum('ijL, nj -> niL',
                                delta_iiL, psi1.P_ani[a])
                   for a, delta_iiL in enumerate(self.delta_aiiL)}

        if psi1 is psi2:
            psit1_nR = psi2.psit_nR
        else:
            ifft(psi1.psit_nG, psit1_nR, self.plan)

        e = 0.0
        for n2, (psit2_R, out_G) in enumerate(zip(psi2.psit_nR, Htpsit_nG)):
            rhot_nR.data[:] = psit1_nR.data * psit2_R.data.conj()
            for a, Q1_niL in Q1_aniL.items():
                P2_i = psi2.P_ani[a][n2]
                Q_anL[a][:] = P2_i.conj() @ Q1_niL
            e += self.inner2(
                psi1, psi2,
                rhot_nR, rhot_nG,
                vrhot_G,
                Q_anL, Q1_aniL, B_ani, n2)
            rhot_nR.data *= psit1_nR.data
            fft(rhot_nR, rhot_nG, self.plan)
            out_G.data -= psi1.f_n @ rhot_nG.data
        return e

    def inner2(self,
               psi1, psi2,
               rhot_nR, rhot_nG,
               vrhot_G,
               Q_anL, Q1_aniL, B_ani, n2) -> float:
        if self.comp_charge_in_real_space:
            return self.inner2_real_space(psi1, psi2,
                                          rhot_nR, rhot_nG,
                                          vrhot_G,
                                          Q_anL, Q1_aniL, B_ani, n2)
        fft(rhot_nR, rhot_nG, plan=self.plan)
        if self.pw.dtype == float:
            # Note that G runs over
            # G0.real, G0.imag, G1.real, G1.imag, ...
            mmm(1.0 / self.pw.dv, Q_anL.data, 'N', self.ghat_GA, 'T',
                1.0, rhot_nG.data.view(float))
        else:
            mmm(1.0 / self.pw.dv, Q_anL.data, 'N', self.ghat_GA, 'T',
                1.0, rhot_nG.data)

        e = 0.0
        for n1, (rhot_R, rhot_G, f1) in enumerate(zip(rhot_nR,
                                                      rhot_nG,
                                                      psi1.f_n)):
            vrhot_G.data = rhot_G.data * self.v_G.data
            if psi2.f_n is not None:
                e12 = rhot_G.integrate(vrhot_G).real
                e += f1 * psi2.f_n[n2] * e12
            rhot_G.data[:] = vrhot_G.data

            if self.pw.dtype == float:
                vrhot_G.data[0] *= 0.5
                A1_A = vrhot_G.data.view(float) @ self.ghat_GA * 2.0
            else:
                A1_A = vrhot_G.data @ self.ghat_GA
            A1 = 0
            for a, Q1_niL in Q1_aniL.items():
                A2 = A1 + Q1_niL.shape[2]
                B_ani[a][n2] -= Q1_niL[n1] @ (f1 * A1_A[A1:A2])
                A1 = A2
        ifft(rhot_nG, rhot_nR, plan=self.plan)
        return e

    def inner2_real_space(self,
                          psi1, psi2,
                          rhot_nR, rhot_nG,
                          vrhot_G,
                          Q_anL, Q1_aniL, B_ani, n2) -> float:
        self.ghat_aLX.add_to(rhot_nR, Q_anL)
        fft(rhot_nR, rhot_nG, plan=self.plan)
        e = 0.0
        for n1, (rhot_R, rhot_G, f1) in enumerate(zip(rhot_nR,
                                                      rhot_nG,
                                                      psi1.f_n)):
            vrhot_G.data = rhot_G.data * self.v_G.data
            if psi2.f_n is not None:
                e += f1 * psi2.f_n[n2] * rhot_G.integrate(vrhot_G).real
            rhot_G.data[:] = vrhot_G.data

        ifft(rhot_nG, rhot_nR, plan=self.plan)

        A1_anL = self.ghat_aLX.integrate(rhot_nR)
        for a, Q1_niL in Q1_aniL.items():
            B_ani[a][n2] -= np.einsum('niL, n, nL -> i',
                                      Q1_niL, psi1.f_n, A1_anL[a])
        return e


def ifft(psit_nG, out_nR, plan):
    for psit_G, out_R in zip(psit_nG, out_nR):
        psit_G.ifft(out=out_R, plan=plan)


def fft(rhot_nR, rhot_nG, plan):
    for rhot_R, rhot_G in zip(rhot_nR, rhot_nG):
        rhot_R.fft(out=rhot_G, plan=plan)
