from __future__ import annotations

from math import nan

import numpy as np
from gpaw.core import PWArray, PWDesc, UGArray, UGDesc
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.hybrids.paw import pawexxvv
from gpaw.mpi import broadcast
# from gpaw.new import zips as zip
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.new.pw.hybrids import fft, truncated_coulomb
from gpaw.new.pw.nschse import Psit, ibz2bz
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.setup import Setups
from gpaw.utilities import unpack_hermitian


class PWHybridHamiltonianK(PWHamiltonian):
    def __init__(self,
                 grid: UGDesc,
                 pw: PWDesc,
                 xc,
                 setups: Setups,
                 relpos_ac,
                 atomdist,
                 log,
                 kpt_comm,
                 comm):
        super().__init__(grid.new(dtype=complex), pw)
        self.pw = pw
        self.exx_fraction = xc.exx_fraction
        self.exx_omega = xc.exx_omega
        self.xc = xc
        self.kpt_comm = kpt_comm
        self.comm = comm
        self.log = log
        self.cgrid = grid.new(dtype=complex, comm=None)
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        xp = np
        self.plan = self.cgrid.fft_plans(xp=xp)

        self.relpos_ac = relpos_ac
        self.setups = setups

        # Stuff for PAW core-core, core-valence and valence-valence correctios:
        self.exx_cc = sum(setup.ExxC for setup in setups) * self.exx_fraction
        self.VC_aii = [unpack_hermitian(setup.X_p * self.exx_fraction)
                       for setup in setups]
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.VV_app = [setup.M_pp * self.exx_fraction for setup in setups]

        self.ghat_aLG = setups.create_compensation_charges(
            pw, relpos_ac, atomdist)

        self.mypsits: list[Psit] = []
        self.nbzk = 0

    def update_wave_functions(self,
                              ibzwfs: PWFDIBZWaveFunctions):
        self.mypsits, _ = ibz2bz(
            ibzwfs, self.setups, self.relpos_ac, self.cgrid, self.plan,
            self.log if self.nbzk == 0 else None)
        self.nbzk = len(ibzwfs.ibz.bz)
        self.xc.energies = {'hybrid_xc': 0.0,
                            'hybrid_kinetic_correction': 0.0}

    def apply_orbital_dependent(self,
                                ibzwfs: IBZWaveFunctions,
                                D_asii,
                                psit2_nG: XArray,
                                spin: int,
                                Htpsit2_nG: XArray,
                                calculate_energy: bool = False) -> None:
        assert isinstance(psit2_nG, PWArray)
        assert isinstance(Htpsit2_nG, PWArray)
        assert isinstance(ibzwfs, PWFDIBZWaveFunctions)
        assert len(ibzwfs.ibz) % self.kpt_comm.size == 0

        D_aii = D_asii[:, spin].copy()
        if ibzwfs.nspins == 1:
            D_aii = D_aii.copy()
            D_aii.data *= 0.5

        for wfs_s in ibzwfs.wfs_qs:
            wfs = wfs_s[spin]
            if np.allclose(wfs.psit_nX.desc.kpt_c, psit2_nG.desc.kpt_c):
                pt_aiG = wfs.pt_aiX
                weight = wfs.weight
                break
        else:  # no break
            1 / 0

        evv, evc, ekin = self._apply1(spin, D_aii, pt_aiG,
                                      psit2_nG, Htpsit2_nG,
                                      wfs.occ_n, calculate_energy)
        if calculate_energy:
            for name, e in [('hybrid_xc', evv + evc),
                            ('hybrid_kinetic_correction', ekin)]:
                e *= ibzwfs.spin_degeneracy * weight
                self.xc.energies[name] += e
            self.xc.energies['hybrid_xc'] += self.exx_cc

    def _apply1(self,
                spin: int,
                D_aii,
                pt_aiG,
                psit_nG: PWArray,
                Htpsit_nG: PWArray,
                f_n: np.ndarray,
                calculate_energy: bool) -> tuple[float, float, float]:
        comm = self.comm
        band_comm = psit_nG.comm

        P_ani = pt_aiG.integrate(psit_nG)

        V_ani = P_ani.new()

        evv = 0.0
        evc = 0.0
        ekin = 0.0
        for a, D_ii in D_aii.items():
            VV_ii = pawexxvv(self.VV_app[a], D_ii)
            VC_ii = self.VC_aii[a]
            V_ii = -VC_ii - 2 * VV_ii
            V_ani[a] = P_ani[a] @ V_ii
            if calculate_energy:
                ec = (D_ii * VC_ii).sum()
                ev = (D_ii * VV_ii).sum()
                ekin += ec + 2 * ev
                evv -= ev
                evc -= ec

        e = 0.0
        for rank in range(self.kpt_comm.size):
            data = None
            if rank == self.kpt_comm.rank:
                psit_nG = psit_nG.gather()
                P_ani = P_ani.gather()
                if psit_nG is not None:
                    data = (psit_nG, P_ani, spin)
            psit_nG, P_ani, s = broadcast(data, rank * band_comm.size, comm)
            e += self._apply2(psit_nG, P_ani, s, Htpsit_nG, V_ani, f_n,
                              calculate_energy)
            pt_aiG.add_to(Htpsit_nG, V_ani)

        if not calculate_energy:
            return nan, nan, nan

        e = comm.sum_scalar(e)
        evv += 0.5 * e
        ekin -= e

        return evv, evc, ekin

    def _apply2(self,
                psit2_nG: PWArray,
                P2_ani: AtomArrays,
                spin: int,
                Htpsit2_nG,
                V2_ani,
                f2_n: np.ndarray,
                calculate_energy: bool) -> float:
        ut2_nR = self.grid_local.empty(len(psit2_nG))
        psit2_nG.ifft(out=ut2_nR, plan=self.plan, periodic=False)

        e = 0.0
        pw2 = psit2_nG.desc
        for psit1 in self.mypsits:
            if psit1.spin == spin:
                pw = pw2.new(kpt=pw2.kpt_c - psit1.kpt_c)
                v_G = truncated_coulomb(pw, self.exx_omega)
                e += self._apply3(
                    v_G, psit1, ut2_nR, P2_ani, Htpsit2_nG, V2_ani, f2_n,
                    calculate_energy)
        e *= -self.exx_fraction / self.nbzk
        return self.comm.sum_scalar(e)

    def _apply3(self,
                v_G: PWArray,
                psit1: Psit,
                ut2_nR: UGArray,
                P2_ani: AtomArrays,
                Htpsit2_nG: PWArray,
                V2_ani,
                f2_n: np.ndarray,
                calculate_energy: bool) -> float:
        ut1_nR = psit1.ut_nR
        Q1_aniL = psit1.Q_aniL
        f1_n = psit1.f_n
        pw = v_G.desc
        ghat_aLG = self.setups.create_compensation_charges(pw, self.relpos_ac)
        v2_G = Htpsit2_nG.desc.empty()
        e = 0.0
        for n1, ut1_R in enumerate(ut1_nR.data):
            rhot_nR = ut2_nR.copy()
            rhot_nR.data *= ut1_R.conj()
            Q_anL = {}
            for a, Q1_niL in Q1_aniL.items():
                Q_anL[a] = P2_ani[a] @ Q1_niL[n1]
            rhot_nG = pw.empty(len(rhot_nR))
            fft(rhot_nR, rhot_nG, plan=self.plan)
            ghat_aLG.add_to(rhot_nG, Q_anL)
            if not calculate_energy:
                rhot_nG.data *= v_G.data
            else:
                for rhot_G, f2 in zip(rhot_nG, f2_n):
                    a_G = rhot_G.copy()
                    rhot_G.data *= v_G.data
                    e12 = a_G.integrate(rhot_G).real * f2 * f1_n[n1]
                    e += e12
            V2_anL = ghat_aLG.integrate(rhot_nG)
            rhot_nG.ifft(out=rhot_nR)
            rhot_nR.data *= ut1_R.data
            x = self.exx_fraction * f1_n[n1] / self.nbzk
            for v2_R, Htpsit2_G in zip(rhot_nR, Htpsit2_nG):
                v2_R.fft(out=v2_G)
                Htpsit2_G.data -= v2_G.data * x
            for a, Q1_niL in Q1_aniL.items():
                V2_ani[a][:] -= x * V2_anL[a] @ Q1_niL[n1].T.conj()
        return e
