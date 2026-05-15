"""PW-mode stress tensor calculation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gpaw.core.atom_arrays import AtomArrays
from gpaw.gpu import synchronize, as_np
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array2D
from gpaw.core import PWArray
from gpaw.utilities import as_real_dtype
if TYPE_CHECKING:
    from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator


def calculate_stress(pot_calc: PlaneWavePotentialCalculator,
                     ibzwfs, density, potential,
                     vt_g: PWArray,
                     nt_g: PWArray,
                     dedtaut_g: PWArray | None) -> Array2D:
    """Calculate symmetrized stress tensor."""
    comm = ibzwfs.comm
    xp = density.nt_sR.xp
    dom = density.nt_sR.desc

    ibzwfs.make_sure_wfs_are_read_from_gpw_file()
    s_vv = get_wfs_stress(ibzwfs, potential.dH_asii)
    s_vv += pot_calc.xc.stress_contribution(
        ibzwfs, density, pot_calc.interpolate)

    if ibzwfs.kpt_comm.rank == 0 and ibzwfs.band_comm.rank == 0:
        vHt_h = potential.vHt_x
        assert vHt_h is not None
        pw = vHt_h.desc
        G_Gv = xp.asarray(pw.G_plus_k_Gv)
        vHt2_hz = vHt_h.data.view(float).reshape((len(G_Gv), 2))**2
        s_vv += (xp.einsum('Gz, Gv, Gw -> vw', vHt2_hz, G_Gv, G_Gv) *
                 pw.dv / (2 * np.pi))
        Q_aL = density.calculate_compensation_charge_coefficients()
        s_vv += pot_calc.poisson_solver.stress_contribution(vHt_h, Q_aL)
        if ibzwfs.domain_comm.rank == 0:
            s_vv -= xp.eye(3) * potential.e_stress
        s_vv += pot_calc.vbar_ag.stress_contribution(nt_g)
        s_vv += density.nct_aX.stress_contribution(vt_g)

        if dedtaut_g is not None:
            s_vv += density.tauct_aX.stress_contribution(dedtaut_g)

    s_vv = as_np(s_vv)

    if xp is not np:
        synchronize()
    comm.sum(s_vv, 0)

    vol = dom.volume
    s_vv = 0.5 / vol * (s_vv + s_vv.T)

    # Symmetrize:
    sigma_vv = np.zeros((3, 3))
    cell_cv = dom.cell_cv
    icell_cv = dom.icell
    rotation_scc = ibzwfs.ibz.symmetries.rotation_scc
    for U_cc in rotation_scc:
        M_vv = (icell_cv.T @ (U_cc @ cell_cv)).T
        sigma_vv += M_vv.T @ s_vv @ M_vv
    sigma_vv /= len(rotation_scc)

    # Make sure all agree on the result (redundant calculation on
    # different cores involving BLAS might give slightly different
    # results):

    sigma_vv += pot_calc.extensions_stress_contribution
    comm.broadcast(sigma_vv, 0)
    return sigma_vv


def get_wfs_stress(ibzwfs: IBZWaveFunctions,
                   dH_asii: AtomArrays) -> Array2D:
    xp = ibzwfs.xp
    sigma_vv = xp.zeros((3, 3))
    for wfs in ibzwfs:
        assert isinstance(wfs, PWFDWaveFunctions)
        occ_n = xp.asarray(wfs.weight * wfs.spin_degeneracy * wfs.myocc_n)
        sigma_vv += get_kinetic_stress(wfs, occ_n)
        sigma_vv += get_paw_stress(wfs, dH_asii, occ_n)
    return sigma_vv


def get_kinetic_stress(wfs: PWFDWaveFunctions,
                       occ_n) -> Array2D:
    psit_nG = wfs.psit_nX
    pw = psit_nG.desc
    xp = psit_nG.xp
    psit_nGz = psit_nG.data.view(
        as_real_dtype(pw.dtype)).reshape(psit_nG.data.shape + (2,))
    psit2_G = xp.einsum('n, nGz, nGz -> G', occ_n, psit_nGz, psit_nGz)
    Gk_Gv = xp.asarray(pw.G_plus_k_Gv)
    sigma_vv = xp.einsum('G, Gv, Gw -> vw', psit2_G, Gk_Gv, Gk_Gv)
    x = pw.dv
    if np.issubdtype(pw.dtype, np.floating):
        x *= 2
    return -x * sigma_vv


def get_paw_stress(wfs: PWFDWaveFunctions,
                   dH_asii: AtomArrays,
                   occ_n) -> Array2D:
    xp = wfs.xp
    eig_n1 = xp.asarray(wfs.myeig_n[:, None])
    a_ani = {}
    s = 0.0
    for a, P_ni in wfs.P_ani.items():
        Pf_ni = P_ni * occ_n[:, None]
        dH_ii = dH_asii[a][wfs.spin]
        dS_ii = xp.asarray(wfs.setups[a].dO_ii)
        a_ni = Pf_ni @ dH_ii - Pf_ni * eig_n1 @ dS_ii
        s += xp.vdot(P_ni, a_ni)
        a_ani[a] = 2 * a_ni.conj()
    s_vv = wfs.pt_aiX.stress_contribution(wfs.psit_nX, a_ani)
    return s_vv - float(s.real) * xp.eye(3)
