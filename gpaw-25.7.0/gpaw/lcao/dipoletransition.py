import numpy as np

from gpaw.mpi import world
from gpaw.typing import ArrayND


# NOTE: This routine is not specific to Raman per se. Maybe it should go
#       somewhere else?
def get_dipole_transitions(wfs) -> ArrayND:
    r"""
    Finds dipole transition matrix elements based on the velocity form.

    Dipole and momentum matrix elements are related by the expression:
    <nk|r|mk> = -i hbar/m <nk|p|mk> / (E_nk-E_mk)

    For n=m this is ill defined, and element will be set to zero.

    Parameters
    ----------
    wfs
        LCAO WaveFunctions object
    """
    p_skvnm = get_momentum_transitions(wfs, False)
    r_skvnm = np.zeros_like(p_skvnm)
    for kpt in wfs.kpt_u:
        # NOTE: Check whether it's this way or other way around.
        deltaE = kpt.eps_n[:, None] - kpt.eps_n[None, :]
        # The treatment of the energy difference in the denominator is
        # from https://doi.org/10.1103/PhysRevB.52.14636 eq 7 and note 18
        # therein
        r_skvnm[kpt.s, kpt.k, :] = -1j * p_skvnm[kpt.s, kpt.k, :] * \
            np.reciprocal(deltaE, where=~np.isclose(deltaE, 0.0))
    wfs.kd.comm.sum(r_skvnm)
    return r_skvnm


def get_momentum_transitions(wfs, savetofile: bool = True) -> ArrayND:
    r"""
    Finds the momentum matrix elements:
    <nk|p|mk> = k \delta_nm - i <nk|\nabla|mk>

    Parameters
    ----------
    wfs
        LCAO WaveFunctions object
    savetofile: bool
        Determines whether matrix is written to the
        file mom_skvnm.npy (default=True)
    """
    assert wfs.bd.comm.size == 1
    assert wfs.mode == 'lcao'
    nbands = wfs.bd.nbands
    nspins = wfs.nspins
    nk = wfs.kd.nibzkpts
    gd = wfs.gd
    dtype = wfs.dtype
    ksl = wfs.ksl

    # print(wfs.kd.comm.size, wfs.gd.comm.size, wfs.bd.comm.size)

    mom_skvnm = np.zeros((nspins, nk, 3, nbands, nbands), dtype=complex)

    dThetadR_qvMM, _ = wfs.manytci.O_qMM_T_qMM(gd.comm, ksl.Mstart,
                                               ksl.Mstop, False,
                                               derivative=True)

    mome_skvnm = np.zeros((nspins, nk, 3, nbands, nbands), dtype=dtype)
    momd_skv = np.zeros((nspins, nk, 3), dtype=dtype)
    moma_skvnm = np.zeros((nspins, nk, 3, nbands, nbands), dtype=dtype)

    B_cv = 2.0 * np.pi * gd.icell_cv
    for kpt in wfs.kpt_u:
        C_nM = kpt.C_nM
        for v in range(3):
            dThetadRv_MM = dThetadR_qvMM[kpt.q, v]
            nabla_nn = -(C_nM.conj() @ dThetadRv_MM.conj() @ C_nM.T)
            gd.comm.sum(nabla_nn)
            mome_skvnm[kpt.s, kpt.k, v] = nabla_nn

        # augmentation part
        moma_vnm = np.zeros((3, nbands, nbands), dtype=dtype)
        for a, P_ni in kpt.P_ani.items():
            nabla_iiv = wfs.setups[a].nabla_iiv
            moma_vnm += np.einsum('ni,ijv,mj->vnm',
                                  P_ni.conj(), nabla_iiv, P_ni, optimize=True)
        gd.comm.sum(moma_vnm)
        moma_skvnm[kpt.s, kpt.k] = moma_vnm

        # diagonal term
        k_v = np.dot(wfs.kd.ibzk_kc[kpt.k], B_cv)
        momd_skv[kpt.s, kpt.k, :] = k_v

    mom_skvnm = - 1j * (mome_skvnm + moma_skvnm)
    # Extract a view of the diagonal elements out
    # Einsum with no optimization since that is empirically faster here
    mom_diag = np.einsum('skvnn->skvn', mom_skvnm)
    # Add the diagonal term. Since mom_diag is a view, this changes mom_skvnm
    mom_diag += momd_skv[..., None]
    wfs.kd.comm.sum(mom_skvnm)

    if world.rank == 0 and savetofile:
        np.save('mom_skvnm.npy', mom_skvnm)
    return mom_skvnm
