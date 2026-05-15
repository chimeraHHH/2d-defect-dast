import numpy as np

from ase.units import Hartree, Bohr

from gpaw.directmin.tools import get_n_occ
from gpaw.utilities import unpack_hermitian
from gpaw.xc.hybrid import HybridXCBase


def calculate_forces(wfs, dens, ham, log=None):
    """Return the atomic forces."""

    assert not isinstance(ham.xc, HybridXCBase)
    assert not ham.xc.name.startswith('GLLB')

    func_name = None
    if hasattr(wfs.eigensolver, 'dm_helper'):
        func_name = getattr(wfs.eigensolver.dm_helper.func, 'name', None)
    elif hasattr(wfs.eigensolver, 'odd'):
        func_name = getattr(wfs.eigensolver.odd, 'name', None)
    if func_name == 'PZ-SIC':
        if wfs.mode == 'fd' or wfs.mode == 'pw':
            return calculate_forces_using_non_diag_lagr_matrix(
                wfs, dens, ham, log)
        elif wfs.mode == 'lcao':
            for kpt in wfs.kpt_u:
                kpt.rho_MM = \
                    wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM)

    natoms = len(wfs.setups)

    # Force from projector functions (and basis set):
    F_wfs_av = np.zeros((natoms, 3))
    wfs.calculate_forces(ham, F_wfs_av)
    wfs.gd.comm.sum(F_wfs_av, 0)
    F_ham_av = np.zeros((natoms, 3))

    try:
        # ODD functionals need force corrections for each spin
        correction = ham.xc.setup_force_corrections
    except AttributeError:
        pass
    else:
        correction(F_ham_av)

    ham.calculate_forces(dens, F_ham_av)

    F_av = F_ham_av + F_wfs_av
    wfs.world.broadcast(F_av, 0)

    if func_name == 'PZ-SIC' and wfs.mode == 'lcao':
        F_av += wfs.eigensolver.dm_helper.func.get_odd_corrections_to_forces(
            wfs, dens)
        for kpt in wfs.kpt_u:
            # need to re-set rho_MM otherwise it will be used
            # it's probably better to in wfs.reset, but
            # when position changes wfs.reset is not called
            kpt.rho_MM = None

    F_av = wfs.kd.symmetry.symmetrize_forces(F_av)

    if log:
        log('\nForces in eV/Ang:')
        c = Hartree / Bohr
        for a, setup in enumerate(wfs.setups):
            log('%3d %-2s %10.5f %10.5f %10.5f' %
                ((a, setup.symbol) + tuple(F_av[a] * c)))
        log()

    return F_av


def calculate_forces_using_non_diag_lagr_matrix(wfs, dens, ham, log=None):

    natoms = len(wfs.setups)
    F_wfs_av = np.zeros((natoms, 3))
    esolv = wfs.eigensolver

    grad_knG = esolv.get_gradients_2(ham, wfs)
    if 'SIC' in esolv.odd.name:
        for kpt in wfs.kpt_u:
            esolv.odd.get_energy_and_gradients_kpt(
                wfs, kpt, grad_knG, esolv.iloop.U_k, add_grad=True)
    for kpt in wfs.kpt_u:
        k = esolv.n_kps * kpt.s + kpt.q
        n_occ = get_n_occ(kpt)[0]

        lamb = wfs.integrate(
            kpt.psit_nG[:n_occ], grad_knG[k][:n_occ], True)

        P_ani = kpt.P_ani
        dP_aniv = wfs.pt.dict(n_occ, derivative=True)
        wfs.pt.derivative(kpt.psit_nG[:n_occ], dP_aniv, kpt.q)
        dH_asp = ham.dH_asp

        for a, dP_niv in dP_aniv.items():
            dP_niv = dP_niv.conj()
            dO_ii = wfs.setups[a].dO_ii
            P_ni = P_ani[a][:n_occ]
            dS_nkv = np.einsum('niv,ij,kj->nkv', dP_niv, dO_ii, P_ni)
            dS_nkv = \
                (dS_nkv + np.transpose(dS_nkv, (1, 0, 2)).conj())
            # '-' becuase there is extra minus in 'pt derivative'
            F_wfs_av[a] -= np.einsum('kn,nkv->v', lamb, dS_nkv).real
            dH_ii = unpack_hermitian(dH_asp[a][kpt.s])
            dh_nv = np.einsum('niv,ij,nj->nv', dP_niv, dH_ii, P_ni)
            # '+' becuase there is extra minus in 'pt derivative'
            F_wfs_av[a] += np.einsum('n,nv->v', kpt.f_n[:n_occ],
                                     2.0 * dh_nv.real)

        if 'SIC' in esolv.odd.name:
            esolv.odd.get_odd_corrections_to_forces(F_wfs_av,
                                                    wfs,
                                                    kpt)

    wfs.kd.comm.sum(F_wfs_av, 0)
    wfs.gd.comm.sum(F_wfs_av, 0)

    F_ham_av = np.zeros((natoms, 3))
    ham.calculate_forces(dens, F_ham_av)
    F_av = F_ham_av + F_wfs_av
    wfs.world.broadcast(F_av, 0)

    F_av = wfs.kd.symmetry.symmetrize_forces(F_av)

    if log:
        log('\nForces in eV/Ang:')
        c = Hartree / Bohr
        for a, setup in enumerate(wfs.setups):
            log('%3d %-2s %10.5f %10.5f %10.5f' %
                ((a, setup.symbol) + tuple(F_av[a] * c)))
        log()

    return F_av
