# from typing import Tuple

import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import serial_comm, broadcast
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.lfc import PWLFC
from .kpts import get_kpt


def calculate_forces(wfs, coulomb, sym, paw_s, ftol=1e-9) -> np.ndarray:
    kd = wfs.kd
    nspins = wfs.nspins

    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.kpt_u)
    nocc = kd.comm.max_scalar(int(nocc))

    dPdR_skaniv = {(kpt.s, kpt.k): wfs.pt.derivative(kpt.psit_nG[:nocc],
                                                     q=kpt.k)
                   for kpt in wfs.kpt_u}

    natoms = len(wfs.spos_ac)
    F_av = np.zeros((natoms, 3))

    for spin in range(nspins):
        kpts = []
        for k in range(kd.nibzkpts):
            kpt = get_kpt(wfs, k, spin, 0, nocc)
            rank_a = kpt.proj.atom_partition.rank_a
            dPdR_aniv = [
                broadcast(dPdR_skaniv[(spin, k)].get(a), rank_a[a], wfs.world)
                for a in range(natoms)]
            kpt.dPdR_aniv = dPdR_aniv
            kpts.append(kpt)
        forces(kpts, paw_s[spin],
               wfs, sym, coulomb, F_av)

    return F_av / nspins


def forces(kpts, paw, wfs, sym, coulomb, F_av):
    pd = kpts[0].psit.pd
    gd = pd.gd.new_descriptor(comm=serial_comm)
    comm = wfs.world
    for i1, i2, s, k1, k2, count in sym.pairs(kpts, wfs, wfs.spos_ac):
        q_c = k2.k_c - k1.k_c
        qd = KPointDescriptor([-q_c])

        pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
        ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
        ghat.set_positions(wfs.spos_ac)

        v_G = coulomb.get_potential(pd12)
        f_av = calculate_exx_for_pair(k1, k2,
                                      ghat, v_G, comm,
                                      paw, count)
        F_av += f_av * (1 / wfs.kd.nbzkpts**2)

    for a, v_ii in paw.VV_aii.items():
        vv_ii = 8 * v_ii + 4 * paw.VC_aii[a]
        for kpt in kpts:
            F_av[a] -= np.einsum('ij, niv, nj, n -> v',
                                 vv_ii,
                                 kpt.dPdR_aniv[a].conj(),
                                 kpt.proj[a],
                                 kpt.f_n).real * kpt.weight


def calculate_exx_for_pair(k1,
                           k2,
                           ghat,
                           v_G,
                           comm,
                           paw,
                           count) -> np.ndarray:

    N1 = len(k1.u_nR)
    N2 = len(k2.u_nR)
    size = comm.size
    rank = comm.rank

    Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                        k1.proj[a],
                        Delta_iiL,
                        k2.proj[a].conj())
              for a, Delta_iiL in enumerate(paw.Delta_aiiL)]

    if k1 is k2:
        n2max = (N1 + size - 1) // size
    else:
        n2max = N2

    rho_nG = ghat.pd.empty(n2max, k1.u_nR.dtype)
    F_av = np.zeros((len(Q_annL), 3))

    for n1, u1_R in enumerate(k1.u_nR):
        if k1 is k2:
            B = (N1 + size - 1) // size
            n2a = min(rank * B, N2)
            n2b = min(n2a + B, N2)
        else:
            n2a = 0
            n2b = N2
        ff_n = k1.f_n[n1] * k2.f_n[n2a:n2b] * 2 * count

        for n2, rho_G in enumerate(rho_nG[:n2b - n2a], n2a):
            rho_G[:] = ghat.pd.fft(u1_R * k2.u_nR[n2].conj())

        ghat.add(rho_nG[:n2b - n2a],
                 {a: Q_nnL[n1, n2a:n2b]
                  for a, Q_nnL in enumerate(Q_annL)})

        vrho_nG = rho_nG[:n2b - n2a]
        vrho_nG *= v_G

        for a, v_nLv in ghat.derivative(vrho_nG).items():
            F_av[a] -= np.einsum('n, nL, nLv -> v',
                                 ff_n,
                                 Q_annL[a][n1, n2a:n2b].conj(),
                                 v_nLv).real

        for a, v_nL in ghat.integrate(vrho_nG).items():
            v_iin = paw.Delta_aiiL[a].dot(v_nL.T)
            F_av[a] -= np.einsum('ijn, iv, nj, n -> v',
                                 v_iin,
                                 k1.dPdR_aniv[a][n1].conj(),
                                 k2.proj[a][n2a:n2b],
                                 ff_n).real * 2

    return F_av
