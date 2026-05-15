import numpy as np

from gpaw.mpi import serial_comm
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.lfc import PWLFC
from gpaw.hybrids.kpts import PWKPoint, RSKPoint, to_real_space
from gpaw.utilities.blas import mmm


def apply1(kpt, Htpsit_xG, wfs, coulomb, sym, paw):
    kd = wfs.kd
    kpts = [PWKPoint(kpt.psit,
                     kpt.projections,
                     kpt.f_n / kpt.weight,  # scale to [0, 1] range
                     kd.ibzk_kc[kpt.k],
                     kd.weight_k[kpt.k])
            for kpt in wfs.kpt_u[kpt.s::wfs.nspins]]
    evv, evc, ekin, v_knG = calculate(kpts, wfs, paw, sym, coulomb)
    return evc, evv, ekin, v_knG


def calculate(kpts, wfs, paw, sym, coulomb):
    pd = kpts[0].psit.pd
    gd = pd.gd.new_descriptor(comm=serial_comm)
    kd = wfs.kd
    comm = wfs.world
    nbands = len(kpts[0].psit.array)
    shapes = [(nbands, len(Delta_iiL))
              for Delta_iiL in paw.Delta_aiiL]
    v_kani = [{a: np.zeros(shape, pd.dtype)
               for a, shape in enumerate(shapes)}
              for _ in range(len(kpts))]
    v_knG = [k.psit.pd.zeros(nbands, global_array=True, q=k.psit.kpt)
             for k in kpts]

    exxvv = 0.0
    ekin = 0.0
    for i1, i2, s, k1, k2, count in sym.pairs(kpts, wfs, wfs.spos_ac):
        q_c = k2.k_c - k1.k_c
        qd = KPointDescriptor([-q_c])

        pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
        ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
        ghat.set_positions(wfs.spos_ac)

        v1_nG = v_knG[i1]
        v1_ani = v_kani[i1]
        v2_nG = v_knG[i2]
        v2_ani = v_kani[i2]

        v_G = coulomb.get_potential(pd12)
        assert i1 == kpts[i1].psit.kpt
        assert i2 == kpts[i2].psit.kpt
        e_nn = calculate_exx_for_pair(k1, k2, ghat, v_G,
                                      kpts[i1].psit.pd,
                                      kpts[i2].psit.pd,
                                      kpts[i1].psit.kpt,
                                      kpts[i2].psit.kpt,
                                      k1.f_n,
                                      k2.f_n,
                                      s,
                                      count,
                                      v1_nG, v1_ani,
                                      v2_nG, v2_ani,
                                      wfs, sym, paw)

        e_nn *= count
        e = k1.f_n.dot(e_nn).dot(k2.f_n) / kd.nbzkpts
        exxvv -= 0.5 * e
        ekin += e

    exxvc = 0.0
    for i, kpt in enumerate(kpts):
        for a, VV_ii in paw.VV_aii.items():
            P_ni = kpt.proj[a]
            vv_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), VV_ii, P_ni).real
            vc_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), paw.VC_aii[a], P_ni).real
            exxvv -= vv_n.dot(kpt.f_n) * kpt.weight
            exxvc -= vc_n.dot(kpt.f_n) * kpt.weight

    w_knG = {}
    G1 = comm.rank * pd.maxmyng
    G2 = (comm.rank + 1) * pd.maxmyng
    for v_nG, v_ani, kpt in zip(v_knG, v_kani, kpts):
        comm.sum(v_nG)
        w_nG = v_nG[:, G1:G2].copy()
        w_knG[len(w_knG)] = w_nG
        for v_ni in v_ani.values():
            comm.sum(v_ni)
        v1_ani = {}
        for a, VV_ii in paw.VV_aii.items():
            P_ni = kpt.proj[a]
            v_ni = P_ni.dot(paw.VC_aii[a] + 2 * VV_ii)
            v1_ani[a] = v_ani[a] - v_ni
            ekin += (np.einsum('n, ni, ni',
                               kpt.f_n, P_ni.conj(), v_ni).real *
                     kpt.weight)
        wfs.pt.add(w_nG, v1_ani, kpt.psit.kpt)

    return (comm.sum_scalar(exxvv),
            comm.sum_scalar(exxvc),
            comm.sum_scalar(ekin),
            w_knG)


def calculate_exx_for_pair(k1,
                           k2,
                           ghat,
                           v_G,
                           pd1, pd2,
                           index1, index2,
                           f1_n, f2_n,
                           s,
                           count,
                           v1_nG,
                           v1_ani,
                           v2_nG,
                           v2_ani,
                           wfs,
                           sym,
                           paw,
                           F_av=None):
    kd = wfs.kd
    comm = wfs.world
    factor = 1.0 / kd.nbzkpts

    N1 = len(k1.u_nR)
    N2 = len(k2.u_nR)

    size = comm.size
    rank = comm.rank

    Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                        k1.proj[a],
                        Delta_iiL,
                        k2.proj[a].conj(),
                        optimize=True)
              for a, Delta_iiL in enumerate(paw.Delta_aiiL)]

    if v2_nG is not None:
        T, T_a, cc = sym.symmetry_operation(s, wfs, inverse=True)

    if k1 is k2:
        n2max = (N1 + size - 1) // size
    else:
        n2max = N2

    e_nn = np.zeros((N1, N2))
    rho_nG = ghat.pd.empty(n2max, k1.u_nR.dtype)
    vrho_nG = ghat.pd.empty(n2max, k1.u_nR.dtype)

    f_GI = ghat.expand()

    for n1, u1_R in enumerate(k1.u_nR):
        if k1 is k2:
            B = (N1 - n1 + size - 1) // size
            n20 = 0
            n2a = min(n1 + rank * B, N2)
            n2b = min(n2a + B, N2)
        else:
            B = (N1 + size - 1) // size
            n20 = min(B * rank, N1)
            n2a = 0
            n2b = N2

        for n2, rho_G in enumerate(rho_nG[:n2b - n2a], n2a):
            rho_G[:] = ghat.pd.fft(u1_R * k2.u_nR[n2].conj())

        add(ghat, rho_nG[:n2b - n2a],
            {a: Q_nnL[n1, n2a:n2b]
             for a, Q_nnL in enumerate(Q_annL)},
            f_GI)
        for n2, rho_G in enumerate(rho_nG[:n2b - n2a], n2a):
            vrho_G = v_G * rho_G
            e = ghat.pd.integrate(rho_G, vrho_G).real
            e_nn[n1, n2] = e
            if k1 is k2:
                e_nn[n2, n1] = e
            vrho_nG[n2 - n2a] = vrho_G

            if v1_nG is not None:
                vrho_R = ghat.pd.ifft(vrho_G)
                if v2_nG is None:
                    assert k1 is not k2
                    v1_nG[n1] -= f2_n[n2] * factor * pd1.fft(
                        vrho_R * k2.u_nR[n2], index1, local=True)
                else:
                    x = factor * count / 2
                    if k1 is k2 and n1 != n2:
                        x *= 2
                    x1 = x / (kd.weight_k[index1] * kd.nbzkpts)
                    x2 = x / (kd.weight_k[index2] * kd.nbzkpts)
                    v1_nG[n1] -= f2_n[n2] * x1 * pd1.fft(
                        vrho_R * k2.u_nR[n2], index1, local=True)
                    v2_nG[n2 + n20] -= f1_n[n1] * x2 * pd2.fft(
                        T(vrho_R.conj() * u1_R), index2,
                        local=True)

        if v1_nG is not None and v2_nG is None:
            for a, v_nL in integrate(ghat, vrho_nG[:n2b - n2a], f_GI):
                v_iin = paw.Delta_aiiL[a].dot(v_nL.T)
                v1_ani[a][n1] -= np.einsum('ijn, nj, n -> i',
                                           v_iin,
                                           k2.proj[a][n2a:n2b],
                                           f2_n[n2a:n2b] * factor)

        if v1_nG is not None and v2_nG is not None:
            x = factor * count / kd.nbzkpts / 2
            x1 = x / kd.weight_k[index1]
            x2 = x / kd.weight_k[index2]
            if k1 is k2:
                x1 *= 2
                x2 *= 2

            for a, v_nL in integrate(ghat, vrho_nG[:n2b - n2a], f_GI):
                if k1 is k2 and n2a <= n1 < n2b:
                    v_nL[n1 - n2a] *= 0.5
                v_iin = paw.Delta_aiiL[a].dot(v_nL.T)
                v1_ani[a][n1] -= np.einsum('ijn, nj, n -> i',
                                           v_iin,
                                           k2.proj[a][n2a:n2b],
                                           f2_n[n2a:n2b] * x1)
                b, S_c, U_ii = T_a[a]
                v_ni = np.einsum('ijn, j, ik -> nk',
                                 v_iin.conj(),
                                 k1.proj[b][n1],
                                 U_ii)
                if v_nL.dtype == complex:
                    v_ni *= np.exp(2j * np.pi * k2.k_c.dot(S_c))
                    if cc:
                        v_ni = v_ni.conj()
                v2_ani[a][n20 + n2a:n20 + n2b] -= v_ni * f1_n[n1] * x2

    return e_nn * factor


def add(ghat, a_xG, c_axi, f_GI):
    c_xI = np.empty(a_xG.shape[:-1] + (ghat.nI,), ghat.pd.dtype)
    for a, I1, I2 in ghat.my_indices:
        c_xI[..., I1:I2] = c_axi[a] * ghat.eikR_qa[0][a].conj()
    nx = np.prod(c_xI.shape[:-1], dtype=int)
    c_xI = c_xI.reshape((nx, ghat.nI))
    a_xG = a_xG.reshape((nx, a_xG.shape[-1])).view(ghat.pd.dtype)
    mmm(1.0 / ghat.pd.gd.dv, c_xI, 'N', f_GI, 'T', 1.0, a_xG)


def integrate(ghat, a_xG, f_GI):
    c_xI = np.zeros(a_xG.shape[:-1] + (ghat.nI,), ghat.pd.dtype)

    nx = np.prod(c_xI.shape[:-1], dtype=int)
    b_xI = c_xI.reshape((nx, ghat.nI))
    a_xG = a_xG.reshape((nx, a_xG.shape[-1]))

    alpha = 1.0 / ghat.pd.gd.N_c.prod()
    if ghat.pd.dtype == float:
        alpha *= 2
        a_xG = a_xG.view(float)
        f_GI[0] *= 0.5
    else:
        f_GI.imag[:] = -f_GI.imag
    mmm(alpha, a_xG, 'N', f_GI, 'N', 0.0, b_xI)
    if ghat.pd.dtype == complex:
        f_GI.imag[:] = -f_GI.imag
    else:
        f_GI[0] *= 2.0
    for a, I1, I2 in ghat.my_indices:
        yield a, ghat.eikR_qa[0][a] * c_xI[..., I1:I2]


def apply2(kpt, psit_xG, Htpsit_xG, wfs, coulomb, sym, paw):
    kd = wfs.kd

    psit = kpt.psit.new(buf=psit_xG)
    P = kpt.projections.new()
    psit.matrix_elements(wfs.pt, out=P)

    kpt1 = PWKPoint(psit,
                    P,
                    kpt.f_n + np.nan,
                    kd.ibzk_kc[kpt.k],
                    np.nan)

    kpts2 = [PWKPoint(kpt.psit,
                      kpt.projections,
                      kpt.f_n / kpt.weight,  # scale to [0, 1] range
                      kd.ibzk_kc[kpt.k],
                      kd.weight_k[kpt.k])
             for kpt in wfs.kpt_u[kpt.s::wfs.nspins]]
    v_nG = calculate2(kpt1, kpts2, wfs, paw, sym, coulomb)
    return v_nG


def calculate2(kpt1, kpts2, wfs, paw, sym, coulomb):
    pd = kpt1.psit.pd
    gd = pd.gd.new_descriptor(comm=serial_comm)
    kd = wfs.kd
    comm = wfs.world
    nbands = len(kpt1.psit.array)
    shapes = [(nbands, len(Delta_iiL))
              for Delta_iiL in paw.Delta_aiiL]
    v_ani = {a: np.zeros(shape, pd.dtype)
             for a, shape in enumerate(shapes)}
    v_nG = kpt1.psit.pd.zeros(nbands, global_array=True, q=kpt1.psit.kpt)

    u1_nR = to_real_space(kpt1.psit)
    proj1 = kpt1.proj.broadcast()
    k1 = RSKPoint(u1_nR,
                  proj1,
                  kpt1.f_n,
                  kpt1.k_c,
                  kpt1.weight)

    N2 = len(kpts2[0].psit.array)
    nsym = len(kd.symmetry.op_scc)

    size = comm.size
    rank = comm.rank
    B = (N2 + size - 1) // size
    na = min(B * rank, N2)
    nb = min(na + B, N2)
    for i2, kpt2 in enumerate(kpts2):
        u2_nR = to_real_space(kpt2.psit, na, nb)
        k0 = RSKPoint(u2_nR,
                      kpt2.proj.broadcast().view(na, nb),
                      kpt2.f_n[na:nb],
                      kpt2.k_c,
                      kpt2.weight)
        for k, i in enumerate(kd.bz2ibz_k):
            if i != i2:
                continue
            s = kd.sym_k[k] + kd.time_reversal_k[k] * nsym
            k2 = sym.apply_symmetry(s, k0, wfs, wfs.spos_ac)
            q_c = k2.k_c - k1.k_c
            qd = KPointDescriptor([-q_c])

            pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
            ghat.set_positions(wfs.spos_ac)

            v_G = coulomb.get_potential(pd12)
            calculate_exx_for_pair(k1, k2, ghat, v_G,
                                   kpt1.psit.pd,
                                   kpts2[i2].psit.pd,
                                   kpt1.psit.kpt,
                                   kpts2[i2].psit.kpt,
                                   k1.f_n,
                                   k2.f_n,
                                   s,
                                   1.0,
                                   v_nG, v_ani,
                                   None, None,
                                   wfs, sym, paw)

    G1 = comm.rank * pd.maxmyng
    G2 = (comm.rank + 1) * pd.maxmyng
    comm.sum(v_nG)
    w_nG = v_nG[:, G1:G2].copy()
    for v_ni in v_ani.values():
        comm.sum(v_ni)
    v1_ani = {}
    for a, VV_ii in paw.VV_aii.items():
        P_ni = kpt1.proj[a]
        v_ni = P_ni.dot(paw.VC_aii[a] + 2 * VV_ii)
        v1_ani[a] = v_ani[a] - v_ni
    wfs.pt.add(w_nG, v1_ani, kpt1.psit.kpt)

    return w_nG
