import numpy as np


class Q0Correction:
    def __init__(self, cell_cv, bzk_kc, N_c):
        self.cell_cv = cell_cv
        self.bzk_kc = bzk_kc
        self.N_c = N_c

        # Check that basic assumptions of cell and k-grid
        # for Q0Correction are fulfilled
        assert N_c[2] == 1  # z-axis is non periodic direction
        eps = 1e-14
        assert (abs(cell_cv[:2, 2]).max() < eps and
                abs(cell_cv[2, :2]).max() < eps and
                cell_cv[2, 2] > 0)

        # Hardcoded crap?
        x0density = 0.1  # ? 0.01

        # Generate numerical q-point grid
        self.rcell_cv = 2 * np.pi * np.linalg.inv(cell_cv).T

        # Prepare stuff
        self.q0cell_cv = np.array([1, 1, 1])**0.5 * self.rcell_cv / self.N_c
        L = cell_cv[2, 2]
        q0density = 2. / L * x0density
        npts_c = np.ceil(np.sum(self.q0cell_cv**2, axis=1)**0.5 /
                         q0density).astype(int)
        npts_c[2] = 1
        npts_c += (npts_c + 1) % 2
        self.npts_c = npts_c

    def add_q0_correction(self, qpd, W_GG, einv_GG,
                          chi0_xvG, chi0_vv, sqrtV_G):
        from ase.dft import monkhorst_pack
        qpts_qc = self.bzk_kc
        pi = np.pi
        L = self.cell_cv[2, 2]

        vc_G0 = sqrtV_G[1:]**2

        B_GG = einv_GG[1:, 1:]
        u_v0G = vc_G0[np.newaxis, :]**0.5 * chi0_xvG[0, :, 1:]
        u_vG0 = vc_G0[np.newaxis, :]**0.5 * chi0_xvG[1, :, 1:]
        U_vv = -chi0_vv
        a_v0G = -np.dot(u_v0G, B_GG)
        a_vG0 = -np.dot(u_vG0, B_GG.T)
        A_vv = U_vv - np.dot(a_v0G, u_vG0.T)
        S_v0G = a_v0G
        S_vG0 = a_vG0
        L_vv = A_vv

        # Get necessary G vectors.
        G_Gv = qpd.get_reciprocal_vectors(add_q=False)[1:]
        G_Gv += np.array([1e-14, 1e-14, 0])
        G2_G = np.sum(G_Gv**2, axis=1)
        Gpar_G = np.sum(G_Gv[:, 0:2]**2, axis=1)**0.5

        # There is still a lot of stuff here,
        # which could go to the constructor! XXX
        iq = np.argmin(np.sum(qpts_qc**2, axis=1))
        assert np.allclose(qpts_qc[iq], 0)
        q0vol = abs(np.linalg.det(self.q0cell_cv))

        qpts_qc = monkhorst_pack(self.npts_c)
        qgamma = np.argmin(np.sum(qpts_qc**2, axis=1))

        qpts_qv = np.dot(qpts_qc, self.q0cell_cv)
        qpts_q = np.sum(qpts_qv**2, axis=1)**0.5
        qpts_q[qgamma] = 1e-14
        qdir_qv = qpts_qv / qpts_q[:, np.newaxis]
        qdir_qvv = qdir_qv[:, :, np.newaxis] * qdir_qv[:, np.newaxis, :]
        nq = len(qpts_qc)
        q0area = q0vol / self.q0cell_cv[2, 2]
        dq0 = q0area / nq
        dq0rad = (dq0 / pi)**0.5
        R = L / 2.
        x0area = q0area * R**2
        dx0rad = dq0rad * R

        exp_q = 4 * pi * (1 - np.exp(-qpts_q * R))
        dv_G = ((pi * L * G2_G * np.exp(-Gpar_G * R) * np.cos(G_Gv[:, 2] * R) -
                 4 * pi * Gpar_G * (1 - np.exp(-Gpar_G * R) *
                                    np.cos(G_Gv[:, 2] * R))) /
                (G2_G**1.5 * Gpar_G *
                 (4 * pi * (1 - np.exp(-Gpar_G * R) *
                            np.cos(G_Gv[:, 2] * R)))**0.5))

        dv_Gv = dv_G[:, np.newaxis] * G_Gv

        # Add corrections
        W_GG[:, 0] = 0.0
        W_GG[0, :] = 0.0

        A_q = np.sum(qdir_qv * np.dot(qdir_qv, L_vv), axis=1)
        frac_q = 1. / (1 + exp_q * A_q)

        # HEAD:
        w00_q = -(exp_q / qpts_q)**2 * A_q * frac_q
        w00_q[qgamma] = 0.0
        W_GG[0, 0] = w00_q.sum() / nq
        Axy = 0.5 * (L_vv[0, 0] + L_vv[1, 1])  # in-plane average
        a0 = 4 * pi * Axy + 1

        W_GG[0, 0] += -((a0 * dx0rad - np.log(a0 * dx0rad + 1)) /
                        a0**2 / x0area * 2 * np.pi * (2 * pi * L)**2 * Axy)

        # WINGS:
        u_q = -exp_q / qpts_q * frac_q
        W_GG[1:, 0] = 1. / nq * np.dot(
            np.sum(qdir_qv * u_q[:, np.newaxis], axis=0),
            S_vG0 * sqrtV_G[np.newaxis, 1:])

        W_GG[0, 1:] = 1. / nq * np.dot(
            np.sum(qdir_qv * u_q[:, np.newaxis], axis=0),
            S_v0G * sqrtV_G[np.newaxis, 1:])

        # BODY:
        # Constant corrections:
        W_GG[1:, 1:] += 1. / nq * sqrtV_G[1:, None] * sqrtV_G[None, 1:] * \
            np.tensordot(S_v0G, np.dot(S_vG0.T,
                                       np.sum(-qdir_qvv *
                                              exp_q[:, None, None] *
                                              frac_q[:, None, None],
                                              axis=0)), axes=(0, 1))
        u_vvv = np.tensordot(u_q[:, None] * qpts_qv, qdir_qvv, axes=(0, 0))
        # Gradient corrections:
        W_GG[1:, 1:] += 1. / nq * np.sum(
            dv_Gv[:, :, None] * np.tensordot(
                S_v0G, np.tensordot(u_vvv, S_vG0 * sqrtV_G[None, 1:],
                                    axes=(2, 0)), axes=(0, 1)), axis=1)
