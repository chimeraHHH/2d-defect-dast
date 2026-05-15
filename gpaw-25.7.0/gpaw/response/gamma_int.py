from collections.abc import Sequence

import numpy as np
from ase.dft.kpoints import monkhorst_pack


class GammaIntegral(Sequence):
    def __init__(self, coulomb, qpd):
        self.coulomb = coulomb
        self.qpd = qpd
        self.integral_domain = GammaIntegralDomain(
            coulomb.truncation, coulomb.kd, qpd)

    def __len__(self):
        return len(self.integral_domain)

    def __getitem__(self, q):
        qweight, qf_v = self.integral_domain[q]
        sqrtV_G = self.coulomb.sqrtV(qpd=self.qpd, q_v=qf_v)

        def chi0_mapping(chi0_GG, chi0_vv, chi0_xvG):
            out_GG = chi0_GG.copy()
            out_GG[0, :] = qf_v @ chi0_xvG[0]
            out_GG[:, 0] = qf_v @ chi0_xvG[1]
            out_GG[0, 0] = qf_v @ chi0_vv @ qf_v
            return out_GG

        return qweight, sqrtV_G, chi0_mapping


class GammaIntegralDomain(Sequence):
    def __init__(self, truncation, kd, qpd):
        N = 4
        N_c = np.array([N, N, N])
        if truncation is not None:
            # Only average periodic directions if trunction is used
            N_c[kd.N_c == 1] = 1
        qf_qc = monkhorst_pack(N_c) / kd.N_c
        qf_qc *= 1.0e-6
        # XXX previously symmetry was used in Gamma integrator.
        # This was not correct, as explained in #709.
        self.qweight = 1. / np.prod(N_c)
        self.qf_qv = 2 * np.pi * (qf_qc @ qpd.gd.icell_cv)

    def __len__(self):
        return len(self.qf_qv)

    def __getitem__(self, q):
        return self.qweight, self.qf_qv[q]
