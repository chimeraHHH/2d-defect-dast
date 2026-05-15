from math import pi
import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
import gpaw.fftw as fftw


class SingleQPWDescriptor(PWDescriptor):
    @staticmethod
    def from_q(q_c, ecut, gd, gammacentered=False):
        """Construct a plane wave descriptor for q_c with a given cutoff."""
        qd = KPointDescriptor([q_c])
        if not isinstance(ecut, dict):
            return SingleQPWDescriptor(ecut, gd, complex, qd,
                                       gammacentered=gammacentered)
        elif ecut['class'] is SingleCylQPWDescriptor:
            return SingleCylQPWDescriptor(gd=gd,
                                          kd=qd,
                                          gammacentered=gammacentered,
                                          dtype=complex,
                                          **ecut['kwargs'])
        else:
            raise NotImplementedError(
                f'Unrecognized QPW class: {ecut["class"]}')

    @property
    def q_c(self):
        return self.kd.bzk_kc[0]

    @property
    def optical_limit(self):
        return np.allclose(self.q_c, 0.0)

    def copy(self):
        return self.copy_with()

    def copy_with(self, ecut=None, gd=None, gammacentered=None):
        if ecut is None:
            ecut = self.ecut
        if gd is None:
            gd = self.gd
        if gammacentered is None:
            gammacentered = self.gammacentered

        return SingleQPWDescriptor.from_q(
            self.q_c, ecut, gd, gammacentered=gammacentered)


class SingleCylQPWDescriptor(SingleQPWDescriptor):
    def __init__(self, ecut_xy, ecut_z, gd, dtype=None, kd=None,
                 fftwflags=fftw.MEASURE, gammacentered=False):

        ecut0 = 0.5 * pi**2 / (gd.h_cv**2).sum(1).max()
        if ecut_xy is None:
            ecut_xy = 0.9999 * ecut0
        else:
            assert ecut_xy <= ecut0

        if ecut_z is None:
            ecut_z = 0.9999 * ecut0
        else:
            assert ecut_z <= ecut0

        self.ecut_z = ecut_z
        self.ecut_xy = ecut_xy

        super().__init__(ecut_xy, gd, dtype, kd,
                         fftwflags, gammacentered)

    def setup_pw_grid(self, i_Qc, Q_Q):
        ng_q = []
        Q_qG = []
        G2_qG = []
        for q, K_v in enumerate(self.K_qv):
            G2_Q = ((self.G_Qv + K_v)**2).sum(axis=1)
            if self.gammacentered:
                mask_Q = ((self.G_Qv[:, 0:2]**2).sum(axis=1)
                          <= 2 * self.ecut_xy) \
                    & ((self.G_Qv[:, 2]**2) <= 2 * self.ecut_z)
            else:
                G3_Q = ((self.G_Qv[:, 0:2] + K_v[0:2])**2).sum(axis=1)
                mask_Q = (G3_Q <= (2 * self.ecut_xy)) \
                    & ((self.G_Qv[:, 2]**2) <= (2 * self.ecut_z))

            if self.dtype == float:
                mask_Q &= ((i_Qc[:, 2] > 0) |
                           (i_Qc[:, 1] > 0) |
                           ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0)))
            Q_G = Q_Q[mask_Q]
            Q_qG.append(Q_G)
            G2_qG.append(G2_Q[Q_G])
            ng = len(Q_G)
            ng_q.append(ng)

        return ng_q, Q_qG, G2_qG

    def copy_with(self, ecut=None, gd=None, gammacentered=None):
        if ecut is None:
            ecut = self.ecut_xy
        if gd is None:
            gd = self.gd
        if gammacentered is None:
            gammacentered = self.gammacentered

        return SingleCylQPWDescriptor.from_q(
            self.q_c, ecut_xy=ecut, ecut_z=self.ecut_z,
            gd=gd, gammacentered=gammacentered)
