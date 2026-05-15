from __future__ import annotations

import numpy as np
from ase.units import pi
from gpaw.response.df import DielectricFunction
try:
    from qeh.bb_calculator.chicalc import ChiCalc, QPoint
except ImportError:
    class ChiCalc():  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError('qeh not installed, \
                               or is too old.')

    class QPoint():  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError('qeh not installed, \
                               or is too old.')


class QEHChiCalc(ChiCalc):
    qdim = {'x': 0, 'y': 1}

    def __init__(self,
                 df: DielectricFunction,
                 qinf_rel: float = 1e-6,
                 direction: str = 'x'):

        ''' GPAW superclass for interfacing with QEH
        building block calculations.

        Parameters
        ----------
        df : DielectricFunction
            the dielectric function calculator
        qinf_rel : float
            the position of the gamma q-point,
            relative to the first non-gamma q-point,
            necessary due to the undefined nature of
            chi_wGG in the gamma q-point.
        direction : str (either 'x' or 'y')
            the direction of the q-grid in terms of
            the reciprocal lattice vectors.
        '''

        self.df = df  # DielectricFunctionCalculator
        self.L = df.gs.gd.cell_cv[2, 2]
        self.omega_w = self.df.chi0calc.wd.omega_w
        self.direction = direction
        self.context = self.df.context
        self.qinf_rel = qinf_rel

        if not (self.df.gs.kd.ibzk_kc == [0, 0, 0]).any():
            raise ValueError("Only Gamma-centered \
                k-point grids are supported")

        qdir = self.qdim[self.direction]
        kd = self.df.gs.kd
        self.Nk = kd.N_c[qdir]

        super().__init__()

    def get_q_grid(self, q_max: float | None = None):
        # First get q-points on the grid
        qdir = self.qdim[self.direction]

        gd = self.df.gs.gd
        icell_cv = gd.icell_cv

        # Make q-grid
        q_qc = np.zeros([self.Nk, 3], dtype=float)
        q_qc[:, qdir] = np.linspace(0, 1, self.Nk,
                                    endpoint=False)

        # Avoid Gamma-point
        q_qc[0] = q_qc[1] * self.qinf_rel

        q_qv = q_qc @ icell_cv * 2 * pi

        # Filter the q-points with q_max
        if q_max is not None:
            q_mask = np.linalg.norm(q_qv, axis=1) <= q_max
            q_qc = q_qc[q_mask]
            q_qv = q_qv[q_mask]

        # Make list of QPoints for calculation
        Q_q = [QPoint(q_c=q_c, q_v=q_v,
                      P_rv=self.determine_P_rv(q_c, q_max))
               for q_c, q_v in zip(q_qc, q_qv)]

        return Q_q

    def determine_P_rv(self, q_c: np.ndarray, q_max: float | None):
        """
        Determine the reciprocal space vectors P_rv that correspond
        to unfold the given q-point out of the 1st BZ
        given a q-point and the maximum q-value.

        Parameters:
            q_c (np.ndarray): The q-point in reciprocal space.
            q_max (float | None): The maximum q-value.

        Returns:
            list: A list of reciprocal space vectors P_rv.
        """
        if q_max is not None:
            icell_cv = self.df.gs.gd.icell_cv
            qdir = self.qdim[self.direction]
            qc_max = q_max / np.linalg.norm(icell_cv[qdir] * 2 * pi)
            P_rv = [icell_cv[qdir] * 2 * pi * i
                    for i in range(0, int(qc_max - q_c[qdir]) + 1)]
            return P_rv
        else:
            return [np.array([0, 0, 0])]

    def get_z_grid(self):
        r = self.df.gs.gd.get_grid_point_coordinates()
        return r[2, 0, 0, :]

    def get_chi_wGG(self, qpoint: QPoint):
        if np.linalg.norm(qpoint.q_c) <= (2 * self.qinf_rel / self.Nk):
            chi0_dyson_eqs = self.df.get_chi0_dyson_eqs([0, 0, 0],
                                                        truncation='2D')
            qpd, chi_wGG, wblocks = chi0_dyson_eqs.rpa_density_response(
                qinf_v=qpoint.q_v, direction=qpoint.q_v)
        else:
            chi0_dyson_eqs = self.df.get_chi0_dyson_eqs(qpoint.q_c,
                                                        truncation='2D')
            qpd, chi_wGG, wblocks = chi0_dyson_eqs.rpa_density_response()

        G_Gv = qpd.get_reciprocal_vectors(add_q=False)

        return chi_wGG, G_Gv, wblocks

    def get_atoms(self):
        return self.df.gs.atoms
