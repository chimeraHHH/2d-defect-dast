"""Wigner-Seitz truncated coulomb interaction.

See:

    Ravishankar Sundararaman and T. A. Arias:
    Phys. Rev. B 87, 165122 (2013)

    Regularization of the Coulomb singularity in exact exchange by
    Wigner-Seitz truncated interactions: Towards chemical accuracy
    in nontrivial systems
"""
from math import pi

import numpy as np
from scipy.special import erf
from ase.units import Bohr
from ase.utils import seterr

import gpaw.mpi as mpi
from gpaw.fftw import get_efficient_fft_size
from gpaw.grid_descriptor import GridDescriptor
from gpaw.core import PWDesc, UGDesc, PWArray
from gpaw.typing import Array1D


class WignerSeitzTruncatedCoulomb:
    def __init__(self, cell_cv, nk_c):
        self.nk_c = nk_c
        bigcell_cv = cell_cv * nk_c[:, np.newaxis]
        L_c = (np.linalg.inv(bigcell_cv)**2).sum(0)**-0.5

        self.rc = 0.5 * L_c.min()
        self.a = 5 / self.rc

        nr_c = [get_efficient_fft_size(2 * int(L * self.a * 3.0))
                for L in L_c]

        self.gd = GridDescriptor(nr_c, bigcell_cv, comm=mpi.serial_comm)
        v_ijk = self.gd.empty()

        pos_ijkv = self.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        corner_xv = np.dot(np.indices((2, 2, 2)).reshape((3, 8)).T, bigcell_cv)

        # Ignore division by zero (in 0,0,0 corner):
        with seterr(invalid='ignore'):
            # Loop over first dimension to avoid too large ndarrays.
            for pos_jkv, v_jk in zip(pos_ijkv, v_ijk):
                # Distances to the 8 corners:
                d_jkxv = pos_jkv[:, :, np.newaxis] - corner_xv
                r_jk = (d_jkxv**2).sum(axis=3).min(2)**0.5
                v_jk[:] = erf(self.a * r_jk) / r_jk

        # Fix 0/0 corner value:
        v_ijk[0, 0, 0] = 2 * self.a / pi**0.5

        self.K_Q = np.fft.fftn(v_ijk) * self.gd.dv

    def get_description(self):
        descriptors = []
        descriptors.append('Inner radius for %dx%dx%d Wigner-Seitz cell: '
                           '%.3f Ang' % (tuple(self.nk_c) + (self.rc * Bohr,)))
        descriptors.append('Range-separation parameter: %.3f Ang^-1' % (
            self.a / Bohr))
        descriptors.append('FFT size for calculating truncated Coulomb: '
                           '%dx%dx%d' % tuple(self.gd.N_c))

        return '\n'.join(descriptors)

    def get_potential_new(self, pw: PWDesc, grid: UGDesc) -> PWArray:
        K_G = pw.zeros()
        assert isinstance(K_G, PWArray)  # !!!
        size = tuple(grid.size)
        if pw.dtype == float:
            size = (size[0], size[1], size[2] // 2 + 1)
        self._get_potential(pw.indices(size)[pw.ng1:pw.ng2],
                            pw.kpt_c,
                            grid.size_c,
                            pw.dtype,
                            pw.reciprocal_vectors(),
                            K_G.data)
        return K_G

    def get_potential(self, pd) -> Array1D:
        K_G = pd.zeros()
        self._get_potential(pd.Q_qG[0],
                            pd.kd.bzk_kc[0],
                            pd.gd.N_c,
                            pd.dtype,
                            pd.get_reciprocal_vectors(add_q=True),
                            K_G)
        return K_G

    def _get_potential(self, Q_G, q_c, N_c, dtype, qG_Gv, K_G) -> None:
        shift_c = (q_c * self.nk_c).round().astype(int)
        max_c = self.gd.N_c // 2
        if dtype == complex:
            for G, Q in enumerate(Q_G):
                Q_c = (np.unravel_index(Q, N_c) + N_c // 2) % N_c - N_c // 2
                Q_c = Q_c * self.nk_c + shift_c
                if (abs(Q_c) < max_c).all():
                    K_G[G] = self.K_Q[tuple(Q_c)]
        else:
            Nr_c = N_c.copy()
            Nr_c[2] = N_c[2] // 2 + 1
            for G, Q in enumerate(Q_G):
                Q_c = np.array(np.unravel_index(Q, Nr_c))
                Q_c[:2] += N_c[:2] // 2
                Q_c[:2] %= N_c[:2]
                Q_c[:2] -= N_c[:2] // 2
                if (abs(Q_c) < max_c).all():
                    K_G[G] = self.K_Q[tuple(Q_c)]

        G2_G = np.sum(qG_Gv**2, axis=1)
        # G2_G = pd.G2_qG[0]
        a = self.a
        G0 = G2_G.argmin()
        if G2_G[G0] < 1e-11:
            K0 = K_G[G0] + pi / a**2
        with np.errstate(invalid='ignore'):
            K_G += 4 * pi * (1 - np.exp(-G2_G / (4 * a**2))) / G2_G
        if G2_G[G0] < 1e-11:
            K_G[G0] = K0
