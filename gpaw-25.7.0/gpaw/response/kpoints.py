from functools import cached_property

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from gpaw.bztools import get_reduced_bz, unique_rows


class KPointFinder:
    def __init__(self, bzk_kc):
        self.kdtree = cKDTree(self._round(bzk_kc))

    @staticmethod
    def _round(bzk_kc):
        return np.mod(np.mod(bzk_kc, 1).round(6), 1)

    def find(self, kpt_c):
        distance, k = self.kdtree.query(self._round(kpt_c))
        if distance > 1.e-6:
            raise ValueError('Requested k-point is not on the grid. '
                             'Please check that your q-points of interest '
                             'are commensurate with the k-point grid.')

        return k


class ResponseKPointGrid:
    def __init__(self, kd):
        self.kd = kd

    @cached_property
    def kptfinder(self):
        return KPointFinder(self.kd.bzk_kc)


class KPointDomain:
    def __init__(self, k_kc, icell_cv):
        self.k_kc = k_kc
        self.icell_cv = icell_cv

    def __len__(self):
        return len(self.k_kc)

    @cached_property
    def k_kv(self):
        return self.k_kc @ (2 * np.pi * self.icell_cv)


class KPointDomainGenerator:
    def __init__(self, symmetries, kpoints):
        self.symmetries = symmetries

        self.kd = kpoints.kd
        self.kptfinder = kpoints.kptfinder

    def how_many_symmetries(self):
        # temporary backwards compatibility for external calls
        return len(self.symmetries)

    def get_infostring(self):
        # Maybe we can avoid calling this somehow, we're only using
        # it to print:
        K_gK = self.group_kpoints()
        ng = len(K_gK)
        txt = f'{ng} groups of equivalent kpoints. '
        percent = (1. - (ng + 0.) / self.kd.nbzkpts) * 100
        txt += f'{percent}% reduction.\n'
        return txt

    def group_kpoints(self, K_k=None):
        """Group kpoints according to the reduced symmetries"""
        if K_k is None:
            K_k = np.arange(self.kd.nbzkpts)
        bz2bz_kS = self.kd.bz2bz_ks  # on kd, s is the global symmetry index
        nk = len(bz2bz_kS)
        sbz2sbz_ks = bz2bz_kS[K_k][:, self.symmetries.S_s]  # s: q-symmetries
        # Avoid -1 (see documentation in gpaw.symmetry)
        sbz2sbz_ks[sbz2sbz_ks == -1] = nk

        smallestk_k = np.sort(sbz2sbz_ks)[:, 0]
        k2g_g = np.unique(smallestk_k, return_index=True)[1]

        K_gs = sbz2sbz_ks[k2g_g]
        K_gK = [np.unique(K_s[K_s != nk]) for K_s in K_gs]

        return K_gK

    def get_kpt_domain(self):
        k_kc = np.array([self.kd.bzk_kc[K_K[0]] for
                         K_K in self.group_kpoints()])
        return k_kc

    def get_tetrahedron_ikpts(self, *, pbc_c, cell_cv):
        """Find irreducible k-points for tetrahedron integration."""
        U_scc = np.array([  # little group of q
            sign * U_cc for U_cc, sign, _ in self.symmetries])

        # Determine the irreducible BZ
        bzk_kc, ibzk_kc, _ = get_reduced_bz(cell_cv,
                                            U_scc,
                                            False,
                                            pbc_c=pbc_c)

        n = 3
        N_xc = np.indices((n, n, n)).reshape((3, n**3)).T - n // 2

        # Find the irreducible kpoints
        tess = Delaunay(ibzk_kc)
        ik_kc = []
        for N_c in N_xc:
            k_kc = self.kd.bzk_kc + N_c
            k_kc = k_kc[tess.find_simplex(k_kc) >= 0]
            if not len(ik_kc) and len(k_kc):
                ik_kc = unique_rows(k_kc)
            elif len(k_kc):
                ik_kc = unique_rows(np.append(k_kc, ik_kc, axis=0))

        return ik_kc

    def get_tetrahedron_kpt_domain(self, *, pbc_c, cell_cv):
        ik_kc = self.get_tetrahedron_ikpts(pbc_c=pbc_c, cell_cv=cell_cv)
        if pbc_c.all():
            k_kc = ik_kc
        else:
            k_kc = np.append(ik_kc,
                             ik_kc + (~pbc_c).astype(int),
                             axis=0)
        return k_kc

    def get_kpoint_weight(self, k_c):
        K = self.kptfinder.find(k_c)
        iK = self.kd.bz2ibz_k[K]
        K_k = self.unfold_ibz_kpoint(iK)
        K_gK = self.group_kpoints(K_k)

        for K_k in K_gK:
            if K in K_k:
                return len(K_k)

    def unfold_ibz_kpoint(self, ik):
        """Return kpoints related to irreducible kpoint."""
        kd = self.kd
        K_k = np.unique(kd.bz2bz_ks[kd.ibz2bz_k[ik]])
        K_k = K_k[K_k != -1]
        return K_k
