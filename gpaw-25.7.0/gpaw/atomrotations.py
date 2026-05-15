import numpy as np

from gpaw.rotation import rotation
from gpaw.utilities import pack_density, unpack_density


class SingleAtomRotations:
    def __init__(self, R_sii):
        self.R_sii = R_sii

    @classmethod
    def new(cls, ni, l_j, R_slmm):
        nsym = len(R_slmm)
        R_sii = np.zeros((nsym, ni, ni))
        i1 = 0
        for l in l_j:
            i2 = i1 + 2 * l + 1
            for s, R_lmm in enumerate(R_slmm):
                R_sii[s, i1:i2, i1:i2] = R_lmm[l]
            i1 = i2
        return cls(R_sii)

    def symmetrize(self, a, D_aii, map_sa):
        ni = self.R_sii.shape[1]
        D_ii = np.zeros((ni, ni))
        for s, R_ii in enumerate(self.R_sii):
            D_ii += R_ii @ D_aii[map_sa[s][a]] @ R_ii.T
        return D_ii / len(map_sa)


class AtomRotations:
    def __init__(self, setups, id_a, symmetry):
        R_slmm = []
        for op_cc in symmetry.op_scc:
            op_vv = np.linalg.inv(symmetry.cell_cv) @ op_cc @ symmetry.cell_cv
            R_slmm.append([rotation(l, op_vv) for l in range(4)])

        rotations = {}
        for key, setup in setups.items():
            rotations[key] = SingleAtomRotations.new(setup.ni, setup.l_j,
                                                     R_slmm)

        self._rotations = rotations
        self._id_a = id_a

    def get_R_asii(self):
        return [self.get_by_a(a).R_sii for a in range(len(self._id_a))]

    def get_by_a(self, a):
        return self._rotations[self._id_a[a]]

    def symmetrize_atomic_density_matrices(self, D_asp, a_sa):
        if not D_asp:
            return

        nspins = next(iter(D_asp.values())).shape[0]

        for s in range(nspins):
            D_aii = [unpack_density(D_asp[a][s]) for a in range(len(D_asp))]
            for a, D_ii in enumerate(D_aii):
                D_asp[a][s] = pack_density(
                    self.get_by_a(a).symmetrize(a, D_aii, a_sa))
