import numpy as np

from gpaw.cgpaw import GG_shuffle

from gpaw.response.symmetry import QSymmetries
from gpaw.response.qpd import SingleQPWDescriptor


class HeadSymmetryOperators:
    def __init__(self, symmetries, gd):
        self.M_svv = initialize_v_maps(symmetries, gd.cell_cv, gd.icell_cv)
        self.sign_s = symmetries.sign_s
        self.nsym = len(symmetries)

    def symmetrize_wvv(self, A_wvv):
        """Symmetrize chi0_wvv"""
        tmp_wvv = np.zeros_like(A_wvv)
        for M_vv, sign in zip(self.M_svv, self.sign_s):
            tmp = np.dot(np.dot(M_vv.T, A_wvv), M_vv)
            if sign == 1:
                tmp_wvv += np.transpose(tmp, (1, 0, 2))
            elif sign == -1:  # transpose head
                tmp_wvv += np.transpose(tmp, (1, 2, 0))
        # Overwrite the input
        A_wvv[:] = tmp_wvv / self.nsym


class BodySymmetryOperators:
    def __init__(self, symmetries, qpd):
        self.G_sG = initialize_G_maps(symmetries, qpd)
        self.sign_s = symmetries.sign_s
        self.nsym = len(symmetries)

    def symmetrize_wGG(self, A_wGG):
        """Symmetrize an array in GG'."""
        for A_GG in A_wGG:
            tmp_GG = np.zeros_like(A_GG, order='C')
            for G_G, sign in zip(self.G_sG, self.sign_s):
                # Numpy:
                # if sign == 1:
                #     tmp_GG += A_GG[G_G, :][:, G_G]
                # if sign == -1:
                #     tmp_GG += A_GG[G_G, :][:, G_G].T
                # C:
                GG_shuffle(G_G, sign, A_GG, tmp_GG)
            A_GG[:] = tmp_GG / self.nsym

    # Set up complex frequency alias
    symmetrize_zGG = symmetrize_wGG


class WingSymmetryOperators(HeadSymmetryOperators):
    def __init__(self, symmetries, qpd):
        super().__init__(symmetries, qpd.gd)
        self.G_sG = initialize_G_maps(symmetries, qpd)

    def symmetrize_wxvG(self, A_wxvG):
        """Symmetrize chi0_wxvG"""
        tmp_wxvG = np.zeros_like(A_wxvG)
        for M_vv, sign, G_G in zip(self.M_svv, self.sign_s, self.G_sG):
            if sign == 1:
                tmp = sign * np.dot(M_vv.T, A_wxvG[..., G_G])
            elif sign == -1:  # transpose wings
                tmp = sign * np.dot(M_vv.T, A_wxvG[:, ::-1, :, G_G])
            tmp_wxvG += np.transpose(tmp, (1, 2, 0, 3))
        # Overwrite the input
        A_wxvG[:] = tmp_wxvG / self.nsym


def initialize_v_maps(symmetries: QSymmetries,
                      cell_cv: np.ndarray,
                      icell_cv: np.ndarray):
    """Calculate cartesian component mapping."""
    return np.array([cell_cv.T @ U_cc.T @ icell_cv
                     for U_cc in symmetries.U_scc])


def initialize_G_maps(symmetries: QSymmetries, qpd: SingleQPWDescriptor):
    """Calculate the Gvector mappings."""
    assert np.allclose(symmetries.q_c, qpd.q_c)
    B_cv = 2.0 * np.pi * qpd.gd.icell_cv
    G_Gv = qpd.get_reciprocal_vectors(add_q=False)
    G_Gc = np.dot(G_Gv, np.linalg.inv(B_cv))
    Q_G = qpd.Q_qG[0]

    G_sG = []
    for U_cc, sign, shift_c in symmetries:
        iU_cc = np.linalg.inv(U_cc).T
        UG_Gc = np.dot(G_Gc - shift_c, sign * iU_cc)

        assert np.allclose(UG_Gc.round(), UG_Gc)
        UQ_G = np.ravel_multi_index(UG_Gc.round().astype(int).T,
                                    qpd.gd.N_c, 'wrap')

        G_G = len(Q_G) * [None]
        for G, UQ in enumerate(UQ_G):
            try:
                G_G[G] = np.argwhere(Q_G == UQ)[0][0]
            except IndexError as err:
                raise RuntimeError(
                    'Something went wrong: a symmetry operation mapped a '
                    'G-vector outside the plane-wave cutoff sphere') from err
        G_sG.append(np.array(G_G, dtype=np.int32))
    return np.array(G_sG)
