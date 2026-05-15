import numpy as np

from gpaw.utilities.blas import mmm
from gpaw.utilities import unpack_hermitian


class BaseAtomicCorrection:
    name = 'base'
    description = 'base class for atomic corrections with LCAO'

    def __init__(self, dS_aii, Mstart, Mstop):
        self.dS_aii = dS_aii
        self.Mstart = Mstart
        self.Mstop = Mstop

    @staticmethod
    def get_dS(atom_partition, setups):
        dO_aii = atom_partition.arraydict([setup.dO_ii.shape
                                           for setup in setups])
        for a in dO_aii:
            dO_aii[a][:] = setups[a].dO_ii
        return dO_aii

    def calculate_hamiltonian(self, kpt, dH_asp, H_MM, yy):
        dH_aii = dH_asp.partition.arraydict(self.dS_aii.shapes_a,
                                            dtype=dH_asp.dtype)

        for a in dH_asp:
            dH_aii[a] = yy * unpack_hermitian(dH_asp[a][kpt.s])

        self.calculate(kpt.q, dH_aii, H_MM, self.Mstart, self.Mstop)

    def add_overlap_correction(self, S_qMM):
        for q, S_MM in enumerate(S_qMM):
            self.calculate(q, self.dS_aii, S_MM, self.Mstart, self.Mstop)

    def calculate(self, q, dX_aii, X_MM):
        raise NotImplementedError


class DenseAtomicCorrection(BaseAtomicCorrection):
    name = 'dense'
    description = 'dense with blas'

    def __init__(self, P_aqMi, dS_aii, Mstart, Mstop):
        BaseAtomicCorrection.__init__(self, dS_aii, Mstart, Mstop)
        self.P_aqMi = P_aqMi

    @classmethod
    def new_from_wfs(cls, wfs):
        return cls(wfs.P_aqMi, cls.get_dS(wfs.atom_partition, wfs.setups),
                   wfs.ksl.Mstart, wfs.ksl.Mstop)

    def calculate(self, q, dX_aii, X_MM, Mstart, Mstop):
        dtype = X_MM.dtype
        P_aqMi = self.P_aqMi

        # P_aqMi is distributed over domains (a) and bands (M).
        # Hence the correction X_MM = sum(P dX P) includes contributions
        # only from local atoms; the result must be summed over gd.comm
        # to get all 'a' contributions, and it will be locally calculated
        # only on the local slice of bands.
        for a, dX_ii in dX_aii.items():
            P_Mi = P_aqMi[a][q]
            assert dtype == P_Mi.dtype
            dXP_iM = np.zeros((dX_ii.shape[1], P_Mi.shape[0]), dtype)
            # (ATLAS can't handle uninitialized output array)
            mmm(1.0, np.asarray(dX_ii, dtype), 'N', P_Mi, 'C', 0.0, dXP_iM)
            mmm(1.0, P_Mi[Mstart:Mstop], 'N', dXP_iM, 'N', 1.0, X_MM)

    def calculate_projections(self, wfs, kpt):
        for a, P_ni in kpt.P_ani.items():
            # ATLAS can't handle uninitialized output array:
            P_ni.fill(117)
            mmm(1.0, kpt.C_nM, 'N', wfs.P_aqMi[a][kpt.q], 'N', 0.0, P_ni)


class SparseAtomicCorrection(BaseAtomicCorrection):
    name = 'sparse'
    description = 'sparse using scipy'

    def __init__(self, Psparse_qIM, P_indices, dS_aii, Mstart, Mstop,
                 tolerance=1e-12):
        BaseAtomicCorrection.__init__(self, dS_aii, Mstart, Mstop)
        self.Psparse_qIM = Psparse_qIM
        self.P_indices = P_indices
        # We currently don't use tolerance although we could speed things
        # up that way.
        #
        # Tolerance is for zeroing elements very close to zero, which
        # often increases sparsity somewhat, even for very small values.
        self.tolerance = tolerance

    @classmethod
    def new_from_wfs(cls, wfs):
        return cls(wfs.P_qIM, wfs.setups.projector_indices(),
                   cls.get_dS(wfs.atom_partition, wfs.setups),
                   wfs.ksl.Mstart, wfs.ksl.Mstop)

    def calculate(self, q, dX_aii, X_MM, Mstart, Mstop):
        P_indices = self.P_indices
        nI = P_indices.max

        import scipy.sparse as sparse
        dXsparse_II = sparse.dok_matrix((nI, nI), dtype=X_MM.dtype)
        for a in dX_aii:
            I1, I2 = P_indices[a]
            dXsparse_II[I1:I2, I1:I2] = dX_aii[a]
        dXsparse_II = dXsparse_II.tocsr()

        Psparse_IM = self.Psparse_qIM[q]
        Psparse_MI = Psparse_IM[:, Mstart:Mstop].transpose().conj()
        Xsparse_MM = Psparse_MI.dot(dXsparse_II.dot(Psparse_IM))
        X_MM[:, :] += Xsparse_MM.todense()

    def calculate_projections(self, wfs, kpt):
        P_indices = self.P_indices
        P_In = self.Psparse_qIM[kpt.q].dot(kpt.C_nM.T.conj())
        for a in kpt.P_ani:
            I1, I2 = P_indices[a]
            kpt.P_ani[a][:, :] = P_In[I1:I2, :].T.conj()
