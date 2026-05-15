import numpy as np
from ase.units import Hartree

from gpaw.lcao.projected_wannier import condition_number, dots, get_bfs
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.tools import tri2full


def get_lcao_xc(calc, P_aqMi, bfs=None, spin=0):
    nq = len(calc.wfs.kd.ibzk_qc)
    nao = calc.wfs.setups.nao
    dtype = calc.wfs.dtype
    if bfs is None:
        bfs = get_bfs(calc)

    if calc.density.nt_sg is None:
        calc.density.interpolate_pseudo_density()
    nt_sg = calc.density.nt_sg
    vxct_sg = calc.density.finegd.zeros(calc.wfs.nspins)
    calc.hamiltonian.xc.calculate(calc.density.finegd, nt_sg, vxct_sg)
    vxct_G = calc.wfs.gd.zeros()
    calc.hamiltonian.restrict_and_collect(vxct_sg[spin], vxct_G)
    Vxc_qMM = np.zeros((nq, nao, nao), dtype)
    for q, Vxc_MM in enumerate(Vxc_qMM):
        bfs.calculate_potential_matrix(vxct_G, Vxc_MM, q)
        tri2full(Vxc_MM, 'L')

    # Add atomic PAW corrections
    for a, P_qMi in P_aqMi.items():
        D_sp = calc.density.D_asp[a][:]
        H_sp = np.zeros_like(D_sp)
        calc.hamiltonian.xc.calculate_paw_correction(calc.wfs.setups[a],
                                                     D_sp, H_sp)
        H_ii = unpack_hermitian(H_sp[spin])
        for Vxc_MM, P_Mi in zip(Vxc_qMM, P_qMi):
            Vxc_MM += dots(P_Mi, H_ii, P_Mi.T.conj())
    return Vxc_qMM * Hartree


class LCAOwrap:
    def __init__(self, calc, spin=0):
        assert calc.wfs.gd.comm.size == 1
        assert calc.wfs.kd.comm.size == 1
        assert calc.wfs.bd.comm.size == 1

        from gpaw.lcao.tools import get_lcao_hamiltonian
        H_skMM, S_kMM = get_lcao_hamiltonian(calc)

        self.calc = calc
        self.dtype = calc.wfs.dtype
        self.spin = spin
        self.H_qww = H_skMM[spin]
        self.S_qww = S_kMM
        self.P_aqwi = calc.wfs.P_aqMi
        self.Nw = self.S_qww.shape[-1]

        for S in self.S_qww:
            print('Condition number: %0.1e' % condition_number(S))

    def get_hamiltonian(self, q=0, indices=None):
        if indices is None:
            return self.H_qww[q]
        else:
            return self.H_qww[q].take(indices, 0).take(indices, 1)

    def get_overlap(self, q=0, indices=None):
        if indices is None:
            return self.S_qww[q]
        else:
            return self.S_qww[q].take(indices, 0).take(indices, 1)

    def get_projections(self, q=0, indices=None):
        if indices is None:
            return {a: P_qwi[q] for a, P_qwi in self.P_aqwi.items()}
        else:
            return {a: P_qwi[q].take(indices, 0)
                    for a, P_qwi in self.P_aqwi.items()}

    def get_orbitals(self, q=-1, indices=None):
        assert q == -1
        if indices is None:
            indices = range(self.Nw)
        Ni = len(indices)
        C_wM = np.zeros((Ni, self.Nw), self.dtype)
        for i, C_M in zip(indices, C_wM):
            C_M[i] = 1.0
        w_wG = self.calc.wfs.gd.zeros(Ni, dtype=self.dtype)
        self.calc.wfs.basis_functions.lcao_to_grid(C_wM, w_wG, q=-1)
        return w_wG

    def get_Fcore(self, q=0, indices=None):
        if indices is None:
            Fcore_ww = np.zeros_like(self.H_qww[q])
        else:
            Fcore_ww = np.zeros((len(indices), len(indices)))
        for a, P_wi in self.get_projections(q, indices).items():
            if self.calc.wfs.setups[a].type != 'ghost':
                X_ii = unpack_hermitian(self.calc.wfs.setups[a].X_p)
                Fcore_ww -= dots(P_wi, X_ii, P_wi.T.conj())
        return Fcore_ww * Hartree

    def get_xc(self, q=0, indices=None):
        if not hasattr(self, 'Vxc_qww'):
            self.Vxc_qww = get_lcao_xc(self.calc, self.P_aqwi,
                                       bfs=self.calc.wfs.basis_functions,
                                       spin=self.spin)
        if indices is None:
            return self.Vxc_qww[q]
        else:
            return self.Vxc_qww[q].take(indices, 0).take(indices, 1)
