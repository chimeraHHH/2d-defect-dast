import pytest
import numpy as np
from ase.build import bulk

from gpaw import GPAW, PW
from gpaw.mpi import world

# This test is asserting that the expert diagonalization
# routine gives the same result as the non-expert version
# in terms of eigenvalues and wavefunctions


def test_pw_expert_diag(in_tmp_dir, scalapack):
    wfs_e = []
    for i, nbands in enumerate([None, 48]):
        si = bulk('Si')
        name = f'si_{i}'
        si.center()
        calc = GPAW(mode=PW(120), kpts=(1, 1, 2),
                    eigensolver='rmm-diis',
                    parallel={'domain': 1},
                    symmetry='off', txt=name + '.txt')
        si.calc = calc
        si.get_potential_energy()
        calc.diagonalize_full_hamiltonian(nbands=nbands)
        string = name + '.gpw'
        calc.write(string, 'all')
        wfs_e.append(calc.wfs)

    # Test against values from reference revision
    epsn_n = np.array([-0.21072488, 0.05651796,
                       0.17629109, 0.17629122,
                       0.26459912])
    wfsold_G = np.array([7.52087176, 54.64580691,
                         -3.813492786, -0.14781141,
                         -0.64662074])

    kpt_u = wfs_e[0].kpt_u
    for kpt in kpt_u:
        if kpt.k == 0:
            psit = kpt.psit_nG[1, 0:5].copy()
            if wfsold_G[0] * psit[0] < 0:
                psit *= -1.
            if world.rank == 0:
                assert np.allclose(epsn_n, kpt.eps_n[0:5], atol=1e-4), \
                    'Eigenvalues have changed'
                assert np.allclose(wfsold_G, psit, atol=5e-3), \
                    'Wavefunctions have changed'

    wfstmp, wfs = wfs_e
    for kpt, kpttmp in zip(wfs.kpt_u, wfstmp.kpt_u):
        if wfs.bd.comm.rank == 0:
            for m, (psi_G, eps) in enumerate(zip(kpt.psit_nG, kpt.eps_n)):
                # Have to do like this if bands are degenerate
                booleanarray = np.abs(kpttmp.eps_n - eps) < 1e-10
                inds = np.argwhere(booleanarray)
                count = len(inds)
                assert count > 0, 'Difference between eigenvalues!'

                psitmp_nG = kpttmp.psit_nG[inds][:, 0, :]
                fidelity = 0
                for psitmp_G in psitmp_nG:
                    fidelity += (
                        np.abs(np.dot(psitmp_G.conj(), psi_G))**2 /
                        np.dot(psitmp_G.conj(), psitmp_G) /
                        np.dot(psi_G.conj(), psi_G))

                assert fidelity == pytest.approx(1, abs=1e-10)


if __name__ == '__main__':
    test_pw_expert_diag(1, 1)
