import numpy as np
import pytest
from ase import Atoms

from gpaw import GPAW, restart
from gpaw.mom import prepare_mom_calculation


@pytest.mark.mom
def test_mom_lcao_forces(in_tmp_dir):
    force_ref = 11.52

    f_sn = [[1., 1., 1., 1., 0., 1., 0.],
            [1., 1., 1., 1., 1., 0., 0.]]
    L = 4.0
    d = 1.13
    delta = 0.01

    atoms = Atoms('CO',
                  [[0.5 * L, 0.5 * L, 0.5 * L - 0.5 * d],
                   [0.5 * L, 0.5 * L, 0.5 * L + 0.5 * d]])
    atoms.set_cell([L, L, L])
    atoms.rotate(1, 'x', center=[0.5 * L, 0.5 * L, 0.5 * L])

    calc = GPAW(mode='lcao',
                basis='dzp',
                nbands=7,
                h=0.24,
                xc='PBE',
                spinpol=True,
                symmetry='off',
                convergence={'energy': 100,
                             'density': 1e-4})

    atoms.calc = calc
    # Ground-state calculation
    atoms.get_potential_energy()
    calc.write('co_lcao_gs.gpw', 'all')

    for mom in [False, True]:
        atoms, calc = restart('co_lcao_gs.gpw', txt='-')

        occ = prepare_mom_calculation(calc, atoms, f_sn, use_projections=mom)
        F = atoms.get_forces()

        # Test overlaps
        occ.initialize_reference_orbitals()
        for kpt in calc.wfs.kpt_u:
            f_n = calc.get_occupation_numbers(spin=kpt.s)
            P = occ.calculate_weights(kpt, 1.0)
            assert (np.allclose(P, f_n))

        E = []
        p = atoms.positions.copy()
        for i in [-1, 1]:
            pnew = p.copy()
            pnew[0, 2] -= delta / 2. * i
            pnew[1, 2] += delta / 2. * i
            atoms.set_positions(pnew)

            E.append(atoms.get_potential_energy())

        f = np.sqrt(((F[1, :] - F[0, :])**2).sum()) * 0.5
        fnum = (E[0] - E[1]) / (2. * delta)  # central difference

        print(fnum, f)
        assert fnum == pytest.approx(force_ref, abs=0.016)
        assert f == pytest.approx(fnum, abs=0.1)
