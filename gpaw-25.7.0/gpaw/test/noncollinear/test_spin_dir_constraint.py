import numpy as np
import pytest

from ase import Atoms

from gpaw.mpi import world
from gpaw.new.constraints import SpinDirectionConstraint
from gpaw.new.ase_interface import GPAW


@pytest.mark.soc
@pytest.mark.skipif(world.size > 1, reason='Gamma-point calculation.')
def test_spin_dir_constraint_H():

    c = 2.5  # Å
    atom = Atoms('H', scaled_positions=[[0.5, 0.5, 0.5]],
                 cell=[c, c, c], pbc=False)

    # Constrain spin to point along x direction
    constraint = SpinDirectionConstraint({0: [1.0, 0.0, 0.0]}, 2.0)

    # Initialize spin along [1 1 1]
    calc = GPAW(
        mode={'name': 'pw', 'ecut': 400}, xc='LDA',
        nbands=1, symmetry='off',
        soc=True, magmoms=np.array([[1, 1, 1]]) / np.sqrt(3),
        parallel={'domain': 1, 'band': 1},
        extensions=[constraint])

    atom.calc = calc
    atom.get_potential_energy()
    calc.write('h2.gpw')
    GPAW('h2.gpw')

    # Assert that spin points along x
    smm_v = calc.dft.density.calculate_magnetic_moments()[0]
    assert smm_v[0] == pytest.approx(1., abs=1e-3)
    assert smm_v[1] == pytest.approx(0., abs=1e-3)
    assert smm_v[2] == pytest.approx(0., abs=1e-3)


@pytest.mark.soc
@pytest.mark.skipif(world.size > 1, reason='Unit test with no'
                                           ' parallelization.')
def test_spin_dir_constraint_derivative():

    rng = np.random.default_rng(seed=23)

    # Generate random data simulating a setup with s and p orbitals.
    M_vii = rng.random([3, 4, 4], dtype=np.float64)
    # Make Hermitian
    M_vii = (M_vii + np.transpose(M_vii, (0, 2, 1))) / 2
    l_j = [0, 1]

    # Initialize constraint
    constraint = SpinDirectionConstraint({0: [0, 0, 1]}, 2.0)

    # Generate some radial inner products
    N0_q = np.zeros(len(l_j) * (len(l_j) + 1) // 2)
    N0_q[0] = 0.8
    N0_q[2] = 0.6

    eL, V_vii = constraint.calculate(M_vii, 0, l_j, N0_q)
    assert eL == 0., 'Do not calculate constraining field energy unless stated'

    # Check that the spin constraint Hamiltonian is calculated correctly by
    # comparing it with an energy derivative w.r.t. a density matrix element
    # calculated through finite difference.

    diff = 1e-5

    M1_vii = M_vii.copy()
    M1_vii[0, 2, 2] += diff / 2
    eL1, _ = constraint.calculate(M1_vii, 0, l_j, N0_q, return_energy=True)

    M2_vii = M_vii.copy()
    M2_vii[0, 2, 2] -= diff / 2
    eL2, _ = constraint.calculate(M2_vii, 0, l_j, N0_q, return_energy=True)

    assert (eL1 - eL2) / diff == pytest.approx(V_vii[0, 2, 2], abs=1e-8)
