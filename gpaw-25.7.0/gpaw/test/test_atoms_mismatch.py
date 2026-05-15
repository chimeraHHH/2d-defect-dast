# Check that atoms object mismatches are detected properly across CPUs.

import pytest
from ase.build import molecule
from gpaw.mpi import world, synchronize_atoms
from gpaw import GPAW
from ase.optimize import BFGS


@pytest.mark.ci
def test_atoms_mismatch(in_tmp_dir):
    system = molecule('H2O')
    synchronize_atoms(system, world)

    if world.rank == 1:
        system.positions[1, 1] += 1e-8  # fail (above tolerance)
    if world.rank == 2:
        system.cell[0, 0] += 1e-7  # fail (above tolerance)
    if world.rank == 3:
        system.positions[1, 1] += 1e-10  # pass (below tolerance)
    if world.rank == 4:
        system.cell[0, 1] += 1e-10  # pass (below tolerance)

    expected_err_ranks = {1: [], 2: [1]}.get(world.size, [1, 2])

    try:
        synchronize_atoms(system, world, tolerance=1e-9)
    except ValueError as e:
        assert (expected_err_ranks == e.args[1]).all()
    else:
        assert world.size == 1

    # Check that GPAW can handle small numerical differences within ASE
    if world.size == 1:
        # Test makes sense only in parallel
        return

    def mess_up_atoms(atoms):
        if world.rank == 1:
            atoms.positions[1, 1] += 1e-13

    system = molecule('H2O')
    system.center(3.0)
    calc = GPAW(mode='fd',
                h=0.2,
                convergence={'energy': 0.01, 'density': 1.0e-2,
                             'eigenstates': 4.0e-3})
    system.calc = calc
    dyn = BFGS(system)
    dyn.attach(mess_up_atoms, 1, system)
    dyn.run(steps=2)
