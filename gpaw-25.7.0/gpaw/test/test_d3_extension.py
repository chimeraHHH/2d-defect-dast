import pytest
from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW
from gpaw import restart
from gpaw.new.extensions import D3
import numpy as np
from ase import Atoms


@pytest.mark.parametrize('parallel', [(1, 1), (1, 2), (2, 2), (2, 1)])
@pytest.mark.parametrize('mode', [{'name': 'pw', 'ecut': 300}, 'lcao'])
def test_d3_extensions(mode, parallel, in_tmp_dir, gpaw_new, dftd3):
    if not gpaw_new:
        pytest.skip('Only GPAW new.')

    from ase.calculators.dftd3 import PureDFTD3
    domain, band = parallel
    if world.size < domain * band:
        pytest.skip('Not enough cores for this test.')
    if world.size > domain * band * 2:
        pytest.skip('Too many cores for this test.')

    # 1. Create a calculation with a particular list of extensions.
    def get_atoms():
        from ase.build import molecule
        atoms = molecule('H2')
        atoms[0].position[1] += 1
        atoms.center(vacuum=2)
        atoms.set_pbc((True, True, True))
        return atoms

    def D3ref(atoms):
        atoms = atoms.copy()
        atoms.calc = PureDFTD3(xc='PBE')
        return atoms.get_potential_energy(), atoms.get_forces()

    atoms = get_atoms()

    def get_calc(atoms):
        # To test multiple extensions, create two sprigs which add
        # up to k=ktot, which is what is tested in this test
        calc = GPAW(extensions=[D3(xc='PBE')],
                    symmetry='off',
                    parallel={'band': band, 'domain': domain},
                    kpts=(2, 1, 1),
                    mode=mode)
        atoms.calc = calc
        return calc

    calc = get_calc(atoms)

    E, F = atoms.get_potential_energy(), atoms.get_forces()
    D3_E, D3_F = D3ref(atoms)

    # Write the GPW file for the restart test later on (4.)
    print('Wrote the potential energy', E)
    calc.write('calc.gpw')

    # 2. Test that moving the atoms works after an SFC convergence
    atoms.positions[0, 2] -= 0.1
    movedE, movedF = atoms.get_potential_energy(), atoms.get_forces()

    movedD3_E, movedD3_F = D3ref(atoms)
    # Reset atoms to their original positions
    atoms.positions[0, 2] += 0.1

    # 3. Calculate a reference result without extensions
    calc = GPAW(mode=mode,
                kpts=(2, 1, 1),
                symmetry='off')
    atoms.calc = calc

    E0, F0 = atoms.get_potential_energy(), atoms.get_forces()

    # Manually evaluate the spring energy, and compare forces
    assert E == pytest.approx(E0 + D3_E)
    assert F == pytest.approx(F0 + D3_F)

    # Evaluate the reference energy and forces also for the moved atoms
    atoms.positions[0, 2] -= 0.1
    movedE0, movedF0 = atoms.get_potential_energy(), atoms.get_forces()
    assert movedE == pytest.approx(movedE0 + movedD3_E)
    assert movedF == pytest.approx(movedF0 + movedD3_F)

    # 4. Test restarting from a file
    atoms, calc = restart('calc.gpw', Class=GPAW)
    # Make sure the cached energies and forces are correct
    # without a new calculation
    assert E == pytest.approx(atoms.get_potential_energy())
    assert F == pytest.approx(atoms.get_forces())

    if mode == 'lcao':
        # See issue #1369
        return

    # Make sure the recalculated energies are forces are correct
    atoms.set_positions(atoms.get_positions() + 1e-10)
    assert E == pytest.approx(atoms.get_potential_energy(), abs=1e-5)
    assert F == pytest.approx(atoms.get_forces(), abs=1e-5)

    # 5. Test full blown relaxation.
    from ase.optimize import BFGS
    atoms = get_atoms()
    calc = get_calc(atoms)
    relax = BFGS(atoms)
    relax.run()
    nsteps = relax.nsteps
    assert atoms.get_distance(0, 1) == pytest.approx(0.76915, abs=1e-2)
    Egs = atoms.get_potential_energy()
    L = atoms.get_distance(0, 1)

    # 6. Test restarting from a relaxation.
    atoms = get_atoms()
    calc = get_calc(atoms)
    relax = BFGS(atoms, restart='relax_restart')
    for _, _ in zip(relax.irun(), range(3)):
        pass
    calc.write('restart_relax.gpw')
    atoms, calc = restart('restart_relax.gpw', Class=GPAW)
    relax = BFGS(atoms, restart='relax_restart')
    relax.run()

    assert relax.nsteps + 3 == nsteps
    assert atoms.get_distance(0, 1) == pytest.approx(L, abs=1e-2)
    assert atoms.get_potential_energy() == pytest.approx(Egs, abs=1e-4)


@pytest.mark.parametrize('parallel', [(1, 1), (1, 2), (2, 2), (2, 1)])
def test_d3_stress(parallel, in_tmp_dir, dftd3):
    from ase.calculators.dftd3 import DFTD3
    from ase.optimize import CellAwareBFGS
    from ase.build import bulk
    from ase.filters import FrechetCellFilter
    from gpaw.new.ase_interface import GPAW

    domain, band = parallel
    if world.size < domain * band:
        pytest.skip('Not enough cores for this test.')
    if world.size > domain * band * 2:
        pytest.skip('Too many cores for this test.')

    def get_atoms():
        atoms = bulk('C', a=3.5)
        atoms.set_cell(atoms.get_cell(),
                       scale_atoms=True)
        return atoms

    kwargs = dict(xc='LDA',
                  parallel={'band': band, 'domain': domain},
                  kpts=(2, 2, 2), txt='relax',
                  convergence={'density': 1e-5},
                  mode=dict(name='pw', ecut=300))

    def get_calc(x):
        return GPAW(**kwargs, **x)

    # 1. Old fashioned D3 calculation
    atoms = get_atoms()
    atoms.calc = DFTD3(xc='PBE', dft=get_calc({}))
    relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1),
                          restart='restart_oldfashioned')
    relax.run(smax=0.001)
    atoms_old_ref = atoms.copy()
    E_ref = atoms.get_potential_energy()

    # 2. New style D3 calculation
    atoms = get_atoms()
    atoms.calc = get_calc(dict(extensions=[D3(xc='PBE')]))
    relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1),
                          restart='restart_new')
    relax.run(smax=0.001)
    nsteps = relax.nsteps

    assert np.allclose(atoms.cell, atoms_old_ref.cell)
    assert np.allclose(atoms.get_scaled_positions(),
                       atoms_old_ref.get_scaled_positions())
    assert E_ref == pytest.approx(atoms.get_potential_energy(), abs=1e-4)

    # 3. Restarting geometry relaxation of new style D3 calculation
    atoms = get_atoms()
    atoms.calc = get_calc(dict(extensions=[D3(xc='PBE')]))
    relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1),
                          restart='restart_cont')
    relax.smax = 1e-4
    for _, _ in zip(relax.irun(), range(3)):
        pass
    atoms.calc.write('restart_cell_relax.gpw')
    atoms, calc = restart('restart_cell_relax.gpw', Class=GPAW)
    relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1),
                          restart='restart_cont')
    relax.run(smax=0.001)

    assert relax.nsteps + 3 == nsteps

    assert np.allclose(atoms.cell, atoms_old_ref.cell, rtol=1e-5, atol=1e-5)
    assert np.allclose(atoms.get_scaled_positions(),
                       atoms_old_ref.get_scaled_positions())
    assert E_ref == pytest.approx(atoms.get_potential_energy(), abs=1e-4)


def test_d3_isolated_atom(dftd3):
    atoms = Atoms('He')
    atoms.center(vacuum=3)
    calc = GPAW(xc='PBE',
                extensions=[D3(xc='PBE')],
                mode='pw')
    atoms.calc = calc
    atoms.get_potential_energy()
    assert np.allclose(atoms.get_forces(), 0, atol=1e-5)
    print(calc.dft.d3)
