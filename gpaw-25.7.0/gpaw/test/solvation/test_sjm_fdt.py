import pytest
from ase.build import fcc111, molecule
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.units import Pascal, fs, m

from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.solvation.sjm import SJM, SJMPower12Potential


@pytest.mark.skip('Too slow: 11 min.')
@pytest.mark.slow
@pytest.mark.old_gpaw_only
def test_sjm_fdt_true(in_tmp_dir):
    """Test if fdt dictionary is correctly set in the calculator.
    Test if fdt initial excess electron value is correctly
    set in the calculator.
    """
    atoms = fcc111('Au', size=(1, 1, 3))
    atoms.center(axis=2, vacuum=10.2)
    atoms.translate([0.0, 0.0, -4.0])

    water = molecule('H2O')
    water.rotate('y', 90.0)
    water.positions += atoms[2].position + (0.0, 0.0, 4.4) - water[0].position

    atoms.extend(water)
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    atoms.pbc = [True, True, False]

    # Solvated jellium parameters.
    sj_input = {
        'target_potential': 4.5,
        'fdt': {
            'dt': 0.5,
            'po_time': 1000.0,
            'th_temp': 300}}

    # Implicit solvent parameters (to SolvationGPAW).
    epsinf = 78.36
    gamma = 18.4 * 1e-3 * Pascal * m
    cavity = EffectivePotentialCavity(
        effective_potential=SJMPower12Potential(H2O_layer=True),
        temperature=298.15,  # K
        surface_calculator=GradientSurface())
    dielectric = LinearDielectric(epsinf=epsinf)
    interactions = [SurfaceInteraction(surface_tension=gamma)]

    # The calculator
    calc = SJM(
        mode='lcao',
        basis='szp(dzp)',
        txt='test_fdt_true.txt',
        gpts=(16, 16, 136),
        kpts=(1, 1, 1),
        xc='PBE',
        maxiter=1000,
        sj=sj_input,
        cavity=cavity,
        dielectric=dielectric,
        interactions=interactions,
        symmetry={'point_group': False})
    atoms.calc = calc

    atoms.get_potential_energy()
    atoms.get_forces()

    # Run 10 steps of Langevin dynamics.
    traj = 'md_fdt_true.traj'
    dyn = Langevin(atoms, 0.5 * fs, temperature_K=300,
                   friction=0.005, trajectory=traj)
    dyn.run(5)

    assert sj_input['fdt'] == atoms.calc.parameters['sj']['fdt']
    assert abs(atoms.calc.parameters['sj']['previous_electrons'][0]) < 1e-6
