from gpaw.utilities.adjust_cell import adjust_cell
from ase.build import molecule
from ase.units import Pascal, m
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)


def test_solvation_swap_atoms():
    h = 0.3
    vac = 3.0
    u0 = 0.180
    epsinf = 80.
    st = 18.4 * 1e-3 * Pascal * m
    T = 298.15

    atomic_radii = {'H': 1.09}

    convergence = {
        'energy': 0.1 / 8.,
        'density': 10.,
        'eigenstates': 10.,
    }

    atoms = molecule('H2O')
    adjust_cell(atoms, vac, h)

    calc = SolvationGPAW(
        mode='fd', xc='LDA', h=h, convergence=convergence,
        cavity=EffectivePotentialCavity(
            effective_potential=Power12Potential(atomic_radii, u0),
            temperature=T,
            surface_calculator=GradientSurface()
        ),
        dielectric=LinearDielectric(epsinf=epsinf),
        interactions=[SurfaceInteraction(surface_tension=st)]
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()

    def env(calc):
        if calc.old:
            return calc.hamiltonian
        return calc.environment

    eps_gradeps = env(calc).dielectric.eps_gradeps

    # same molecules, different cell, reallocate
    atoms = molecule('H2O')
    atoms.positions[0][0] = atoms.positions[0][0] - 1.
    adjust_cell(atoms, vac, h)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()

    assert env(calc).dielectric.eps_gradeps is not eps_gradeps
    eps_gradeps = env(calc).dielectric.eps_gradeps

    # small position change, no reallocate
    atoms.positions[0][0] = atoms.positions[0][0] + 1e-2
    atoms.get_potential_energy()
    atoms.get_forces()
    assert env(calc).dielectric.eps_gradeps is eps_gradeps
    eps_gradeps = env(calc).dielectric.eps_gradeps
    radii = env(calc).cavity.effective_potential.r12_a

    # completely different atoms object, reallocate, read new radii
    atoms = molecule('NH3')
    adjust_cell(atoms, vac, h)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()
    assert env(calc).dielectric.eps_gradeps is not eps_gradeps
    assert env(calc).cavity.effective_potential.r12_a is not radii
