import warnings

import pytest
from ase.build import molecule

from gpaw.solvation import (EffectivePotentialCavity, LinearDielectric,
                            Power12Potential, SolvationGPAW)
from gpaw.solvation.poisson import ADM12PoissonSolver
from gpaw.utilities.adjust_cell import adjust_cell

h = 0.3
vac = 3.0
u0 = 0.180
epsinf = 80.0
T = 298.15
atomic_radii = {'H': 1.09}


convergence = {
    'energy': 0.05 / 8,
    'density': 10.0,
    'eigenstates': 10.0}


@pytest.mark.old_gpaw_only
def test_solvation_pbc():
    atoms = molecule('H2O')
    adjust_cell(atoms, vac, h)
    atoms.pbc = True

    with warnings.catch_warnings():
        # Ignore production code warning for ADM12PoissonSolver
        warnings.simplefilter('ignore')
        psolver = ADM12PoissonSolver(eps=1e-7)

    atoms.calc = SolvationGPAW(
        mode='fd',
        xc='LDA',
        h=h,
        convergence=convergence,
        cavity=EffectivePotentialCavity(
            effective_potential=Power12Potential(
                atomic_radii=atomic_radii, u0=u0),
            temperature=T),
        dielectric=LinearDielectric(epsinf=epsinf),
        poissonsolver=psolver)
    atoms.get_potential_energy()
    atoms.get_forces()
