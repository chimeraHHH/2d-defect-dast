"""

tests solvation parameters as in

V. M. Sanchez, M. Sued, and D. A. Scherlis,
The Journal of Chemical Physics, vol. 131, no. 17, p. 174108, 2009
"""

from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell
import pytest
from ase import Atoms
from ase.units import mol, kcal, Pascal, m, Bohr
from gpaw.solvation import (
    SolvationGPAW,
    FG02SmoothStepCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction,
    SSS09Density)


def test_solvation_sss09():
    SKIP_VAC_CALC = True

    h = 0.24
    vac = 4.0

    epsinf = 78.36
    rho0 = 1.0 / Bohr ** 3
    beta = 2.4
    st = 72. * 1e-3 * Pascal * m

    atomic_radii = {'Cl': 2.059}

    convergence = {
        'energy': 0.05 / 8.,
        'density': 10.,
        'eigenstates': 10.}

    atoms = Atoms('Cl')
    adjust_cell(atoms, vac, h)

    if not SKIP_VAC_CALC:
        atoms.calc = GPAW(mode='fd',
                          xc='PBE',
                          h=h,
                          charge=-1,
                          convergence=convergence)
        Evac = atoms.get_potential_energy()
        print(Evac)
    else:
        # h=0.24, vac=4.0, setups: 0.9.11271, convergence: only energy 0.05 / 8
        Evac = -3.83245253419

    atoms.calc = SolvationGPAW(
        mode='fd', xc='PBE', h=h, charge=-1, convergence=convergence,
        cavity=FG02SmoothStepCavity(
            rho0=rho0, beta=beta,
            density=SSS09Density(atomic_radii=atomic_radii),
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=epsinf),
        interactions=[SurfaceInteraction(surface_tension=st)])
    Ewater = atoms.get_potential_energy()
    assert atoms.calc.get_number_of_iterations() < 40
    atoms.get_forces()
    DGSol = (Ewater - Evac) / (kcal / mol)
    print('Delta Gsol: %s kcal / mol' % DGSol)

    assert DGSol == pytest.approx(-75., abs=10.)
