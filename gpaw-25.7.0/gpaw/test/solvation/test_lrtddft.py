from ase.build import molecule

import pytest
import gpaw.solvation as solv
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.lrtddft import LrTDDFT
from gpaw import PoissonSolver


@pytest.mark.old_gpaw_only
def test_solvation_lrtddft():
    h = 0.3
    vac = 3.0

    atoms = molecule('H2')
    adjust_cell(atoms, vac, h)

    calc = solv.SolvationGPAW(
        mode='fd', xc='PBE', h=0.2,  # non-solvent DFT parameters
        nbands=3,
        convergence={'energy': 0.1, 'eigenstates': 0.01, 'density': 0.1},
        # convenient way to use HW14 water parameters:
        **solv.get_HW14_water_kwargs())

    # do the ground state calculation
    atoms.calc = calc
    atoms.get_potential_energy()
    print(id(calc.hamiltonian.poisson.dielectric))
    print(id(calc.hamiltonian.dielectric))

    # linear response using ground state Poisson
    lrw = LrTDDFT(calc)
    lrw.diagonalize()

    # We test the agreement of a pure RPA kernel
    # with setting eps to 1

    lr = LrTDDFT(calc,
                 poisson=PoissonSolver('fd', nn=calc.hamiltonian.poisson.nn))
    lr.diagonalize()

    calc.hamiltonian.poisson.dielectric.epsinf = 1.
    lr1 = LrTDDFT(calc)
    lr1.diagonalize()
    for ex, ex1 in zip(lr, lr1):
        assert ex.energy == pytest.approx(ex1.energy, abs=1e-14)
