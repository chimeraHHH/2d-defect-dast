import pytest
from ase.build import bulk
from gpaw import GPAW
from gpaw.poisson import FFTPoissonSolver, FDPoissonSolver, FastPoissonSolver
import numpy as np


@pytest.mark.parametrize('poisson', [FFTPoissonSolver, FDPoissonSolver,
                                     FastPoissonSolver])
def test_poisson_cell_change(poisson):
    atoms = bulk('Na')
    calc = GPAW(mode='lcao', poissonsolver=poisson())
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.set_cell(atoms.cell * 1.1, scale_atoms=True)
    Ep = atoms.get_potential_energy()
    calc = GPAW(mode='lcao', poissonsolver=poisson())
    atoms.calc = calc
    Ep2 = atoms.get_potential_energy()
    assert np.allclose(Ep, Ep2, rtol=1e-4, atol=1e-5)
