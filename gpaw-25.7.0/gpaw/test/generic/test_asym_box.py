"""Check for change in total energy and lowest eigenvalue regarding to box."""
from ase import Atoms
from ase.parallel import parprint
from gpaw import GPAW, PoissonSolver
from gpaw.utilities.adjust_cell import adjust_cell
import pytest


def test_generic_asym_box(in_tmp_dir):
    h = 0.2
    s = Atoms('He')
    adjust_cell(s, 3, h=h)

    c = GPAW(mode='fd', charge=1, txt='He_plus.txt',
             poissonsolver=PoissonSolver(use_charge_center=True),
             convergence={     # run fast
                 'energy': 0.001,
                 'eigenstates': 1e-4,
                 'density': 1e-3})
    s.calc = c
    e_small = s.get_potential_energy()
    eps_small = c.get_eigenvalues()[0]

    cell = s.get_cell()
    cell[0] *= 2
    s.set_cell(cell)
    e_big = s.get_potential_energy()
    eps_big = c.get_eigenvalues()[0]

    parprint('Energies and Eigenvalues:')
    parprint('     Small Box    Wide Box')
    parprint('E:   {0:9.3f}     {1:9.3f}'.format(e_small, e_big))
    parprint('eps: {0:9.3f}     {1:9.3f}'.format(eps_small, eps_big))
    assert e_small == pytest.approx(e_big, abs=2.5e-4)
    assert eps_small == pytest.approx(eps_big, abs=6e-4)
