"""Test selfconsistent RSF calculation with Yukawa potential including vc."""
import pytest
from gpaw.mpi import world
from ase import Atoms
from gpaw import GPAW, KohnShamConvergenceError
from gpaw.xc.hybrid import HybridXC
from gpaw.poisson import PoissonSolver
from gpaw.occupations import FermiDirac
from gpaw.test import gen
from gpaw.eigensolvers import RMMDIIS
from gpaw.utilities.adjust_cell import adjust_cell

pytestmark = pytest.mark.skipif(world.size < 4,
                                reason='world.size < 4')


@pytest.mark.old_gpaw_only
def test_exx_exx_scf(in_tmp_dir, add_cwd_to_setup_paths):
    h = 0.3

    # No energies - simpley convergence test, esp. for 3d TM

    # for atom in ['F', 'Cl', 'Br', 'Cu', 'Ag']:
    for atom in ['Ti']:
        gen(atom, xcname='PBE', scalarrel=False, exx=True)
        work_atom = Atoms(atom, [(0, 0, 0)])
        adjust_cell(work_atom, 4, h=h)
        work_atom.translate([0.01, 0.02, 0.03])
        work_atom.set_initial_magnetic_moments([2.0])
        calculator = GPAW(mode='fd',
                          convergence={'energy': 0.01,
                                       'eigenstates': 3,
                                       'density': 3},
                          eigensolver=RMMDIIS(),
                          poissonsolver=PoissonSolver(use_charge_center=True),
                          occupations=FermiDirac(width=0.0, fixmagmom=True),
                          h=h, maxiter=35,  # Up to 24 are needed by now
                          xc=HybridXC('PBE0'),
                          txt=atom + '-PBE0.txt')
        work_atom.calc = calculator
        try:
            work_atom.get_potential_energy()
        except KohnShamConvergenceError:
            pass
        assert calculator.scf.converged, 'Calculation not converged'
