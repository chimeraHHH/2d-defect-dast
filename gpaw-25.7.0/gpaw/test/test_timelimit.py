import pytest
from ase.build import molecule

from gpaw import GPAW, KohnShamConvergenceError
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.tddft import TDDFT
from gpaw.utilities.timelimit import TimeLimiter


def test_timelimit(in_tmp_dir, gpaw_new):
    if gpaw_new:
        pytest.skip('rewrite later using calc.callbacks')
    # Atoms
    atoms = molecule('Na2')
    atoms.center(vacuum=4.0)

    # Ground-state calculation that will never converge
    maxiter = 10
    calc = GPAW(mode='lcao', basis='sz(dzp)', setups='1', nbands=1,
                convergence={'density': 1e-100},
                symmetry={'point_group': False},
                maxiter=maxiter)
    atoms.calc = calc

    tl = TimeLimiter(calc, timelimit=0, output='scf.txt')
    tl.reset('scf', min_updates=3)
    try:
        atoms.get_potential_energy()
    except KohnShamConvergenceError:
        assert calc.scf.maxiter < maxiter, 'TimeLimiter did not break SCF loop'
    else:
        raise AssertionError('SCF loop ended too early')
    calc.write('gs.gpw', mode='all')

    # LCAOTDDFT calculation that will never finish
    td_calc = LCAOTDDFT('gs.gpw')
    tl = TimeLimiter(td_calc, timelimit=0, output='lcaotddft.txt')
    tl.reset('tddft', min_updates=3)
    td_calc.propagate(10, maxiter - td_calc.niter)
    assert td_calc.maxiter < maxiter, 'TimeLimiter did not break TDDFT loop'

    # Test mode='fd'

    # Prepare ground state
    calc = GPAW(mode='fd', setups='1', maxiter=1, nbands=1,
                symmetry={'point_group': False})
    atoms.calc = calc
    try:
        atoms.get_potential_energy()
    except KohnShamConvergenceError:
        pass
    calc.write('gs.gpw', mode='all')

    # TDDFT calculation that will never finish
    td_calc = TDDFT('gs.gpw')
    tl = TimeLimiter(td_calc, timelimit=0, output='tddft.txt')
    tl.reset('tddft', min_updates=3)
    td_calc.propagate(10, maxiter - td_calc.niter)
    assert td_calc.maxiter < maxiter, 'TimeLimiter did not break TDDFT loop'
