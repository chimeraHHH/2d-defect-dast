# Test that the atomic corrections of LCAO work correctly,
# by verifying that the different implementations yield the same numbers.
#
# For example the corrections P^* dH P to the Hamiltonian.
#
# This is done by invoking GPAW once for each type of calculation.

import pytest
from ase.build import molecule, bulk

from gpaw import GPAW, LCAO
from gpaw.mpi import world


def system1():
    system = molecule('CH3CH2OH')
    system.center(vacuum=3.0)
    system.pbc = (0, 1, 1)
    system = system.repeat((1, 1, 2))
    system.rattle(stdev=0.05)
    return system


def system2():
    return bulk('Cu', orthorhombic=True) * (2, 1, 2)


@pytest.mark.parametrize('atoms, kpts, eref', [
    (system1(), [1, 1, 1], -58.845),
    (system2(), [2, 3, 4], -22.691)])
def test_lcao_atomic_corrections(atoms, in_tmp_dir, scalapack, kpts, eref,
                                 gpaw_new):
    # Use a cell large enough that some overlaps are zero.
    # Thus the matrices will have at least some sparsity.

    if gpaw_new:
        if world.size >= 4:
            pytest.skip('Not implemented')
        corrections = ['ignored for now']
    else:
        corrections = ['dense', 'sparse']

    energies = []
    for i, correction in enumerate(corrections):
        parallel = {}
        if world.size >= 4:
            parallel['band'] = 2
            # if correction.name != 'dense':
            parallel['sl_auto'] = True
        calc = GPAW(mode=LCAO(atomic_correction=correction),
                    basis='sz(dzp)',
                    # spinpol=True,
                    parallel=parallel,
                    txt=f'gpaw.{i}.txt',
                    h=0.35, kpts=kpts,
                    convergence={'maximum iterations': 2})
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        energies.append(energy)
        if calc.world.rank == 0:
            print('e', energy)

    master = calc.wfs.world.rank == 0
    if master:
        print('energies', energies)

    e0 = energies[0]
    errs = []
    for energy, c in zip(energies, corrections):
        assert energy == pytest.approx(eref, abs=0.001)
        err = abs(energy - e0)
        errs.append(err)
        if master:
            print('err=%e :: name=%s' % (err, correction))

    maxerr = max(errs)
    assert maxerr < 1e-11, maxerr
