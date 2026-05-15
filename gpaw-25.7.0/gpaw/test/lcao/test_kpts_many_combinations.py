import pytest
from itertools import count
from ase.build import bulk
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl


def ikwargs():
    for augment_grids in [False, True]:
        for sl_auto in ([0, 1] if compiled_with_sl() else [0]):
            if world.size == 16:
                # This won't happen in ordinary test suite
                parsizes = [[4, 2, 2], [2, 4, 2], [2, 2, 4]]
            elif world.size == 8:
                parsizes = [[2, 2, 2]]
            elif world.size == 4:
                parsizes = [[1, 2, 2]]
            elif world.size == 2:
                parsizes = [[1, 1, 2]]
            else:
                assert world.size == 1
                parsizes = [[1, 1, 1]]
            for kpt, band, domain in parsizes:
                parallel = dict(kpt=kpt,
                                band=band,
                                domain=domain,
                                sl_auto=sl_auto,
                                augment_grids=augment_grids)
                yield dict(parallel=parallel)


@pytest.mark.old_gpaw_only
def test_lcao_kpts_many_combinations(in_tmp_dir):
    counter = count()

    for spinpol in [False, True]:
        # We want a non-trivial cell:
        atoms0 = bulk('Ti') * (2, 1, 1)
        atoms0.cell[0] *= 1.2
        # We want most arrays to be different so we can detect ordering/shape
        # trouble:
        atoms0.symbols = 'HOFePb'
        atoms0.rattle(stdev=0.1)

        if spinpol:
            atoms0.set_initial_magnetic_moments([1.0] * len(atoms0))

        kwargs = []
        energies = []
        eerrs = []
        forces = []
        ferrs = []

        from time import time
        for kwargs in ikwargs():
            i = next(counter)

            if world.rank == 0:
                print(i, kwargs)

            if kwargs['parallel']['kpt'] == 4 and not spinpol:
                continue  # Core without kpoints

            calc = GPAW(
                mode='lcao',
                basis='sz(dzp)',
                xc='PBE', h=0.3,
                symmetry={'point_group': False},  # No symmetry here anyway
                txt='gpaw.{:02d}.spin{}.txt'.format(int(spinpol), i),
                kpts=(4, 1, 1),
                convergence={'maximum iterations': 2},
                **kwargs)
            atoms = atoms0.copy()
            t1 = time()
            atoms.calc = calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            t2 = time()
            if world.rank == 0:
                print('T', t2 - t1)
            energies.append(e)
            forces.append(f)
            corrname = calc.wfs.atomic_correction.name
            if kwargs['parallel']['sl_auto']:
                assert corrname == 'sparse'
            else:
                assert corrname == 'dense'

            if energies:
                eerr = abs(e - energies[0])
                ferr = abs(f - forces[0]).max()
                eerrs.append(eerr)
                ferrs.append(ferr)
                if world.rank == 0:
                    print('Eerr {} Ferr {}'.format(eerr, ferr))
                    print()

        if world.rank == 0:
            print('eerrs', eerrs)
            print('ferrs', ferrs)

            maxeerr = max(eerrs)
            maxferr = max(ferrs)
            print('maxeerr', maxeerr)
            print('maxferr', maxferr)
            assert maxeerr < 1e-9, maxeerr
            assert maxferr < 1e-10, maxferr
