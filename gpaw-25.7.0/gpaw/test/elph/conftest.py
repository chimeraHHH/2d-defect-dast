import pytest

from ase.build import bulk

from gpaw import GPAW

from gpaw.elph import DisplacementRunner
from gpaw.elph import Supercell

SUPERCELL = (2, 1, 1)


def get_calc(txt, parallel={}):
    return GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts={'size': (1, 2, 2), 'gamma': False},
                symmetry={'point_group': False},
                convergence={'forces': 1.e-4},
                parallel=parallel,
                txt=txt)


# NOTE: This fixture might need a proper scope assigned
@pytest.fixture(scope='module')
def elph_cache(module_tmp_path):
    """Minimum elph cache for Li

    Uses 1x2x2 k-points and 2x1x1 SC to allow for parallelisaiton
    test.
    Takes 6s on 4 cores.
    """
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    calc = get_calc(txt='elph_li.txt')
    atoms.calc = calc
    elph = DisplacementRunner(atoms, calc,
                              supercell=SUPERCELL, name='elph',
                              calculate_forces=True)
    elph.run()
    return elph


# NOTE: This fixture might need a proper scope assigned
@pytest.fixture(scope='module')
def supercell_cache(module_tmp_path, elph_cache):
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    atoms_N = atoms * SUPERCELL
    elph_cache
    calc = get_calc(parallel={'domain': 1, 'band': 1},
                    # parallel={'sl_auto': True, 'augment_grids':True,
                    #          'band': 2, 'kpt': 1, 'domain': 1 },
                    txt='gs_li.txt')
    atoms_N.calc = calc
    atoms_N.get_potential_energy()

    # create supercell cache
    sc = Supercell(atoms, supercell=SUPERCELL)
    sc.calculate_supercell_matrix(calc)
    return sc
