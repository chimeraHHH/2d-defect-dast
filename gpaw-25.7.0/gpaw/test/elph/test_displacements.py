"""Basic test of elph/DisplacementRunner

The main functionality is imported from ase.phonons.Displacement,
so won't test this here. Just test, whether the potential is reliably
extracted.
"""
import numpy as np
import pytest

from ase.build import bulk
from ase.utils.filecache import MultiFileJSONCache

from gpaw import GPAW
from gpaw.elph import DisplacementRunner
from gpaw.mpi import world


@pytest.mark.elph
def test_displacements(in_tmp_dir):
    # 2 atoms with one 1 valence electron each
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    assert len(atoms) == 2
    # Note: usually we need to disable point group symmetry for displacements,
    # but we won't run actual displacements, so it saves time not to bother.
    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                # need more than one point for proper test
                kpts={'size': (2, 2, 2), 'gamma': True},
                # symmetry={'point_group': False},
                # parallel={'sl_auto': True, 'augment_grids':True,
                #          'band': 1, 'kpt': 1, 'domain': 4 },
                txt='elph_displacements.txt')
    atoms.calc = calc

    elph = DisplacementRunner(atoms=atoms, calc=atoms.calc,
                              supercell=(1, 1, 1), name='elph',
                              calculate_forces=False)
    elph.indices = []
    elph.run()
    del elph

    # read stuff back
    if world.rank == 0:
        cache = MultiFileJSONCache('elph')
        info = cache['info']
        assert info['supercell'] == [1, 1, 1]
        assert info['natom'] == 2
        assert info['delta'] == 0.01
        assert 'dr_version' in info

        # the following might change if defaults are changed
        Vt_G = cache['eq']['Vt_sG'][0]
        # print(Vt_G.shape, np.min(Vt_G), np.max(Vt_G))
        assert Vt_G.shape == (16, 16, 16)
        assert pytest.approx(np.min(Vt_G), abs=1e-6) == -0.5636114
        assert pytest.approx(np.max(Vt_G), abs=1e-6) == -0.1254635

        dH_all_asp = cache['eq']['dH_all_asp']
        # print(np.min(dH_all_asp[0][0]), np.max(dH_all_asp[0][0]))
        assert len(dH_all_asp) == 2
        dH_p = dH_all_asp[0][0]
        assert dH_p.shape == (15,)
        assert pytest.approx(np.min(dH_p), abs=1e-6) == -0.2249569
        assert pytest.approx(np.max(dH_p), abs=1e-6) == 0.0335627
