"""Basic test of elph/supercell/calculate_gradient

Check that calculate_gradient is capable of reading stuff properly.
"""
import ase.units as units
import numpy as np
import pytest
from ase.build import bulk
from gpaw import GPAW
from gpaw.elph import DisplacementRunner, Supercell
from gpaw.mpi import world


@pytest.mark.elph
def test_gradient(in_tmp_dir):
    # 2 atoms with one 1 valence electron each
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    assert len(atoms) == 2
    # Note: usually we need to disable point group symmetry for displacements,
    # but we won't run actual displacements, so it saves time not to bother.
    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts={'size': (2, 2, 2), 'gamma': False},
                # symmetry={'point_group': False},
                txt='elph_displacements.txt')
    atoms.calc = calc

    elph = DisplacementRunner(atoms=atoms, calc=atoms.calc,
                              supercell=(1, 1, 1), name='elph',
                              calculate_forces=False)
    elph.indices = []
    elph.run()

    # Barrier will be included in elph.run() in the future:
    # https://gitlab.com/ase/ase/-/merge_requests/2903
    world.barrier()

    Vt_sG = elph.cache['eq']['Vt_sG']
    dH_all_asp = elph.cache['eq']['dH_all_asp']

    # create displaced entries without calculation to save time
    V1expected = []
    for a in (0, 1):
        for i in range(3):
            V1expected.append(2 * 0.1 * (a + 1) * (i + 1) /
                              (2 * 0.01 / units.Bohr))
            for sign in [-1, 1]:
                disp = elph._disp(a, i, sign)
                with elph.cache.lock(disp.name) as handle:
                    if handle is None:
                        continue
                    try:
                        Vfake = Vt_sG + sign * 0.1 * (a + 1) * (i + 1)
                        # dHfake = {}
                        # for atom in dH_all_asp.keys():
                        #    dHfake[atom] = dH_all_asp[atom] + (sign * 0.2 *
                        #                                       (a+1) * (i+1))
                        result = {'Vt_sG': Vfake, 'dH_all_asp': dH_all_asp}
                        handle.save(result)
                    finally:
                        pass

    # calculate gradient
    V1t_xsG, dH1_xasp = Supercell.calculate_gradient('elph')
    assert np.array(V1expected) == pytest.approx(V1t_xsG[:, 0, 0, 0, 0])
    # should probably add something about dH as well, but the principle is the
    # same
