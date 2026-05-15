"""Unit test for elph/supercell and elph/gmatrix checking
whether they properly respect the self.indices.

Note: If the users gives an index twice that's passed through, because that's
      how ase.phonons does it.

"""
import numpy as np
import pytest

from ase.build import bulk
# from ase.phonons import Phonons
from gpaw.elph import Supercell, ElectronPhononMatrix
# from gpaw.mpi import world


class FakeElectronPhononMatrix(ElectronPhononMatrix):
    def _set_supercell_cache(self, supercell_cache: str,
                             load_sc_as_needed: bool):
        pass

    def _read_phonon_cache(self):
        pass


def _supercell_all(atoms):
    natoms = len(atoms)
    sc = Supercell(atoms)
    assert len(sc.indices) == natoms
    assert sc.indices[0] == 0
    assert sc.indices[-1] == natoms - 1


def _gmatrix_all(atoms):
    natoms = len(atoms)
    fepm = FakeElectronPhononMatrix(atoms, None, {"name": "phonon"})
    assert len(fepm.indices) == natoms
    assert fepm.indices[0] == 0
    assert fepm.indices[-1] == natoms - 1


def _supercell_indices(atoms, indices):
    natoms = len(atoms)
    nindices = len(indices)
    sci = Supercell(atoms, indices=indices)
    assert len(sci.indices) != natoms
    assert len(sci.indices) == nindices
    assert sci.indices[0] == indices[0]
    assert sci.indices[-1] == indices[-1]


def _gmatrix_indices(atoms, indices):
    natoms = len(atoms)
    nindices = len(indices)
    fepm = FakeElectronPhononMatrix(atoms, None, {"name": "phonon"},
                                    indices=indices)
    assert len(fepm.indices) != natoms
    assert len(fepm.indices) == nindices
    assert fepm.indices[0] == indices[0]
    assert fepm.indices[-1] == indices[-1]


@pytest.mark.elph
def test_indices():

    # arbitrary atoms object
    n = 3
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True) * (n, n, n)
    assert len(atoms) == 2 * n**3

    # case 1: self.indices not given
    _supercell_all(atoms)
    _gmatrix_all(atoms)

    # TODO: Supercell.calculate_gradient()? Would need fake cache.
    # TODO: Supercell.calculate_supercell_matrix()? Would need fake cache and
    #       stuff.

    # case 2: gives some random indices
    seed = 123456
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(atoms),
                           rng.integers(0, len(atoms) - 1))
    _supercell_indices(atoms, indices)
    _gmatrix_indices(atoms, indices)
    # TODO: Supercell.calculate_gradient()? Would need fake cache.
    # TODO: Supercell.calculate_supercell_matrix()? Would need fake cache
    #       and stuff.
