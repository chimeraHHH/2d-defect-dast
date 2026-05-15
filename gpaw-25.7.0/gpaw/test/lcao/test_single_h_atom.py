import pytest
from ase import Atoms
from gpaw import GPAW


def test_h_atom(in_tmp_dir):
    """Test lcao with magmom."""
    h = Atoms('H', magmoms=[0.1])
    h.center(vacuum=2)
    h.calc = GPAW(txt='h.txt',
                  mode='lcao',
                  basis='dz(dzp)',
                  h=0.2,
                  nbands=2)
    eref = -13.02531 - (-12.128958)  # lda spin-polarized - lda spin-paired
    assert h.get_potential_energy() == pytest.approx(eref, abs=0.12)
    assert h.get_magnetic_moment() == pytest.approx(1.0, abs=1e-4)
