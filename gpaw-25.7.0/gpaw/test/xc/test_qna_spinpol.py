import pytest
from ase.atoms import Atoms

from gpaw import GPAW, PW


@pytest.mark.old_gpaw_only
def test_xc_qna_spinpol(in_tmp_dir):
    QNA = {'alpha': 2.0,
           'name': 'QNA',
           'orbital_dependent': False,
           'parameters': {'H': (0.1485, 0.005)},
           'setup_name': 'PBE',
           'type': 'qna-gga'}

    atoms = Atoms('H', pbc=True, cell=[5, 5, 5])
    atoms.set_initial_magnetic_moments([1])

    calc = GPAW(mode=PW(400),
                kpts=(1, 1, 1),
                xc=QNA,
                parallel={'domain': 1},
                txt='qna_spinpol.txt')

    atoms.calc = calc
    atoms.get_potential_energy()
    magmoms = atoms.get_magnetic_moments()

    tol = 0.003
    assert 0.25374 == pytest.approx(magmoms[0], abs=tol)
