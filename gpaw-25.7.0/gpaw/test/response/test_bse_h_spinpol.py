import pytest
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.response.bse import BSE


@pytest.mark.response
@pytest.mark.serial
def test_bse_spinpol(in_tmp_dir):
    atoms = Atoms('H', magmoms=[1], pbc=True)
    atoms.center(vacuum=1.5)
    atoms.calc = GPAW(mode=PW(180, force_complex_dtype=True),
                      nbands=6,
                      convergence={'bands': 4})
    atoms.get_potential_energy()

    gw_kn = np.zeros((1, 4))
    gw_kn[0] = [-10, 2, 2, 4]

    bse = BSE(atoms.calc,
              ecut=10,
              nbands=2,
              gw_kn=gw_kn,
              valence_bands=[0],
              conduction_bands=[1, 2, 3])

    bsematrix = bse.get_bse_matrix()
    w_T, _, _ = bse.diagonalize_bse_matrix(bsematrix)
    assert w_T[0] == pytest.approx(0.013, abs=0.001)
