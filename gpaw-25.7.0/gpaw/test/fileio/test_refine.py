"""Test automatically write out of restart files"""

import pytest
from ase import Atoms

from gpaw import GPAW


@pytest.mark.old_gpaw_only
def test_fileio_refine(in_tmp_dir):
    restart_wf = 'gpaw-restart-wf.gpw'
    # H2
    H = Atoms('HH', [(0, 0, 0), (0, 0, 1)])
    H.center(vacuum=2.0)

    if 1:
        calc = GPAW(mode='fd',
                    nbands=2,
                    convergence={'eigenstates': 0.001,
                                 'energy': 0.1,
                                 'density': 0.1})
        H.calc = calc
        H.get_potential_energy()
        calc.write(restart_wf, 'all')

        # refine the result directly
        H.calc = calc.new(convergence={'energy': 0.00001})
        Edirect = H.get_potential_energy()

    # refine the result after reading from a file
    H = GPAW(restart_wf, convergence={'energy': 0.00001}).get_atoms()
    Erestart = H.get_potential_energy()

    print(Edirect, Erestart)
    # Note: the different density mixing introduces small differences
    assert Edirect == pytest.approx(Erestart, abs=4e-5)
