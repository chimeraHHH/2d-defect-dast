"""Test automagical calculation of wfs"""

from gpaw import GPAW
from ase import Atoms

# H2


def test_fileio_wfs_auto(in_tmp_dir):
    H = Atoms('HH', [(0, 0, 0), (0, 0, 1)])
    H.center(vacuum=2.0)

    calc = GPAW(mode='fd',
                nbands=2,
                convergence={'eigenstates': 1e-3})
    H.calc = calc
    H.get_potential_energy()
    calc.write('tmp')

    calc = GPAW('tmp')
    calc.converge_wave_functions()
