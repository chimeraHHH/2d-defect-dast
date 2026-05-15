"""Test read/write of restart files between fd and lcao mode"""
import os

import pytest
from ase import Atom, Atoms

from gpaw import GPAW, FermiDirac, restart


@pytest.mark.old_gpaw_only
def test_lcao_fd2lcao_restart(in_tmp_dir):
    energy_tolerance = 0.001

    if not os.path.isfile('Na4_fd.gpw'):
        # Do grid kpts calculation
        a = 3.31
        atoms = Atoms([Atom('Na', (i * a, 0, 0))
                       for i in range(4)], pbc=(1, 0, 0))
        atoms.center(vacuum=3.5)
        atoms.center(vacuum=a / 2, axis=0)

        calc = GPAW(mode='fd',
                    nbands=-3,
                    h=0.3,
                    setups={'Na': '1'},
                    xc={'name': 'PBE', 'stencil': 1},
                    occupations=FermiDirac(width=0.1),
                    kpts=(3, 1, 1),
                    # basis='dzp',
                    txt='Na4_fd.txt')
        atoms.calc = calc
        etot_fd = atoms.get_potential_energy()
        print('Etot:', etot_fd, 'eV in fd-mode')
        calc.write('Na4_fd.gpw')
        del atoms, calc

        assert etot_fd == pytest.approx(-1.99055, abs=energy_tolerance)

    if os.path.isfile('Na4_fd.gpw'):
        # LCAO calculation based on grid kpts calculation
        atoms, calc = restart('Na4_fd.gpw',
                              # basis='dzp',
                              mode='lcao',
                              txt='Na4_lcao.txt')
        etot_lcao = atoms.get_potential_energy()
        print('Etot:', etot_lcao, 'eV in lcao-mode')
        calc.write('Na4_lcao.gpw')
        del atoms, calc

        assert etot_lcao == pytest.approx(-1.9616, abs=energy_tolerance)
