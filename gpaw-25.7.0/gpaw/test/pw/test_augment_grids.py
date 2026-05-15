import numpy as np
import pytest
from ase import Atoms

from gpaw import PW, FermiDirac
from gpaw.calculator import GPAW
from gpaw.new.calculation import DFTCalculation
from gpaw.mpi import rank


@pytest.mark.old_gpaw_only
@pytest.mark.stress
def test_pw_augment_grids(in_tmp_dir, gpaw_new):
    ecut = 200
    kpoints = [1, 1, 4]
    atoms = Atoms('HLi', cell=[6, 6, 3.4], pbc=True,
                  positions=[[3, 3, 0], [3, 3, 1.6]])

    def calculate(aug):
        if gpaw_new:
            dft = DFTCalculation.from_parameters(
                atoms,
                dict(mode=PW(ecut),
                     parallel={'augment_grids': aug},
                     kpts={'size': kpoints},
                     occupations=FermiDirac(width=0.1)),
                log=f'gpaw.aug{aug}.txt')
            dft.converge(steps=4)
            dft.energy()
            dft.forces()
            dft.stress()
            e = dft.results['energy']
            f = dft.results['forces']
            s = dft.results['stress']
        else:
            atoms.calc = GPAW(mode=PW(ecut),
                              txt=f'gpaw.aug{aug}.txt',
                              parallel={'augment_grids': aug},
                              kpts={'size': kpoints},
                              occupations=FermiDirac(width=0.1),
                              convergence={'maximum iterations': 4})
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            s = atoms.get_stress()
        return e, f, s

    e1, f1, s1 = calculate(False)
    e2, f2, s2 = calculate(True)

    eerr = abs(e2 - e1)
    ferr = np.abs(f2 - f1).max()
    serr = np.abs(s2 - s1).max()
    if rank == 0:
        print('errs', eerr, ferr, serr)
    assert eerr < 5e-12, f'bad energy: err={eerr}'
    assert ferr < 5e-12, f'bad forces: err={ferr}'
    assert serr < 5e-12, f'bad stress: err={serr}'
