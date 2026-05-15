import pytest
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.mixer import MixerSum


def test_fixmom():
    a = 2.87
    bulk = Atoms('Fe2',
                 scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
                 magmoms=[2.20, 2.20],
                 cell=(a, a, a),
                 pbc=True)
    mom0 = sum(bulk.get_initial_magnetic_moments())
    conv = {'eigenstates': 0.1, 'density': 0.001, 'energy': 0.01}
    calc = GPAW(mode='pw',
                mixer=MixerSum(0.1, 3),
                nbands=11,
                kpts=(3, 3, 3),
                convergence=conv,
                occupations=FermiDirac(0.1, fixmagmom=True))
    bulk.calc = calc
    bulk.get_potential_energy()
    mom = bulk.get_magnetic_moment()
    assert mom == pytest.approx(mom0, abs=0.05)
