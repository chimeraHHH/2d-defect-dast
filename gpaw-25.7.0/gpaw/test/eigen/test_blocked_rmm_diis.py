from ase import Atom, Atoms

from gpaw import GPAW, Mixer, RMMDIIS
import pytest


def test_eigen_blocked_rmm_diis(in_tmp_dir):
    a = 4.0
    n = 20
    d = 1.0
    x = d / 3**0.5
    atoms = Atoms([Atom('C', (0.0, 0.0, 0.0)),
                   Atom('H', (x, x, x)),
                   Atom('H', (-x, -x, x)),
                   Atom('H', (x, -x, -x)),
                   Atom('H', (-x, x, -x))],
                  cell=(a, a, a), pbc=True)
    base_params = dict(
        mode='fd',
        gpts=(n, n, n),
        nbands=4,
        mixer=Mixer(0.25, 3, 1))
    calc = GPAW(**base_params, txt='a.txt', eigensolver='rmm-diis')
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    niter0 = calc.get_number_of_iterations()

    es = RMMDIIS(blocksize=3)
    calc = GPAW(**base_params, txt='b.txt', eigensolver=es)
    atoms.calc = calc
    e1 = atoms.get_potential_energy()
    niter1 = calc.get_number_of_iterations()
    assert e0 == pytest.approx(e1, abs=0.000001)
    assert niter0 == pytest.approx(niter1, abs=0)
