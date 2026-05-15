from ase import Atoms
from ase.units import Hartree
from gpaw import GPAW, PoissonSolver, FermiDirac, Davidson, MixerSum
import pytest


def test_xc_revPBE_Li():
    a = 5.0
    n = 24
    li = Atoms('Li', magmoms=[1.0], cell=(a, a, a), pbc=True)

    params = dict(
        mode='fd',
        gpts=(n, n, n),
        nbands=1,
        xc=dict(name='oldPBE', stencil=1),
        poissonsolver=PoissonSolver(),
        mixer=MixerSum(0.6, 5, 10.0),
        eigensolver=Davidson(4),
        convergence=dict(eigenstates=4.5e-8),
        occupations=FermiDirac(0.0))

    li.calc = GPAW(**params)
    e = li.get_potential_energy() + li.calc.get_reference_energy()
    assert e == pytest.approx(-7.462 * Hartree, abs=1.4)

    params['xc'] = dict(name='revPBE', stencil=1)
    li.calc = GPAW(**params)
    erev = li.get_potential_energy() + li.calc.get_reference_energy()

    assert erev == pytest.approx(-7.487 * Hartree, abs=1.3)
    assert e - erev == pytest.approx(0.025 * Hartree, abs=0.002 * Hartree)

    print(e, erev)
    energy_tolerance = 0.002
    assert e == pytest.approx(-204.381098849, abs=energy_tolerance)
    assert erev == pytest.approx(-205.012303379, abs=energy_tolerance)
