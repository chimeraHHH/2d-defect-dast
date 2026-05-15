import pytest
from ase import Atoms

from gpaw import GPAW, LCAO
from gpaw.poisson import FDPoissonSolver


def test_lcao_h2o():
    a = 6.0
    b = a / 2
    mol = Atoms('OHH',
                [(b, b, 0.1219 + b),
                 (b, 0.7633 + b, -0.4876 + b),
                 (b, -0.7633 + b, -0.4876 + b)],
                pbc=False, cell=[a, a, a])
    calc = GPAW(gpts=(32, 32, 32),
                nbands=4,
                mode='lcao',
                poissonsolver=FDPoissonSolver())
    mol.calc = calc
    e = mol.get_potential_energy()
    niter = calc.get_number_of_iterations()

    assert e == pytest.approx(-10.271, abs=2e-3)
    assert niter == pytest.approx(8, abs=1)

    # Check that complex wave functions are allowed with
    # gamma point calculations
    calc = GPAW(gpts=(32, 32, 32),
                nbands=4,
                mode=LCAO(force_complex_dtype=True),
                poissonsolver=FDPoissonSolver())
    mol.calc = calc
    ec = mol.get_potential_energy()
    assert e == pytest.approx(ec, abs=1e-5)
