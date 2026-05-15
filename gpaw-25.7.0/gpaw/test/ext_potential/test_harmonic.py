import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Ha

from gpaw import GPAW
from gpaw.external import ExternalPotential, known_potentials
from gpaw.poisson import NoInteractionPoissonSolver


class HarmonicPotential(ExternalPotential):
    def calculate_potential(self, gd):
        a = gd.cell_cv[0, 0]
        r_vg = gd.get_grid_point_coordinates()
        self.vext_g = 0.5 * ((r_vg - a / 2)**2).sum(0)

    def todict(self):
        return {'name': 'HarmonicPotential'}


@pytest.mark.old_gpaw_only
def test_ext_potential_harmonic(in_tmp_dir):
    """Test against the analytic result (no xc, no Coulomb)."""
    a = 4.0
    x = Atoms(cell=(a, a, a))  # no atoms

    calc = GPAW(mode='fd',
                charge=-8,
                nbands=4,
                h=0.2,
                xc={'name': 'null'},
                external=HarmonicPotential(),
                poissonsolver=NoInteractionPoissonSolver(),
                eigensolver='cg')

    x.calc = calc
    x.get_potential_energy()

    eigs = calc.get_eigenvalues()
    assert eigs[0] == pytest.approx(1.5 * Ha, abs=0.002)
    assert abs(eigs[1:] - 2.5 * Ha).max() == pytest.approx(0, abs=0.003)

    # Check write + read:
    calc.write('harmonic.gpw')
    known_potentials['HarmonicPotential'] = HarmonicPotential
    GPAW('harmonic.gpw')


class PöschlTellerPotential(ExternalPotential):
    """Slab with Pöschel-Teller well along z-direction.

    See:

        https://en.wikipedia.org/wiki/P%C3%B6schl%E2%80%93Teller_potential
    """
    def calculate_potential(self, gd):
        a = gd.cell_cv[2, 2]
        r_vg = gd.get_grid_point_coordinates()
        lam = 2
        self.vext_g = -lam * (lam + 1) / 2 * np.cosh(r_vg[2] - a / 2)**-2

    def todict(self):
        return {'name': 'HarmonicPotential'}


@pytest.mark.old_gpaw_only
def test_pt_potential():
    """Test againts analytic result (no xc, no Coulomb)."""
    d = 6.0
    a = 2
    x = Atoms(cell=(a, a, d), pbc=[1, 1, 0])  # no atoms

    calc = GPAW(charge=-12,
                nbands=8,
                mode='pw',
                xc={'name': 'null'},
                external=PöschlTellerPotential(),
                poissonsolver=NoInteractionPoissonSolver())

    x.calc = calc
    x.get_potential_energy()

    eigs = calc.get_eigenvalues() / Ha
    print(eigs)
    k = 2 * np.pi / (a / Bohr)
    e0 = -2
    e1234 = -2 + 0.5 * k**2
    e5 = -0.5
    assert eigs[0] == pytest.approx(e0, abs=0.0002)
    assert eigs[1] == pytest.approx(e1234, abs=0.001)
    assert np.ptp(eigs[1:5]) == pytest.approx(0)
    assert eigs[5] == pytest.approx(e5, abs=0.001)
