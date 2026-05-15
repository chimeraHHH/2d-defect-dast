import pytest
from ase.build import molecule

from gpaw import GPAW, FermiDirac, PoissonSolver
from gpaw.test import gen


@pytest.mark.old_gpaw_only
def test_corehole_h2o_dks():
    def xc(name):
        return {'name': name, 'stencil': 1}

    # Generate setup for oxygen with a core-hole:
    s = gen('O', name='fch1s', xcname='PBE', corehole=(1, 0, 1.0))

    atoms = molecule('H2O')
    atoms.center(vacuum=2.5)

    params = dict(mode='fd',
                  xc=xc('PBE'),
                  poissonsolver=PoissonSolver('fd',
                                              use_charge_center=True))
    atoms.calc = GPAW(**params)
    e1 = atoms.get_potential_energy() + atoms.calc.get_reference_energy()

    atoms[0].magmom = 1
    atoms.calc = GPAW(charge=-1,
                      setups={'O': s},
                      occupations=FermiDirac(0.0, fixmagmom=True),
                      **params)
    e2 = atoms.get_potential_energy() + atoms.calc.get_reference_energy()

    atoms[0].magmom = 0
    atoms.calc = GPAW(charge=0,
                      setups={'O': s},
                      occupations=FermiDirac(0.0, fixmagmom=True),
                      spinpol=True,
                      **params)
    e3 = atoms.get_potential_energy() + atoms.calc.get_reference_energy()

    print('Energy difference %.3f eV' % (e2 - e1))
    print('XPS %.3f eV' % (e3 - e1))

    print(e2 - e1)
    print(e3 - e1)
    assert abs(e2 - e1 - 533.070) < 0.001
    assert abs(e3 - e1 - 538.549) < 0.001

    energy_tolerance = 0.02
    print(e1)
    print(e2)
    print(e3)
    assert e1 == pytest.approx(-2080.3651, abs=energy_tolerance)
    assert e2 == pytest.approx(-1547.2944, abs=energy_tolerance)
    assert e3 == pytest.approx(-1541.8152, abs=energy_tolerance)
