# Test loading of sg15 setups as setups='sg15' and that the calculation
# agrees with PAW for the H2 eigenvalue.

from ase.build import molecule
from gpaw import GPAW, Davidson, Mixer


def test_pseudopotential_sg15_hydrogen(sg15_hydrogen):
    system = molecule('H2')
    system.center(vacuum=2.5)

    def getkwargs():
        return dict(mode='fd',
                    eigensolver=Davidson(4),
                    mixer=Mixer(0.8, 5, 10.0),
                    xc='oldPBE')

    calc1 = GPAW(setups={'H': sg15_hydrogen}, h=0.13, **getkwargs())
    system.calc = calc1
    system.get_potential_energy()
    eps1 = calc1.get_eigenvalues()

    calc2 = GPAW(h=0.2, **getkwargs())
    system.calc = calc2
    system.get_potential_energy()
    eps2 = calc2.get_eigenvalues()

    err = eps2[0] - eps1[0]

    # It is not the most accurate calculation ever, let's just make sure things
    # are not completely messed up.
    print('sg15 vs paw error', err)
    assert abs(err) < 0.02  # 0.0055.... as of current test.
