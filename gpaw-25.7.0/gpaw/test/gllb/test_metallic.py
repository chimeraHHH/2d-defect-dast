import pytest
import numpy as np
from ase.build import bulk
from gpaw import GPAW


def run(xc, repeat=1):
    atoms = bulk('Ag') * repeat
    k = 4 // repeat
    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                h=0.3,
                setups={'Ag': '11'},
                nbands=6 * repeat**3,
                xc=xc,
                parallel={'domain': 1},
                kpts={'size': (k, k, k), 'gamma': False},
                txt='-')
    atoms.calc = calc
    atoms.get_potential_energy()
    x_i, y_i = calc.get_dos(npts=1001)
    x_i -= calc.get_fermi_level()
    y_i /= repeat**3
    return x_i, y_i


@pytest.mark.gllb
@pytest.mark.libxc
def test_metallic_GLLBSCM():
    # GLLBSC should behave like GLLBSCM for metals and
    # repeated cell should give exactly the same results
    x1_i, y1_i = run(xc='GLLBSCM', repeat=1)
    for x_i, y_i in [run(xc='GLLBSC', repeat=1),
                     run(xc='GLLBSCM', repeat=2)]:
        # Test that the DOSes are the same
        assert np.allclose(x1_i, x_i, rtol=0, atol=1e-8), \
            "DOS energies don't match, " \
            "error = {}".format(np.max(np.abs(x1_i - x_i)))
        assert np.allclose(y1_i, y_i, rtol=0, atol=1e-6), \
            "DOS values don't match, " \
            "error = {}".format(np.max(np.abs(y1_i - y_i)))


if __name__ == '__main__':
    test_metallic_GLLBSCM()
