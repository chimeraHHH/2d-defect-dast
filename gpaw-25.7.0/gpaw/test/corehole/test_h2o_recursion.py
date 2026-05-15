from math import cos, pi, sin

import pytest
from ase import Atom, Atoms

from gpaw import GPAW, Davidson, Mixer, PoissonSolver
from gpaw.test import gen


@pytest.mark.old_gpaw_only
def test_corehole_h2o_recursion(in_tmp_dir):
    # Generate setup for oxygen with half a core-hole:
    s = gen('O', name='hch1s', corehole=(1, 0, 0.5))

    a = 5.0
    d = 0.9575
    t = pi / 180 * 104.51
    H2O = Atoms([Atom('O', (0, 0, 0)),
                 Atom('H', (d, 0, 0)),
                 Atom('H', (d * cos(t), d * sin(t), 0))],
                cell=(a, a, a), pbc=False)
    H2O.center()
    calc = GPAW(mode='fd', nbands=10, h=0.2, setups={'O': s},
                eigensolver=Davidson(4),
                mixer=Mixer(0.5),
                xc='oldLDA',
                poissonsolver=PoissonSolver(use_charge_center=True))
    H2O.calc = calc
    e = H2O.get_potential_energy()
    niter = calc.get_number_of_iterations()

    from gpaw.xas import RecursionMethod

    if 1:
        r = RecursionMethod(calc)
        r.run(400)
        r.write('h2o.pckl')
    else:
        r = RecursionMethod(filename='h2o.pckl')

    print(e, niter)
    energy_tolerance = 0.002
    assert e == pytest.approx(-17.980, abs=energy_tolerance)
