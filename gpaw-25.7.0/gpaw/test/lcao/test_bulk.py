import numpy as np
from ase import Atoms
from gpaw import GPAW
import pytest


def test_lcao_bulk(in_tmp_dir):
    bulk = Atoms('Li', pbc=True)
    k = 4
    g = 8
    calc = GPAW(gpts=(g, g, g), kpts=(k, k, k),
                mode='lcao', basis='dzp')
    bulk.calc = calc
    e = []
    niter = []
    A = [2.6, 2.65, 2.7, 2.75, 2.8]
    for a in A:
        bulk.set_cell((a, a, a))
        e.append(bulk.get_potential_energy())
        niter.append(calc.get_number_of_iterations())

    a = np.roots(np.polyder(np.polyfit(A, e, 2), 1))[0]
    print('a =', a)
    assert a == pytest.approx(2.6378, abs=0.0003)

    e_ref = [-1.8677343236247692, -1.8690343169380492, -1.8654175796625045,
             -1.8566274574918875, -1.8432374955346396]
    niter_ref = [6, 6, 6, 6, 6]

    print(e)
    energy_tolerance = 0.0003
    niter_tolerance = 0

    for i in range(len(A)):
        assert e[i] == pytest.approx(e_ref[i], abs=energy_tolerance)
        assert niter[i] == pytest.approx(niter_ref[i], abs=niter_tolerance)

    wf1 = calc.get_pseudo_wave_function(kpt=3, band=0)
    calc.write('Li', mode='all')
    calc2 = GPAW('Li')
    # calc2.initialize_positions()
    wf2 = calc2.get_pseudo_wave_function(kpt=3, band=0)
    assert abs(wf1 - wf2).max() == pytest.approx(0, abs=1e-9)
