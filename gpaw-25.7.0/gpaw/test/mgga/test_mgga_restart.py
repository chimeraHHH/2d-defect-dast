import pytest
from ase import Atom, Atoms
from gpaw import GPAW


@pytest.mark.mgga
def test_mgga_mgga_restart(in_tmp_dir):
    fname = 'H2_PBE.gpw'
    fwfname = 'H2_wf_PBE.gpw'
    txt = None
    txt = '-'

    s = Atoms([Atom('H'), Atom('H', [0, 0, 1])])
    s.center(vacuum=3.)
    s.calc = GPAW(xc={'name': 'PBE', 'stencil': 1},
                  mode='fd',
                  h=.3,
                  convergence={'density': 1e-4, 'eigenstates': 1e-6})
    s.get_potential_energy()
    s.calc.write(fname)
    s.calc.write(fwfname, 'all')

    # full information
    calc = GPAW(fwfname, txt=txt)
    E_PBE = calc.get_potential_energy(s)
    dE = calc.get_xc_difference({'name': 'TPSS', 'stencil': 1})
    E_1 = E_PBE + dE
    print('E PBE, TPSS=', E_PBE, E_1)

    # no wfs
    calc = GPAW(fname, txt=txt)
    E_PBE_no_wfs = calc.get_potential_energy(s)
    dE = calc.get_xc_difference({'name': 'TPSS', 'stencil': 1})
    E_2 = E_PBE_no_wfs + dE
    print('E PBE, TPSS=', E_PBE_no_wfs, E_2)

    print('diff=', E_1 - E_2)
    assert abs(E_1 - E_2) < 0.005

    energy_tolerance = 0.002
    assert E_PBE == pytest.approx(-5.341, abs=energy_tolerance)
    assert E_PBE_no_wfs == pytest.approx(-5.33901, abs=energy_tolerance)
    assert E_1 == pytest.approx(-5.57685, abs=energy_tolerance)
    assert E_2 == pytest.approx(-5.57685, abs=energy_tolerance)
