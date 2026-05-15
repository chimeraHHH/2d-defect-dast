import pytest
from ase.build import bulk
from gpaw import GPAW
from gpaw.mixer import Mixer
from gpaw.test import gen


@pytest.mark.ofdft
@pytest.mark.libxc
def test_ofdft_ofdft_pbc(in_tmp_dir):
    symbol = 'C'
    result = -224.206
    electrons = 48

    xcname = 'LDA_K_TF+LDA_X'
    g = gen(symbol, xcname=xcname, scalarrel=False, orbital_free=True)
    h = 0.14
    a = 2.8
    atoms = bulk(symbol, 'diamond', a=a, cubic=True)   # Generate diamond
    mixer = Mixer(0.1, 5)

    calc = GPAW(mode='fd',
                h=h,
                xc=xcname,
                setups={'C': g},
                maxiter=120,
                eigensolver='cg',
                mixer=mixer)

    atoms.calc = calc

    e = atoms.get_potential_energy()

    n = calc.get_all_electron_density()

    dv = atoms.get_volume() / calc.get_number_of_grid_points().prod()
    I = n.sum() * dv / 2**3

    assert I == pytest.approx(electrons, abs=1.0e-6)
    assert e == pytest.approx(result, abs=1.0e-2)
