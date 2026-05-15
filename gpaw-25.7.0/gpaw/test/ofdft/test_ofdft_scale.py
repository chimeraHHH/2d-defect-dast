import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.mixer import Mixer
from gpaw.test import gen
from gpaw.eigensolvers import CG


@pytest.mark.ofdft
@pytest.mark.libxc
def test_ofdft_ofdft_scale(in_tmp_dir):
    h = 0.18
    a = 10.0
    c = a / 2

    elements = ['C']
    results = [0.016]
    electrons = [6]
    lambda_coeff = 2.0
    xcname = '1.0_LDA_K_TF+1.0_LDA_X'

    setups = {}
    for symbol in elements:
        g = gen(symbol, xcname=xcname, scalarrel=False, orbital_free=True,
                tw_coeff=lambda_coeff)
        setups[symbol] = g

    for element, result, e in zip(elements, results, electrons):
        atom = Atoms(element,
                     positions=[(c, c, c)],
                     cell=(a, a, a))

        mixer = Mixer(0.3, 5, 1)
        eigensolver = CG(tw_coeff=lambda_coeff)
        calc = GPAW(mode='fd',
                    h=h,
                    xc=xcname,
                    setups=setups,
                    maxiter=240,
                    mixer=mixer, eigensolver=eigensolver)

        atom.calc = calc

        E = atom.get_total_energy()
        n = calc.get_all_electron_density()

        dv = atom.get_volume() / calc.get_number_of_grid_points().prod()
        I = n.sum() * dv / 2**3

        assert I == pytest.approx(e, abs=1.0e-6)
        assert result == pytest.approx(E, abs=1.0e-3)
