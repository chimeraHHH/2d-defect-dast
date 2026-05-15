import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.mixer import Mixer
from gpaw.test import gen


@pytest.mark.ofdft
@pytest.mark.libxc
def test_ofdft_ofdft(in_tmp_dir):
    a = 6.0
    c = a / 2
    # d = 1.8

    elements = ['C', 'Be']
    results = [0.243, 9.9773]
    electrons = [6, 3]
    charges = [0, 1]
    xcname = '1.0_LDA_K_TF+1.0_LDA_X'

    setups = {}
    for symbol in elements:
        s = gen(symbol, xcname=xcname, scalarrel=False, orbital_free=True,
                gpernode=75)
        setups[symbol] = s

    for element, result, e, charge in zip(elements,
                                          results,
                                          electrons,
                                          charges):
        atom = Atoms(element,
                     positions=[(c, c, c)],
                     cell=(a, a, a))

        mixer = Mixer(0.3, 5, 1)
        calc = GPAW(mode='fd',
                    gpts=(32, 32, 32),
                    txt='-',
                    xc=xcname,
                    setups=setups,
                    eigensolver='cg', mixer=mixer, charge=charge)

        atom.calc = calc

        E = atom.get_total_energy()
        n = calc.get_all_electron_density()

        dv = atom.get_volume() / calc.get_number_of_grid_points().prod()
        I = n.sum() * dv / 2**3

        assert I == pytest.approx(e, abs=1.0e-6)
        assert result == pytest.approx(E, abs=1.0e-2)
