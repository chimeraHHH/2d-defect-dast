import re

import pytest
import numpy as np
from ase.build import molecule
from ase.units import Hartree

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.gauss import Gauss
from gpaw.lrtddft import LrTDDFT, photoabsorption_spectrum
from gpaw.lrtddft.kssingle import KSSingles


@pytest.mark.lrtddft
@pytest.mark.slow
@pytest.mark.skipif(world.size > 1, reason='test is serial')
def test_lrtddft_3(in_tmp_dir):
    from io import StringIO

    txt = None
    N2 = molecule('N2', vacuum=2.0)

    calc = GPAW(mode='fd',
                h=0.25,
                nbands=-2,
                spinpol=True,
                xc='LDA',
                txt=txt)

    N2.calc = calc
    _ = N2.get_potential_energy()
    calc.write('N2_wfs.gpw', 'all')

    # selections
    for obj in [KSSingles, LrTDDFT]:
        # selection using state numbers
        el = obj(restrict={'istart': 3, 'jend': 6}, txt=txt)
        el.calculate(N2)
        if hasattr(obj, 'diagonalize'):
            el.diagonalize()

        assert len(el) == 8
        # selection using an energy range
        el = obj(restrict={'energy_range': 8}, txt=txt)
        el.calculate(N2)
        if hasattr(obj, 'diagonalize'):
            el.diagonalize()

        assert len(el) == 4
        el = obj(restrict={'energy_range': 11.5}, txt=txt)
        el.calculate(N2)

        if hasattr(obj, 'diagonalize'):
            el.diagonalize()

        # Used to be == 18, but we lowered vacuum so test runs fast
        # and now it's:
        assert len(el) == 16

        if hasattr(obj, 'diagonalize'):
            el.diagonalize(restrict={'energy_range': 8})
            assert len(el) == 4

    lr = LrTDDFT(calc, nspins=2)
    lr.write('lrtddft3.dat.gz')
    lr.diagonalize()

    # This is done to test if writing and reading again yields the same result
    lr2 = LrTDDFT.read('lrtddft3.dat.gz')
    lr2.diagonalize()

    # Post processing

    Epeak = 19.5  # The peak we want to investigate (this is alone)
    Elist = np.asarray([lrsingle.get_energy() * Hartree
                        for lrsingle in lr])
    n = np.argmin(np.abs(Elist - Epeak))  # index of the peak

    E = lr[n].get_energy() * Hartree
    osz = lr[n].get_oscillator_strength()
    print('Original object        :', E, osz[0])

    # Test the output of analyse
    sio = StringIO()
    lr.analyse(n, out=sio)
    s = sio.getvalue()

    match = re.findall(
        r'%i: E=([0-9]*\.[0-9]*) eV, f=([0-9]*\.[0-9]*)*' % n, s)
    Eanalyse = float(match[0][0])
    oszanalyse = float(match[0][1])
    print('From analyse           :', Eanalyse, oszanalyse)
    assert E == pytest.approx(Eanalyse,
                              abs=1e-3)  # Written precision in analyse
    assert osz[0] == pytest.approx(oszanalyse, abs=1e-3)

    E2 = lr2[n].get_energy() * Hartree
    osz2 = lr2[n].get_oscillator_strength()
    print('Written and read object:', E2, osz2[0])

    # Compare values of original and written/read objects
    assert E == pytest.approx(E2, abs=1e-4)
    for i in range(len(osz)):
        assert osz[i] == pytest.approx(osz2[i], abs=1.7e-4)

    width = 0.05
    photoabsorption_spectrum(lr,
                             spectrum_file='lrtddft3-spectrum.dat',
                             width=width)
    # We need to be able to check the heights in the spectrum
    weight = Gauss(width).get(0)

    spectrum = np.loadtxt('lrtddft3-spectrum.dat', usecols=(0, 1))
    idx = (spectrum[:, 0] >= E - 0.1) & (spectrum[:, 0] <= E + 0.1)
    peak = np.argmax(spectrum[idx, 1]) + np.nonzero(idx)[0][0]
    Espec = spectrum[peak, 0]
    oszspec = spectrum[peak, 1] / weight

    print('Values from spectrum   :', Espec, oszspec)
    # Compare calculated values with values written to file
    assert E == pytest.approx(Espec,
                              abs=1e-2)  # The spectrum has a low sampling
    assert osz[0] == pytest.approx(oszspec, abs=1e-2)
