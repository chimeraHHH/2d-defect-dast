from functools import cached_property
import numpy as np
import pytest

from gpaw.test import findpeak
from gpaw.response.df import DielectricFunction
from ase.units import Bohr, Hartree


class Helper:
    def __init__(self, gpw, integrationmode):
        self._gpw = gpw
        self._integrationmode = integrationmode

    @cached_property
    def df(self):
        return DielectricFunction(
            self._gpw,
            frequencies={'type': 'nonlinear',
                         'domega0': 0.03},
            ecut=10,
            rate=0.1,
            integrationmode=self._integrationmode,
            txt=None)

    @cached_property
    def lfc(self):
        # lfc == local field corrections (?)
        return {axis: self.df.get_dielectric_function(direction=axis)[1]
                for axis in 'xyz'}

    @cached_property
    def wp(self):
        chi0_drude = self.df.chi0calc.chi0_opt_ext_calc.drude_calc.calculate(
            self.df.chi0calc.chi0_opt_ext_calc.wd, 0.1)
        return chi0_drude.plasmafreq_vv[0, 0]**0.5

    @cached_property
    def w_w(self):
        return self.df.chi0calc.chi0_opt_ext_calc.wd.omega_w

    def _compare_peak(self, calc, axis):
        df1LFCx = self.lfc[axis]
        df2LFCx = calc.lfc[axis]
        # w_x equal for paired & polarized tetra
        w1, I1 = findpeak(self.w_w, -(1. / df1LFCx).imag)
        w2, I2 = findpeak(self.w_w, -(1. / df2LFCx).imag)
        assert w1 == pytest.approx(w2, abs=1e-3)
        assert I1 == pytest.approx(I2, abs=0.1)

    def compare_peaks(self, calc):
        for axis in 'xyz':
            self._compare_peak(calc, axis)


@pytest.mark.dielectricfunction
@pytest.mark.tetrahedron
@pytest.mark.response
def test_chi0_intraband(in_tmp_dir, gpw_files):
    """Comparing the plasmon peaks found in bulk sodium for two different
    atomic structures. Testing for idential plasmon peaks. Not using
    physical sodium cell."""
    intraband_spinpaired = gpw_files['intraband_spinpaired_fulldiag']
    intraband_spinpolarized = gpw_files['intraband_spinpolarized_fulldiag']

    calc1 = Helper(intraband_spinpaired, 'tetrahedron integration')
    calc2 = Helper(intraband_spinpaired, 'point integration')
    calc3 = Helper(intraband_spinpolarized, 'tetrahedron integration')
    calc4 = Helper(intraband_spinpolarized, 'point integration')

    # Compare plasmon frequencies and intensities
    w_w = calc1.w_w

    # frequency grids must be the same
    for calc in [calc1, calc2, calc3, calc4]:
        assert np.allclose(calc.w_w, w_w, atol=1e-5, rtol=1e-4)

    # Analytical Drude result
    n = 1 / (calc1.df.gs.volume * Bohr**-3)
    drude_wp = np.sqrt(4 * np.pi * n)

    # From https://doi.org/10.1021/jp810808h
    ref_wp = 5.71 / Hartree

    # spin paired matches spin polar - tetra
    assert calc1.wp == pytest.approx(calc3.wp, abs=1e-2)
    # spin paired matches spin polar - none
    assert calc2.wp == pytest.approx(calc4.wp, abs=1e-2)
    # Use larger margin when comparing to Drude
    assert calc1.wp == pytest.approx(drude_wp, abs=0.5)
    # Use larger margin when comparing to Drude
    assert calc2.wp == pytest.approx(drude_wp, abs=0.5)
    # paired tetra match paper
    assert calc1.wp == pytest.approx(ref_wp, abs=0.1)
    # paired none match paper
    assert calc2.wp == pytest.approx(ref_wp, abs=0.1)

    # w_x, w_y and w_z equal for paired & polarized tetra
    calc1.compare_peaks(calc3)

    # w_x, w_y and w_z equal for paired & polarized none
    calc2.compare_peaks(calc4)
