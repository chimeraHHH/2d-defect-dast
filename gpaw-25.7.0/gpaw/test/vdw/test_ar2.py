from math import sqrt
import pytest
from ase import Atoms
from gpaw import GPAW, Mixer, Davidson
from gpaw.xc.vdw import VDWFunctional


@pytest.mark.slow
@pytest.mark.libxc
def test_vdw_ar2(in_tmp_dir):
    energy_tolerance = 0.002

    def test():
        vdw = VDWFunctional('vdW-DF', verbose=1)
        d = 3.9
        x = d / sqrt(3)
        L = 3.0 + 2 * 4.0
        dimer = Atoms('Ar2', [(0, 0, 0), (x, x, x)], cell=(L, L, L))
        dimer.center()
        calc = GPAW(mode='fd', h=0.2, xc=dict(name='revPBE', stencil=1),
                    mixer=Mixer(0.8, 7, 50.0),
                    eigensolver=Davidson(5))
        dimer.calc = calc
        e2 = dimer.get_potential_energy()
        calc.write('Ar2.gpw')
        e2vdw = calc.get_xc_difference(vdw)
        e2vdwb = GPAW('Ar2.gpw').get_xc_difference(vdw)
        print(e2vdwb - e2vdw)
        assert abs(e2vdwb - e2vdw) < 1e-9
        if 0:
            # See ASE issue !1466
            del dimer[1]
        else:
            dimer.calc = None
            del dimer[1]
            dimer.calc = calc
        e = dimer.get_potential_energy()
        evdw = calc.get_xc_difference(vdw)

        E = 2 * e - e2
        Evdw = E + 2 * evdw - e2vdw
        print(E, Evdw)
        assert abs(E - -0.0048) < 6e-4, abs(E)
        assert abs(Evdw - +0.0223) < 3e-3, abs(Evdw)

        print(e2, e)
        assert e2 == pytest.approx(-0.005, abs=energy_tolerance)
        assert e == pytest.approx(-0.005, abs=energy_tolerance)

    test()
