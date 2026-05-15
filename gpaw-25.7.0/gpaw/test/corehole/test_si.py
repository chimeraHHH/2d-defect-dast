import pytest
import numpy as np

from gpaw import GPAW
from gpaw.test import gen
from gpaw.xas import XAS, RecursionMethod
import gpaw.mpi as mpi


@pytest.mark.old_gpaw_only
def test_corehole_si(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):
    # restart from file
    calc = GPAW(gpw_files['si_corehole_pw'])
    si = calc.atoms

    if mpi.size == 1:
        xas = XAS(calc)
        x, y = xas.get_spectra()
    else:
        x = np.linspace(0, 10, 50)

    k = 2
    calc = calc.new(kpts=(k, k, k))
    calc.initialize(si)
    calc.set_positions(si)
    assert calc.wfs.dtype == complex

    r = RecursionMethod(calc)
    r.run(40)
    if mpi.size == 1:
        z = r.get_spectra(x)

    if 0:
        import matplotlib.pyplot as plt
        plt.plot(x, y[0])
        plt.plot(x, sum(y))
        plt.plot(x, z[0])
        plt.show()

    # 2p corehole
    s = gen('Si', name='hch2p', corehole=(2, 1, 0.5), gpernode=30)
    calc = GPAW(gpw_files['si_corehole_pw'],
                setups={0: s})
    si.calc = calc

    def stopcalc():
        calc.scf.converged = True

    calc.attach(stopcalc, 1)
    _ = si.get_potential_energy()


def test_si_nonortho(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):
    # Generate setup for oxygen with half a core-hole:
    # restart from file
    # code moved to fixtures: si_corehole_sym,
    # si_corehole_nosym_pw, si_corehole_sym_pw
    calc1 = GPAW(gpw_files['si_corehole_sym_pw'])
    calc2 = GPAW(gpw_files['si_corehole_nosym_pw'])
    if mpi.size == 1:
        xas1 = XAS(calc1)
        x, y1 = xas1.get_spectra()
        xas2 = XAS(calc2)
        x2, y2 = xas2.get_spectra(E_in=x)

        assert (np.sum(abs(y1 - y2)[0, :500]**2) < 5e-9)
        assert (np.sum(abs(y1 - y2)[1, :500]**2) < 5e-9)
        assert (np.sum(abs(y1 - y2)[2, :500]**2) < 5e-9)
