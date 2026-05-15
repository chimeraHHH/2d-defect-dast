import pytest
import numpy as np
from ase.units import Bohr
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.df import DielectricFunction, read_response_function
from gpaw.test import findpeak


@pytest.mark.dielectricfunction
@pytest.mark.response
@pytest.mark.parametrize('eshift', [None, 4])
@pytest.mark.libxc
def test_response_diamond_absorption(in_tmp_dir, eshift):
    a = 6.75 * Bohr
    atoms = bulk('C', 'diamond', a=a)

    calc = GPAW(mode='pw',
                kpts=(3, 3, 3),
                eigensolver='rmm-diis',
                occupations=FermiDirac(0.001))

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('C.gpw', 'all')

    if eshift is None:
        eM1_ = 9.727
        eM2_ = 9.548
        w0_ = 10.7782
        I0_ = 5.47
        w_ = 10.7532
        I_ = 5.98
    else:
        eM1_ = 6.993
        eM2_ = 6.904
        w0_ = 14.784
        I0_ = 5.47
        w_ = 14.757
        I_ = 5.998

    # Test the old interface to the dielectric constant
    df = DielectricFunction('C.gpw', frequencies=(0.,), eta=0.001, ecut=50,
                            hilbert=False, eshift=eshift)
    eM1, eM2 = df.get_macroscopic_dielectric_constant()
    assert eM1 == pytest.approx(eM1_, abs=0.01)
    assert eM2 == pytest.approx(eM2_, abs=0.01)

    # ----- RPA dielectric function ----- #
    dfcalc = DielectricFunction(
        'C.gpw', eta=0.25, ecut=50,
        frequencies=np.linspace(0, 24., 241), hilbert=False, eshift=eshift)
    eps = dfcalc.get_literal_dielectric_function()

    # Test the new interface to the dielectric constant
    eM1, eM2 = eps.dielectric_constant()
    assert eM1 == pytest.approx(eM1_, abs=0.01)
    assert eM2 == pytest.approx(eM2_, abs=0.01)

    # Test the macroscopic dielectric function
    omega_w, eps0M_w, epsM_w = eps.macroscopic_dielectric_function().arrays
    w0, I0 = findpeak(omega_w, eps0M_w.imag)
    assert w0 == pytest.approx(w0_, abs=0.01)
    assert I0 / (4 * np.pi) == pytest.approx(I0_, abs=0.1)
    w, I = findpeak(omega_w, epsM_w.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I / (4 * np.pi) == pytest.approx(I_, abs=0.1)

    # Test polarizability
    omega_w, a0rpa_w, arpa_w = eps.polarizability().arrays
    w0, I0 = findpeak(omega_w, a0rpa_w.imag)
    assert w0 == pytest.approx(w0_, abs=0.01)
    assert I0 == pytest.approx(I0_, abs=0.01)
    w, I = findpeak(omega_w, arpa_w.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I == pytest.approx(I_, abs=0.01)

    # Test that the macroscopic dielectric function can be calculated also from
    # the inverse dielectric function and the bare dielectric function
    epsinv = dfcalc.get_inverse_dielectric_function()
    _, _, epsM_frominv_w = epsinv.macroscopic_dielectric_function().arrays
    assert epsM_frominv_w == pytest.approx(epsM_w, rel=1e-6)
    epsbare = dfcalc.get_bare_dielectric_function()
    _, _, epsM_frombare_w = epsbare.macroscopic_dielectric_function().arrays
    assert epsM_frombare_w == pytest.approx(epsM_w, rel=1e-6)

    # ----- TDDFT absorption spectra ----- #

    # Absorption spectrum calculation ALDA
    if eshift is None:
        w_ = 10.7562
        I_ = 5.8803
    else:
        w_ = 14.7615
        I_ = 5.7946

    epsinv = dfcalc.get_inverse_dielectric_function(xc='ALDA', rshelmax=0)
    # Here we base the check on a written results file
    epsinv.polarizability().write(filename='ALDA_pol.csv')
    dfcalc.context.comm.barrier()
    omega_w, a0alda_w, aalda_w = read_response_function('ALDA_pol.csv')

    assert a0alda_w == pytest.approx(a0rpa_w, rel=1e-4)
    w, I = findpeak(omega_w, aalda_w.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I == pytest.approx(I_, abs=0.1)

    # Absorption spectrum calculation long-range kernel
    if eshift is None:
        w_ = 10.2906
        I_ = 5.6955
    else:
        w_ = 14.2901
        I_ = 5.5508

    epsinv = dfcalc.get_inverse_dielectric_function(xc='LR0.25')
    omega_w, a0lr_w, alr_w = epsinv.polarizability().arrays

    assert a0lr_w == pytest.approx(a0rpa_w, rel=1e-4)
    w, I = findpeak(omega_w, alr_w.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I == pytest.approx(I_, abs=0.1)

    # Absorption spectrum calculation Bootstrap
    if eshift is None:
        w_ = 10.4600
        I_ = 6.0263
    else:
        w_ = 14.2626
        I_ = 5.3896

    epsinv = dfcalc.get_inverse_dielectric_function(xc='Bootstrap')
    omega_w, a0btsr_w, abtsr_w = epsinv.polarizability().arrays

    assert a0btsr_w == pytest.approx(a0rpa_w, rel=1e-4)
    w, I = findpeak(omega_w, abtsr_w.imag)
    assert w == pytest.approx(w_, abs=0.02)
    assert I == pytest.approx(I_, abs=0.2)

    # import matplotlib.pyplot as plt
    # plt.plot(omega_w, a0rpa_w.imag, label='IP')
    # plt.plot(omega_w, arpa_w.imag, label='RPA')
    # plt.plot(omega_w, aalda_w.imag, label='ALDA')
    # plt.plot(omega_w, alr_w.imag, label='LR0.25')
    # plt.plot(omega_w, abtsr_w.imag, label='Bootstrap')
    # plt.legend()
    # plt.show()
