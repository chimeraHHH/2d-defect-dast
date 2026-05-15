import time
import pytest
import numpy as np

from ase.build import bulk
from ase.parallel import parprint
from ase.utils.timing import Timer

from gpaw import GPAW, PW, FermiDirac
from gpaw.test import findpeak
from gpaw.mpi import size, world

from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.df import DielectricFunction, read_response_function
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.susceptibility import ChiFactory
from gpaw.response.pair_functions import read_pair_function


@pytest.mark.dielectricfunction
@pytest.mark.kspair
@pytest.mark.response
def test_response_silicon_chi_RPA(in_tmp_dir):
    assert size <= 4**3

    # Ground state calculation

    t1 = time.time()

    a = 5.431
    atoms = bulk('Si', 'diamond', a=a)
    atoms.center()
    calc = GPAW(mode=PW(200),
                nbands=8,
                kpts=(4, 4, 4),
                parallel={'domain': 1},
                occupations=FermiDirac(width=0.05),
                xc='LDA')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si', 'all')
    t2 = time.time()

    # Excited state calculation
    q = np.array([1 / 4.0, 0, 0])
    w = np.linspace(0, 24, 241)
    eta = 0.2

    # Using DF
    df = DielectricFunction(calc='Si',
                            frequencies=w, eta=eta, ecut=50,
                            hilbert=False)
    df.get_dynamic_susceptibility(xc='RPA', q_c=q, filename='Si_chi1.csv')

    t3 = time.time()

    world.barrier()

    # Using the ChiFactory
    gs = ResponseGroundStateAdapter(calc)
    chiks_calc = ChiKSCalculator(gs, ecut=50)
    chi_factory = ChiFactory(chiks_calc)
    chiks, chi = chi_factory('00', q, w + 1.j * eta)
    chi.write_macroscopic_component('Si_chi2.csv')
    chi_factory.context.write_timer()
    chi_factory.context.set_timer(Timer())

    t4 = time.time()

    # Calculate also the ALDA susceptibility manually
    hxc_kernel = chi_factory.get_hxc_kernel('ALDA', '00', chiks.qpd)
    chi = chi_factory.dyson_solver(chiks, hxc_kernel)
    chi.write_macroscopic_component('Si_chi3.csv')
    chi_factory.context.write_timer()

    t5 = time.time()

    world.barrier()

    parprint('')
    parprint('For ground  state calc, it took', (t2 - t1) / 60, 'minutes')
    parprint('For excited state calc 1, it took', (t3 - t2) / 60, 'minutes')
    parprint('For excited state calc 2, it took', (t4 - t3) / 60, 'minutes')
    parprint('For excited state calc 3, it took', (t5 - t4) / 60, 'minutes')

    w1_w, _, chi1_w = read_response_function('Si_chi1.csv')
    wpeak1, Ipeak1 = findpeak(w1_w, -chi1_w.imag)
    w2_w, chi2_w = read_pair_function('Si_chi2.csv')
    wpeak2, Ipeak2 = findpeak(w2_w, -chi2_w.imag)
    w3_w, chi3_w = read_pair_function('Si_chi3.csv')
    wpeak3, Ipeak3 = findpeak(w3_w, -chi3_w.imag)

    # The two response codes should hold identical results
    assert wpeak1 == pytest.approx(wpeak2, abs=0.02)
    assert Ipeak1 == pytest.approx(Ipeak2, abs=1.0)

    # Compare to test values
    assert wpeak2 == pytest.approx(16.69145, abs=0.02)  # RPA
    assert wpeak3 == pytest.approx(16.30622, abs=0.02)  # ALDA
