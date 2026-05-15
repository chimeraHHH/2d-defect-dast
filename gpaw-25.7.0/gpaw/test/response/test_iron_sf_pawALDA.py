import pytest

import numpy as np

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator, SelfEnhancementCalculator
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.dyson import DysonSolver
from gpaw.response.susceptibility import (spectral_decomposition,
                                          read_eigenmode_lineshapes)
from gpaw.response.pair_functions import read_pair_function

from gpaw.test import findpeak
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.kspair
@pytest.mark.response
def test_response_iron_sf_pawALDA(in_tmp_dir, gpw_files, scalapack):
    # ---------- Inputs ---------- #

    # Magnetic response calculation
    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    frq_w = np.linspace(-1.0, 3.0, 161)
    ecut = 100
    eta = 0.5
    rshewmin = 1e-8

    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    context = ResponseContext(txt='iron_susceptibility.txt')
    calc = GPAW(gpw_files['fe_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['fe_pw']
    gs = ResponseGroundStateAdapter(calc)

    calc_args = (gs,)
    calc_kwargs = dict(context=context,
                       nbands=nbands,
                       ecut=ecut,
                       gammacentered=True,
                       bandsummation='double',
                       nblocks=nblocks)
    chiks_calc = ChiKSCalculator(*calc_args, **calc_kwargs)
    xi_calc = SelfEnhancementCalculator(*calc_args,
                                        rshewmin=rshewmin,
                                        **calc_kwargs)
    dyson_solver = DysonSolver(context)

    for q, q_c in enumerate(q_qc):
        # Calculate χ_KS^+- and Ξ^++
        zd = ComplexFrequencyDescriptor.from_array(frq_w + 1j * eta)
        chiks = chiks_calc.calculate('+-', q_c, zd)
        xi = xi_calc.calculate('+-', q_c, zd)

        # Distribute frequencies and invert dyson equation
        chiks = chiks.copy_with_global_frequency_distribution()
        xi = xi.copy_with_global_frequency_distribution()
        chi = dyson_solver(chiks, xi)

        # plot_inverse_enhancement(xi)

        # Write macroscopic component and acoustic magnon mode
        chi.write_macroscopic_component(f'iron_chiM_q{q}.csv')
        Amaj, _ = spectral_decomposition(chi)
        Amaj.write_eigenmode_lineshapes(f'iron_Amaj_q{q}.csv')

    context.write_timer()

    # plot_magnons()

    refs_q = [
        # (wpeak, Ipeak, Apeak)
        (0.001, 0.476, 3.104),
        (0.165, 0.428, 2.816)]

    world.barrier()  # wait for csv-file written above ...
    for q, refs in enumerate(refs_q):
        w_w, chiM_w, a_w = extract_data(q)
        wpeak1, Ipeak = findpeak(w_w, -chiM_w.imag / np.pi)
        wpeak2, Apeak = findpeak(w_w, a_w)

        assert wpeak1 == pytest.approx(wpeak2, abs=0.002)  # eV
        assert wpeak1 == pytest.approx(refs[0], abs=0.01)  # eV
        assert Ipeak == pytest.approx(refs[1], abs=0.01)  # a.u.
        assert Apeak == pytest.approx(refs[2], abs=0.05)  # a.u.


def extract_data(q):
    w1_w, chiM_w = read_pair_function(f'iron_chiM_q{q}.csv')
    w2_w, a_wm = read_eigenmode_lineshapes(f'iron_Amaj_q{q}.csv')
    assert np.allclose(w1_w, w2_w)
    return w1_w, chiM_w, a_wm[:, 0]


def plot_inverse_enhancement(xi):
    import matplotlib.pyplot as plt
    from ase.units import Ha
    invenh_mywM, _ = np.linalg.eig(xi.array)
    invenh_wM = xi.blocks1d.all_gather(invenh_mywM)
    omega_w = xi.zd.omega_w * Ha

    for M in range(invenh_wM.shape[1]):
        plt.subplot(1, 2, 1)
        plt.scatter(omega_w, invenh_wM[:, M].real)
        plt.subplot(1, 2, 2)
        plt.scatter(omega_w, invenh_wM[:, M].imag)
    plt.subplot(1, 2, 1)
    plt.axhline(1., c='0.5')
    if world.rank == 0:
        plt.show()


def plot_magnons():
    import matplotlib.pyplot as plt
    for q in range(2):
        w_w, chiM_w, a_w = extract_data(q)
        plt.subplot(1, 2, 1)
        plt.plot(w_w, - chiM_w.imag / np.pi, label=f'{q}')
        plt.subplot(1, 2, 2)
        plt.plot(w_w, a_w, label=f'{q}')
        plt.legend(title='q')
    if world.rank == 0:
        plt.show()
