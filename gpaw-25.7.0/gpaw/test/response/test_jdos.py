# General modules
import pytest

from itertools import product

import numpy as np

# import matplotlib.pyplot as plt

# Script modules
from ase.units import Hartree

from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.jdos import JDOSCalculator
from gpaw.response.kpoints import KPointFinder
from gpaw.test.response.test_chiks import (generate_system_s,
                                           generate_qrel_q, get_q_c,
                                           generate_nblocks_n)
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.response
@pytest.mark.kspair
@pytest.mark.parametrize('system,qrel',
                         product(generate_system_s(), generate_qrel_q()))
def test_jdos(in_tmp_dir, gpw_files, system, qrel):
    # ---------- Inputs ---------- #

    # What material, spin-component and q-vector to calculate the jdos for
    wfs, spincomponent = system
    q_c = get_q_c(wfs, qrel)

    # Where to evaluate the jdos
    omega_w = np.linspace(-10.0, 10.0, 321)
    eta = 0.2
    zd = ComplexFrequencyDescriptor.from_array(omega_w + 1.j * eta)

    # Calculation parameters (which should not affect the result)
    qsymmetry_s = [True, False]
    bandsummation_b = ['double', 'pairwise']
    nblocks_n = generate_nblocks_n()

    # ---------- Script ---------- #

    # Set up the ground state adapter based on the fixture
    calc = GPAW(gpw_files[wfs], parallel=dict(domain=1))
    nbands = response_band_cutoff[wfs]
    gs = ResponseGroundStateAdapter(calc)

    # Calculate the jdos manually
    serial_calc = GPAW(gpw_files[wfs], communicator=mpi.serial_comm)
    jdos_refcalc = MyManualJDOS(serial_calc)
    jdosref_w = jdos_refcalc.calculate(spincomponent, q_c,
                                       omega_w,
                                       eta=eta,
                                       nbands=nbands)

    # Calculate the jdos using the PairFunctionIntegrator module
    for qsymmetry in qsymmetry_s:
        for bandsummation in bandsummation_b:
            for nblocks in nblocks_n:
                jdos_calc = JDOSCalculator(gs,
                                           nbands=nbands,
                                           qsymmetry=qsymmetry,
                                           bandsummation=bandsummation,
                                           nblocks=nblocks)
                jdos = jdos_calc.calculate(spincomponent, q_c, zd)
                jdos_w = jdos.array
                assert jdos_w == pytest.approx(jdosref_w)

        # plt.subplot()
        # plt.plot(wd.omega_w * Hartree, jdos_w)
        # plt.plot(wd.omega_w * Hartree, jdosref_w)
        # plt.title(f'{q_c} {spincomponent}')
        # plt.show()


class MyManualJDOS:
    def __init__(self, calc):
        self.calc = calc
        self.nspins = calc.wfs.nspins

        kd = calc.wfs.kd
        gd = calc.wfs.gd
        self.kd = kd
        self.kptfinder = KPointFinder(kd.bzk_kc)
        self.kweight = 1 / (gd.volume * len(kd.bzk_kc))

    def calculate(self, spincomponent, q_c, omega_w,
                  eta=0.2,
                  nbands=None):
        r"""Calculate the joint density of states:
                       __  __
                    1  \   \
        g_j(q, ω) = ‾  /   /  (f_nks - f_mk+qs') δ(ω-[ε_mk+qs' - ε_nks])
                    V  ‾‾  ‾‾
                       k   n,m

        for a given spin component specifying the spin transitions s -> s'.
        """
        q_c = np.asarray(q_c)
        # Internal frequencies in Hartree
        omega_w = omega_w / Hartree
        eta = eta / Hartree
        # Allocate array
        jdos_w = np.zeros_like(omega_w)

        for K1, k1_c in enumerate(self.kd.bzk_kc):
            # de = e2 - e1, df = f2 - f1
            de_t, df_t = self.get_transitions(K1, k1_c, q_c,
                                              spincomponent, nbands)

            if self.nspins == 1:
                df_t *= 2

            # Set up jdos
            delta_wt = self.delta(omega_w, eta, de_t)
            jdos_wt = - df_t[np.newaxis] * delta_wt

            # Sum over transitions
            jdos_w += np.sum(jdos_wt, axis=1)

        return self.kweight * jdos_w

    @staticmethod
    def delta(omega_w, eta, de_t):
        r"""Create lorentzian delta-functions

                ~ 1       η
        δ(ω-Δε) = ‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                  π (ω-Δε)^2 + η^2
        """
        x_wt = omega_w[:, np.newaxis] - de_t[np.newaxis]
        return eta / np.pi / (x_wt**2. + eta**2.)

    def get_transitions(self, K1, k1_c, q_c, spincomponent, nbands):
        assert isinstance(nbands, int)
        if spincomponent == '00':
            if self.nspins == 1:
                s1_s = [0]
                s2_s = [0]
            else:
                s1_s = [0, 1]
                s2_s = [0, 1]
        elif spincomponent == '+-':
            s1_s = [0]
            s2_s = [1]
        else:
            raise ValueError(spincomponent)

        # Find k_c + q_c
        K2 = self.kptfinder.find(k1_c + q_c)

        de_t = []
        df_t = []
        kd = self.kd
        calc = self.calc
        for s1, s2 in zip(s1_s, s2_s):
            # Get composite u=(s,k) indices and KPoint objects
            u1 = kd.bz2ibz_k[K1] * self.nspins + s1
            u2 = kd.bz2ibz_k[K2] * self.nspins + s2
            kpt1, kpt2 = calc.wfs.kpt_u[u1], calc.wfs.kpt_u[u2]

            # Extract eigenenergies and occupation numbers
            eps1_n = kpt1.eps_n[:nbands]
            eps2_n = kpt2.eps_n[:nbands]
            f1_n = kpt1.f_n[:nbands] / kpt1.weight
            f2_n = kpt2.f_n[:nbands] / kpt2.weight

            # Append data
            de_nm = eps2_n[:, np.newaxis] - eps1_n[np.newaxis]
            df_nm = f2_n[:, np.newaxis] - f1_n[np.newaxis]
            de_t += list(de_nm.flatten())
            df_t += list(df_nm.flatten())
        de_t = np.array(de_t)
        df_t = np.array(df_t)

        return de_t, df_t
