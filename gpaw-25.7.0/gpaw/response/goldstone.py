from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from gpaw.response.dyson import HXCScaling, DysonEquation, DysonEquations

if TYPE_CHECKING:
    from gpaw.response.chiks import SelfEnhancementCalculator


class GoldstoneScaling(HXCScaling):
    """Scale the Dyson equation to fulfill a Goldstone condition."""

    def _calculate_scaling(self, dyson_equations):
        """Calculate scaling coefficient λ."""
        self.check_descriptors(dyson_equations)

        # Find the frequency to determine the scaling from and identify where
        # the Dyson equation in question is distributed
        wgs = self.find_goldstone_frequency(
            dyson_equations.zd.omega_w)
        wblocks = dyson_equations.zblocks
        rgs, mywgs = wblocks.find_global_index(wgs)

        # Let the rank which holds the Goldstone frequency find and broadcast λ
        lambdbuf = np.empty(1, dtype=float)
        if wblocks.blockcomm.rank == rgs:
            lambdbuf[:] = self.find_goldstone_scaling(dyson_equations[mywgs])
        wblocks.blockcomm.broadcast(lambdbuf, rgs)
        lambd = lambdbuf[0]

        return lambd

    @staticmethod
    def check_descriptors(dyson_equations):
        if not (dyson_equations.qpd.optical_limit and
                dyson_equations.spincomponent in ['+-', '-+']):
            raise ValueError(
                'The Goldstone condition only applies to χ^(+-)(q=0).')

    @abstractmethod
    def find_goldstone_frequency(self, omega_w):
        """Determine frequency index for the Goldstone condition."""

    @abstractmethod
    def find_goldstone_scaling(self, dyson_equation: DysonEquation) -> float:
        """Calculate the Goldstone scaling parameter λ."""


class FMGoldstoneScaling(GoldstoneScaling):
    """Fulfil ferromagnetic Goldstone condition."""

    @staticmethod
    def find_goldstone_frequency(omega_w):
        """Ferromagnetic Goldstone condition is based on χ^(+-)(ω=0)."""
        wgs = np.abs(omega_w).argmin()
        assert abs(omega_w[wgs]) < 1.e-8, \
            "Frequency grid needs to include ω=0."
        return wgs

    @staticmethod
    def find_goldstone_scaling(dyson_equation):
        return find_fm_goldstone_scaling(dyson_equation)


class NewFMGoldstoneScaling(FMGoldstoneScaling):
    """Fulfil Goldstone condition by maximizing a^(+-)(ω=0)."""

    def __init__(self,
                 lambd: float | None = None,
                 m_G: np.ndarray | None = None):
        """Construct the scaling object.

        If the λ-parameter hasn't yet been calculated, the (normalized)
        spin-polarization |m> is needed in order to extract the acoustic
        magnon mode lineshape, a^(+-)(ω=0) = -Im[<m|χ^(+-)(ω=0)|m>]/π.
        """
        super().__init__(lambd=lambd)
        self.m_G = m_G

    @classmethod
    def from_xi_calculator(cls, xi_calc: SelfEnhancementCalculator):
        """Construct scaling object with |m> consistent with a xi_calc."""
        return cls(m_G=cls.calculate_m(xi_calc))

    @staticmethod
    def calculate_m(xi_calc: SelfEnhancementCalculator):
        """Calculate the normalized spin-polarization |m>."""
        from gpaw.response.localft import (LocalFTCalculator,
                                           add_spin_polarization)
        localft_calc = LocalFTCalculator.from_rshe_parameters(
            xi_calc.gs, xi_calc.context,
            rshelmax=xi_calc.rshelmax, rshewmin=xi_calc.rshewmin)
        qpd = xi_calc.get_pw_descriptor(q_c=[0., 0., 0.])
        nz_G = localft_calc(qpd, add_spin_polarization)
        return nz_G / np.linalg.norm(nz_G)

    def find_goldstone_scaling(self, dyson_equation):
        assert self.m_G is not None, \
            'Please supply spin-polarization to calculate λ'

        def acoustic_antispectrum(lambd):
            """Calculate -a^(+-)(ω=0)."""
            return - calculate_acoustic_spectrum(
                lambd, dyson_equation, self.m_G)

        # Maximize a^(+-)(ω=0)
        res = minimize(acoustic_antispectrum, x0=[1.], bounds=[(0.1, 10.)])
        assert res.success
        return res.x[0]


class RefinedFMGoldstoneScaling(HXCScaling):
    """Ensures that a^(+-)(ω) has a maximum in ω=0."""

    def __init__(self,
                 lambd: float | None = None,
                 base_scaling: NewFMGoldstoneScaling | None = None):
        """Construct the scaling object.

        If the λ-parameter hasn't yet been calculated, we use a base scaling
        class, which calculates λ approximately (and gives us access to |m>),
        thus providing a starting point for the refinement.
        """
        super().__init__(lambd=lambd)
        self._base_scaling = base_scaling

    @classmethod
    def from_xi_calculator(cls, xi_calc: SelfEnhancementCalculator):
        return cls(
            base_scaling=NewFMGoldstoneScaling.from_xi_calculator(xi_calc))

    @property
    def m_G(self):
        assert self._base_scaling is not None
        assert self._base_scaling.m_G is not None
        return self._base_scaling.m_G

    def _calculate_scaling(self, dyson_equations: DysonEquations) -> float:
        """Calculate the scaling coefficient λ."""
        # First we calculate the base scaling based on a^(+-)(ω=0)
        assert self._base_scaling is not None
        self._base_scaling.calculate_scaling(dyson_equations)
        base_lambd = self._base_scaling.lambd

        # Secondly, we extract the spectral peak position by performing a
        # parabolic fit to the five points with lowest |ω|.
        omega_W = dyson_equations.zd.omega_w
        wblocks = dyson_equations.zblocks
        fiveW_w = np.argpartition(np.abs(omega_W), 5)[:5]
        omega_w = omega_W[fiveW_w]

        def near_acoustic_spectrum(lambd):
            a_w = np.empty(5, dtype=float)
            for w, W in enumerate(fiveW_w):
                wrank, myw = wblocks.find_global_index(W)
                if wblocks.blockcomm.rank == wrank:
                    a_w[w] = calculate_acoustic_spectrum(
                        lambd, dyson_equations[myw], self.m_G)
                wblocks.blockcomm.broadcast(a_w[w:w + 1], wrank)
            return a_w

        def acoustic_magnon_frequency(lambd):
            a_w = near_acoustic_spectrum(lambd)
            a, b, c = np.polyfit(omega_w, a_w, 2)
            return -b / (2 * a)

        # Lastly, we minimize the (absolute) peak frequency |ω_0| to obtain the
        # refined λ. To do so efficiently, we define a (hyperbolic) cost
        # function which is linear in |ω_0| in the meV range, but parabolic in
        # the μeV range, such that derivatives remain smooth at the minimum.

        def cost_function(lambd):
            return np.sqrt(5e-5 + acoustic_magnon_frequency(lambd)**2)

        res = minimize(cost_function, x0=[base_lambd],
                       bounds=[(base_lambd * 0.975, base_lambd * 1.025)])
        assert res.success, res
        return res.x[0]


class AFMGoldstoneScaling(GoldstoneScaling):
    """Fulfil antiferromagnetic Goldstone condition."""

    @staticmethod
    def find_goldstone_frequency(omega_w):
        """Antiferromagnetic Goldstone condition is based on ω->0^+."""
        # Set ω<=0. to np.inf
        omega1_w = np.where(omega_w < 1.e-8, np.inf, omega_w)
        # Sort for the two smallest positive frequencies
        omega2_w = np.partition(omega1_w, 1)
        # Find original index of second smallest positive frequency
        wgs = np.abs(omega_w - omega2_w[1]).argmin()
        return wgs

    @staticmethod
    def find_goldstone_scaling(dyson_equation):
        return find_afm_goldstone_scaling(dyson_equation)


def find_fm_goldstone_scaling(dyson_equation):
    """Find goldstone scaling of the kernel by ensuring that the
    macroscopic inverse enhancement function has a root in (q=0, omega=0)."""
    fnct = partial(calculate_macroscopic_kappa,
                   dyson_equation=dyson_equation)

    def is_converged(kappaM):
        return abs(kappaM) < 1e-7

    return find_root(fnct, is_converged)


def find_afm_goldstone_scaling(dyson_equation):
    """Find goldstone scaling of the kernel by ensuring that the
    macroscopic magnon spectrum vanishes at q=0. for finite frequencies."""
    fnct = partial(calculate_macroscopic_spectrum,
                   dyson_equation=dyson_equation)

    def is_converged(SM):
        # We want the macroscopic spectrum to be strictly positive
        return 0. < SM < 1.e-7

    return find_root(fnct, is_converged)


def find_root(fnct, is_converged):
    """Find the root f(λ)=0, where the scaling parameter λ~1.

    |f(λ)| is minimized iteratively, assuming that f(λ) is continuous and
    monotonically decreasing with λ for λ∊]0.1, 10[.
    """
    lambd = 1.  # initial guess for the scaling parameter
    value = fnct(lambd)
    lambd_incr = 0.1 * np.sign(value)  # Increase λ to decrease f(λ)
    while not is_converged(value) or abs(lambd_incr) > 1.e-7:
        # Update λ
        lambd += lambd_incr
        if lambd <= 0.1 or lambd >= 10.:
            raise Exception(f'Found an invalid λ-value of {lambd:.4f}')
        # Update value and refine increment, if we have passed f(λ)=0
        value = fnct(lambd)
        if np.sign(value) != np.sign(lambd_incr):
            lambd_incr *= -0.2
    return lambd


def calculate_macroscopic_kappa(lambd, dyson_equation):
    """Invert dyson equation and calculate the inverse enhancement function."""
    chi_GG = dyson_equation.invert(lambd=lambd)
    return (dyson_equation.chiks_GG[0, 0] / chi_GG[0, 0]).real


def calculate_macroscopic_spectrum(lambd, dyson_equation):
    """Invert dyson equation and extract the macroscopic spectrum."""
    chi_GG = dyson_equation.invert(lambd=lambd)
    return - chi_GG[0, 0].imag / np.pi


def calculate_acoustic_spectrum(lambd, dyson_equation, m_G):
    """Invert the dyson equation and extract the acoustic spectrum."""
    chi_GG = dyson_equation.invert(lambd=lambd)
    chi_projection = np.conj(m_G) @ chi_GG @ m_G
    return - chi_projection.imag / np.pi
