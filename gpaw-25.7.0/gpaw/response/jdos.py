import numpy as np

from gpaw.response import ResponseContext
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.pair_integrator import (PairFunctionIntegrator,
                                           DynamicPairFunction)
from gpaw.response.chiks import get_temporal_part


class JDOS(DynamicPairFunction):

    def __init__(self, spincomponent, q_c, zd):
        self.spincomponent = spincomponent
        super().__init__(q_c, zd)

    def zeros(self):
        nz = len(self.zd)
        return np.zeros(nz, float)


class JDOSCalculator(PairFunctionIntegrator):
    r"""Joint density of states calculator.

    Here, the joint density of states of collinear systems is defined as the
    spectral part of the four-component Kohn-Sham susceptibility,
    see [PRB 103, 245110 (2021)]:

                   __  __
                1  \   \   /
    g^μν(q,ω) = ‾  /   /   | σ^μ_ss' σ^ν_s's (f_nks - f_n'k+qs')
                V  ‾‾  ‾‾  \
                   k   t                                \
                             x δ(ħω - [ε_n'k's'-ε_nks]) |
                                                        /

    where t is a composite band and spin transition index: (n, s) -> (n', s').
    """

    def __init__(self, gs, context=None,
                 nbands=None, bandsummation='pairwise',
                 **kwargs):
        """Contruct the JDOSCalculator

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        nbands : int
            Number of bands to include in the sum over states
        bandsummation : str
            Band summation strategy (does not change the result, but can affect
            the run-time).
            'pairwise': sum over pairs of bands
            'double': double sum over band indices.
        kwargs : see gpaw.response.pair_integrator.PairFunctionIntegrator
        """
        if context is None:
            context = ResponseContext()
        assert isinstance(context, ResponseContext)

        super().__init__(gs, context, **kwargs)

        self.nbands = nbands
        self.bandsummation = bandsummation

    def calculate(self, spincomponent, q_c, zd):
        """Calculate g^μν(q,ω) using a lorentzian broadening of the δ-function

        Parameters
        ----------
        spincomponent : str
            Spin component (μν) of the joint density of states.
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
        q_c : list or np.array
            Wave vector in relative coordinates
        zd : ComplexFrequencyDescriptor
            Complex frequencies ħz to evaluate g^μν(q,ω) at, where z = ω + iη
            and η > 0 yields the HWHM of the resulting lorentzian broadening.
        """
        assert isinstance(zd, ComplexFrequencyDescriptor)
        assert zd.upper_half_plane

        # Prepare to sum over bands and spins
        transitions = self.get_band_and_spin_transitions(
            spincomponent, nbands=self.nbands,
            bandsummation=self.bandsummation)

        self.context.print(self.get_info_string(
            q_c, len(zd), spincomponent, self.nbands, len(transitions)))

        # Set up output data structure
        jdos = JDOS(spincomponent, q_c, zd)

        # Perform actual in-place integration
        self.context.print('Integrating the joint density of states:')
        self._integrate(jdos, transitions)

        return jdos

    def add_integrand(self, kptpair, weight, jdos):
        r"""Add the g^μν(q,ω) integrand of the outer k-point integral:
                        __
                  -1    \  σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
        (...)_k = ‾‾ Im /  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                  π     ‾‾   ħω - (ε_n'k's' - ε_nks) + iħη
                        t

        NB: Since the implemented spin matrices are real, the dissipative part
        is equal to the imaginary part (up to a factor of π) of the full
        integrand.
        """
        # Construct jdos integrand via the imaginary part of the frequency
        # dependence in χ_KS^μν(q,z)
        if jdos.spincomponent == '00' and self.gs.nspins == 1:
            weight = 2 * weight
        x_mytz = get_temporal_part(jdos.spincomponent, jdos.zd.hz_z,
                                   kptpair, self.bandsummation)
        integrand_mytz = -x_mytz.imag / np.pi

        with self.context.timer('Perform sum over t-transitions'):
            # Sum over local transitions
            integrand_z = np.sum(integrand_mytz, axis=0)
            # Sum over distributed transitions
            kptpair.tblocks.blockcomm.sum(integrand_z)
        jdos.array[:] += weight * integrand_z

    def get_info_string(self, q_c, nz, spincomponent, nbands, nt):
        """Get information about the joint density of states calculation"""
        info_list = ['',
                     'Calculating the joint density of states with:',
                     f'    q_c: [{q_c[0]}, {q_c[1]}, {q_c[2]}]',
                     f'    Number of frequency points: {nz}',
                     f'    Spin component: {spincomponent}',
                     self.get_band_and_transitions_info_string(nbands, nt),
                     '',
                     self.get_basic_info_string()]
        return '\n'.join(info_list)
