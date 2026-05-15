from __future__ import annotations
from time import ctime

from typing import TYPE_CHECKING
import numpy as np
from ase.units import Ha

from gpaw.response.symmetrize import HeadSymmetryOperators
from gpaw.response.integrators import Integrand, HilbertTetrahedron, Intraband
from gpaw.response.chi0_base import Chi0ComponentCalculator
from gpaw.response.chi0_data import Chi0DrudeData
from gpaw.response.frequencies import FrequencyGridDescriptor

if TYPE_CHECKING:
    from gpaw.response.kpoints import KPointDomainGenerator


class Chi0DrudeCalculator(Chi0ComponentCalculator):
    """Class for calculating the plasma frequency contribution to Chi0,
    that is, the contribution from intraband transitions inside of metallic
    bands. This corresponds directly to the dielectric function in the Drude
    model."""

    def __init__(self, *args, **kwargs):
        # Serial block distribution
        super().__init__(*args, nblocks=1, **kwargs)

        # task: IntegralTask from gpaw.response.integrators
        # wd: FrequencyDescriptor from gpaw.response.frequencies
        self.task, self.wd = self.construct_integral_task_and_wd()

    def calculate(self, wd, rate) -> Chi0DrudeData:
        """Calculate the Drude dielectric response.

        Parameters
        ----------
        wd : FrequencyDescriptor from gpaw.response.frequencies
            Frequencies to evaluate the reponse function at.
        rate : float
            Plasma frequency decay rate (in eV), corresponding to the
            imaginary part of the complex frequency.
        """
        self.print_info(wd, rate)

        chi0_drude = Chi0DrudeData.from_frequency_descriptor(wd, rate)
        self._calculate(chi0_drude)

        return chi0_drude

    def _calculate(self, chi0_drude: Chi0DrudeData):
        """In-place calculation of the Drude dielectric response function,
        based on the free-space plasma frequency of the intraband transitions.
        """
        q_c = [0., 0., 0.]
        # symmetries: QSymmetries from gpaw.response.symmetry
        # generator: KPointDomainGenerator from gpaw.response.kpoints
        # domain: Domain from from gpaw.response.integrators
        symmetries, generator, domain, prefactor = self.get_integration_domain(
            q_c=q_c, spins=range(self.gs.nspins))

        # The plasma frequency integral is special in the way that only
        # the spectral part is needed
        integrand = PlasmaFrequencyIntegrand(
            self, generator, self.gs.gd.cell_cv)

        # Integrate using temporary array
        tmp_plasmafreq_wvv = np.zeros((1,) + chi0_drude.vv_shape, complex)

        # integrator: Integrator from gpaw.response.integrators (or child)
        self.integrator.integrate(task=self.task,
                                  domain=domain,  # Integration domain
                                  integrand=integrand,
                                  wd=self.wd,
                                  out_wxx=tmp_plasmafreq_wvv)  # Output array
        tmp_plasmafreq_wvv *= prefactor

        # Symmetrize the plasma frequency
        operators = HeadSymmetryOperators(symmetries, self.gs.gd)
        plasmafreq_vv = tmp_plasmafreq_wvv[0].copy()
        operators.symmetrize_wvv(plasmafreq_vv[np.newaxis])

        # Store and print the plasma frequency
        chi0_drude.plasmafreq_vv += 4 * np.pi * plasmafreq_vv
        self.context.print('Plasma frequency:', flush=False)
        self.context.print((chi0_drude.plasmafreq_vv**0.5 * Ha).round(2))

        # Calculate the Drude dielectric response function from the
        # free-space plasma frequency
        # χ_D(ω+iη) = ω_p^2 / (ω+iη)^2

        # zd: ComplexFrequencyDescriptor from gpaw.response.frequencies
        assert chi0_drude.zd.upper_half_plane
        chi0_drude.chi_Zvv += plasmafreq_vv[np.newaxis] \
            / chi0_drude.zd.hz_z[:, np.newaxis, np.newaxis]**2

    def construct_integral_task_and_wd(self):
        if self.integrationmode == 'tetrahedron integration':
            # Calculate intraband transitions at T=0
            fermi_level = self.gs.fermi_level
            wd = FrequencyGridDescriptor([-fermi_level])
            task = HilbertTetrahedron(self.integrator.blockcomm)
        else:
            task = Intraband()

            # We want to pass None for frequency descriptor, but
            # if that goes wrong we'll get TypeError which is unhelpful.
            # This dummy class will give us error messages that allow finding
            # this spot in the code.
            class NotAFrequencyDescriptor:
                pass

            wd = NotAFrequencyDescriptor()
        return task, wd

    def print_info(self, wd, rate):
        isl = ['',
               f'{ctime()}',
               'Calculating drude chi0 with:',
               f'    Number of frequency points:{len(wd)}',
               f'    Plasma frequency decay rate: {rate} eV',
               '',
               self.get_gs_info_string(tab='    ')]

        # context: ResponseContext from gpaw.response.context
        self.context.print('\n'.join(isl))


class PlasmaFrequencyIntegrand(Integrand):
    def __init__(self, chi0drudecalc: Chi0DrudeCalculator,
                 generator: KPointDomainGenerator,
                 cell_cv: np.ndarray):
        self._drude = chi0drudecalc
        self.generator = generator
        self.cell_cv = cell_cv

    def _band_summation(self):
        # Intraband response needs only integrate partially unoccupied bands.
        # gs: ResponseGroundStateAdapter from gpaw.response.groundstate
        return self._drude.gs.nocc1, self._drude.gs.nocc2

    def matrix_element(self, point):
        """NB: In dire need of documentation! XXX."""
        k_v = point.kpt_c  # XXX _v vs _c discrepancy
        n1, n2 = self._band_summation()
        k_c = np.dot(self.cell_cv, k_v) / (2 * np.pi)

        # kptpair_factory: KPointPairFactory from gpaw.response.pair
        kptpair_factory = self._drude.kptpair_factory

        K0 = kptpair_factory.gs.kpoints.kptfinder.find(k_c)  # XXX

        kpt1 = kptpair_factory.get_k_point(point.spin, K0, n1, n2)
        n_n = np.arange(n1, n2)

        pair_calc = kptpair_factory.pair_calculator(
            blockcomm=self._drude.blockcomm)
        vel_nv = pair_calc.intraband_pair_density(kpt1, n_n)

        if self._drude.integrationmode == 'point integration':
            f_n = kpt1.f_n
            width = self._drude.gs.get_occupations_width()
            if width > 1e-15:
                dfde_n = - 1. / width * (f_n - f_n**2.0)
            else:
                dfde_n = np.zeros_like(f_n)
            vel_nv *= np.sqrt(-dfde_n[:, np.newaxis])
            weight = np.sqrt(self.generator.get_kpoint_weight(k_c) /
                             self.generator.how_many_symmetries())
            vel_nv *= weight

        return vel_nv

    def eigenvalues(self, point):
        """A function that can return the intraband eigenvalues.

        A method describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        n1, n2 = self._band_summation()
        # gs: ResponseGroundStateAdapter from gpaw.response.groundstate
        gs = self._drude.gs
        # kd: KPointDescriptor object from gpaw.kpt_descriptor
        kd = gs.kd
        k_v = point.kpt_c  # XXX v/c discrepancy
        # gd: GridDescriptor from gpaw.grid_descriptor
        k_c = np.dot(self.cell_cv, k_v) / (2 * np.pi)
        K1 = gs.kpoints.kptfinder.find(k_c)
        ik = kd.bz2ibz_k[K1]
        kpt1 = gs.kpt_qs[ik][point.spin]
        assert gs.kd.comm.size == 1

        return kpt1.eps_n[n1:n2]
