from __future__ import annotations

from pathlib import Path

import numpy as np

from ase.units import Hartree

from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.pair_functions import Chi, get_pw_coordinates
from gpaw.response.qpd import SingleQPWDescriptor
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.coulomb_kernels import NewCoulombKernel
from gpaw.response.fxc_kernels import FXCKernel, AdiabaticFXCCalculator
from gpaw.response.dyson import (DysonSolver, HXCKernel, HXCScaling, PWKernel,
                                 NoKernel)


class ChiFactory:
    r"""User interface to calculate individual elements of the four-component
    susceptibility tensor χ^μν, see [PRB 103, 245110 (2021)]."""

    def __init__(self,
                 chiks_calc: ChiKSCalculator,
                 fxc_calculator: AdiabaticFXCCalculator | None = None):
        """Contruct a many-body susceptibility factory."""
        self.chiks_calc = chiks_calc
        self.gs = chiks_calc.gs
        self.context = chiks_calc.context
        self.dyson_solver = DysonSolver(self.context)

        # If no fxc_calculator is supplied, fall back to default
        if fxc_calculator is None:
            fxc_calculator = AdiabaticFXCCalculator.from_rshe_parameters(
                self.gs, self.context)
        else:
            assert fxc_calculator.gs is chiks_calc.gs
            assert fxc_calculator.context is chiks_calc.context
        self.fxc_calculator = fxc_calculator

        # Prepare a buffer for the fxc kernels
        self.fxc_kernel_cache: dict[str, FXCKernel] = {}

    def __call__(self, spincomponent, q_c, complex_frequencies,
                 fxc: str | None = None, hxc_scaling: HXCScaling | None = None,
                 txt=None) -> tuple[Chi, Chi]:
        r"""Calculate a given element (spincomponent) of the four-component
        Kohn-Sham susceptibility tensor and construct a corresponding many-body
        susceptibility object within a given approximation to the
        exchange-correlation kernel.

        Parameters
        ----------
        spincomponent : str
            Spin component (μν) of the susceptibility.
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
        q_c : list or ndarray
            Wave vector
        complex_frequencies : np.array or ComplexFrequencyDescriptor
            Array of complex frequencies to evaluate the response function at
            or a descriptor of those frequencies.
        fxc : str or None
            Approximation to the (local) xc kernel. If left as None, xc-effects
            are neglected from the Dyson equation (RPA).
            Other choices: ALDA, ALDA_X, ALDA_x
        hxc_scaling : HXCScaling (or None, if irrelevant)
            Supply an HXCScaling object to scale the hxc kernel.
        txt : str
            Save output of the calculation of this specific component into
            a file with the filename of the given input.
        """
        # Initiate new output file, if supplied
        if txt is not None:
            self.context.new_txt_and_timer(txt)

        # Print to output file
        self.context.print('---------------', flush=False)
        self.context.print('Calculating susceptibility spincomponent='
                           f'{spincomponent} with q_c={q_c}', flush=False)
        self.context.print('---------------')

        # Calculate chiks
        chiks = self.calculate_chiks(spincomponent, q_c, complex_frequencies)
        # Construct the hxc kernel
        hxc_kernel = self.get_hxc_kernel(fxc, spincomponent, chiks.qpd)
        # Solve dyson equation
        chi = self.dyson_solver(chiks, hxc_kernel, hxc_scaling=hxc_scaling)

        return chiks, chi

    def get_hxc_kernel(self, fxc: str | None, spincomponent: str,
                       qpd: SingleQPWDescriptor) -> HXCKernel:
        return HXCKernel(
            hartree_kernel=self.get_hartree_kernel(spincomponent, qpd),
            xc_kernel=self.get_xc_kernel(fxc, spincomponent, qpd))

    def get_hartree_kernel(self, spincomponent: str,
                           qpd: SingleQPWDescriptor) -> PWKernel:
        if spincomponent in ['+-', '-+']:
            # No Hartree term in Dyson equation
            return NoKernel.from_qpd(qpd)
        else:
            return NewCoulombKernel.from_qpd(
                qpd, N_c=self.gs.kd.N_c, pbc_c=self.gs.atoms.get_pbc())

    def get_xc_kernel(self, fxc: str | None, spincomponent: str,
                      qpd: SingleQPWDescriptor) -> PWKernel:
        """Get the requested xc-kernel object."""
        if fxc is None:
            # No xc-kernel
            return NoKernel.from_qpd(qpd)

        if qpd.gammacentered:
            # When using a gamma-centered plane-wave basis, we can reuse the
            # fxc kernel for all q-vectors. Thus, we keep a cache of calculated
            # kernels
            key = f'{fxc},{spincomponent}'
            if key not in self.fxc_kernel_cache:
                self.fxc_kernel_cache[key] = self.fxc_calculator(
                    fxc, spincomponent, qpd)
            fxc_kernel = self.fxc_kernel_cache[key]
        else:
            # Always compute the kernel
            fxc_kernel = self.fxc_calculator(fxc, spincomponent, qpd)

        return fxc_kernel

    def calculate_chiks(self, spincomponent, q_c, complex_frequencies):
        """Calculate the Kohn-Sham susceptibility."""
        q_c = np.asarray(q_c)
        if isinstance(complex_frequencies, ComplexFrequencyDescriptor):
            zd = complex_frequencies
        else:
            zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

        # Perform actual calculation
        chiks = self.chiks_calc.calculate(spincomponent, q_c, zd)
        # Distribute frequencies over world
        chiks = chiks.copy_with_global_frequency_distribution()

        return chiks


def spectral_decomposition(chi, pos_eigs=1, neg_eigs=0):
    """Decompose the susceptibility in terms of spectral functions.

    The full spectrum of induced excitations is extracted and separated into
    contributions corresponding to the pos_eigs and neg_eigs largest positive
    and negative eigenvalues respectively.
    """
    # Initiate an EigendecomposedSpectrum object with the full spectrum
    full_spectrum = EigendecomposedSpectrum.from_chi(chi)

    # Separate the positive and negative eigenvalues for each frequency
    Apos = full_spectrum.get_positive_eigenvalue_spectrum()
    Aneg = full_spectrum.get_negative_eigenvalue_spectrum()

    # Keep only a fixed number of eigenvalues
    Apos = Apos.reduce_number_of_eigenvalues(pos_eigs)
    Aneg = Aneg.reduce_number_of_eigenvalues(neg_eigs)

    return Apos, Aneg


class EigendecomposedSpectrum:
    """Data object for eigendecomposed susceptibility spectra."""

    def __init__(self, omega_w, G_Gc, s_we, v_wGe, A_w=None,
                 wblocks: Blocks1D | None = None):
        """Construct the EigendecomposedSpectrum.

        Parameters
        ----------
        omega_w : np.array
            Global array of frequencies in eV.
        G_Gc : np.array
            Reciprocal lattice vectors in relative coordinates.
        s_we : np.array
            Sorted eigenvalues (in decreasing order) at all frequencies.
            Here, e is the eigenvalue index.
        v_wGe : np.array
            Eigenvectors for corresponding to the (sorted) eigenvalues. With
            all eigenvalues present in the representation, v_Ge should
            constitute the unitary transformation matrix between the eigenbasis
            and the plane-wave representation.
        A_w : np.array or None
            Full spectral weight as a function of frequency. If given as None,
            A_w will be calculated as the sum of all eigenvalues (equal to the
            trace of the spectrum, if no eigenvalues have been discarded).
        wblocks : Blocks1D
            Frequency block parallelization, if any.
        """
        self.omega_w = omega_w
        self.G_Gc = G_Gc

        self.s_we = s_we
        self.v_wGe = v_wGe

        self._A_w = A_w
        if wblocks is None:
            # Create a serial Blocks1D instance
            from gpaw.mpi import serial_comm
            wblocks = Blocks1D(serial_comm, len(omega_w))
        self.wblocks = wblocks

    @classmethod
    def from_chi(cls, chi):
        """Construct the eigendecomposed spectrum of a given susceptibility.

        The spectrum of induced excitations, S_GG'^(μν)(q,ω), which are encoded
        in a given susceptibility, can be extracted directly from its the
        dissipative part:

                            1
        S_GG'^(μν)(q,ω) = - ‾ χ_GG'^(μν")(q,ω)
                            π
        """
        assert chi.distribution == 'zGG'

        # Extract the spectrum of induced excitations
        chid = chi.copy_dissipative_part()
        S_wGG = - chid.array / np.pi

        # Extract frequencies (in eV) and reciprocal lattice vectors
        omega_w = chid.zd.omega_w * Hartree
        G_Gc = get_pw_coordinates(chid.qpd)

        return cls.from_spectrum(omega_w, G_Gc, S_wGG, wblocks=chid.blocks1d)

    @classmethod
    def from_spectrum(cls, omega_w, G_Gc, S_wGG, wblocks=None):
        """Perform an eigenvalue decomposition of a given spectrum."""
        # Find eigenvalues and eigenvectors of the spectrum
        s_wK, v_wGK = np.linalg.eigh(S_wGG)

        # Sort by spectral intensity (eigenvalues in descending order)
        sorted_indices_wK = np.argsort(-s_wK)
        s_we = np.take_along_axis(s_wK, sorted_indices_wK, axis=1)
        v_wGe = np.take_along_axis(
            v_wGK, sorted_indices_wK[:, np.newaxis, :], axis=2)

        return cls(omega_w, G_Gc, s_we, v_wGe, wblocks=wblocks)

    @classmethod
    def from_file(cls, filename):
        """Construct the eigendecomposed spectrum from a .npz file."""
        assert Path(filename).suffix == '.npz', filename
        npz = np.load(filename)
        return cls(npz['omega_w'], npz['G_Gc'],
                   npz['s_we'], npz['v_wGe'], A_w=npz['A_w'])

    def write(self, filename):
        """Write the eigendecomposed spectrum as a .npz file."""
        assert Path(filename).suffix == '.npz', filename

        # Gather data from the different blocks of frequencies to root
        s_we = self.wblocks.gather(self.s_we)
        v_wGe = self.wblocks.gather(self.v_wGe)
        A_w = self.wblocks.gather(self.A_w)

        # Let root write the spectrum to a pickle file
        if self.wblocks.blockcomm.rank == 0:
            np.savez(filename, omega_w=self.omega_w, G_Gc=self.G_Gc,
                     s_we=s_we, v_wGe=v_wGe, A_w=A_w)

    @property
    def nG(self):
        return self.G_Gc.shape[0]

    @property
    def neigs(self):
        return self.s_we.shape[1]

    @property
    def A_w(self):
        if self._A_w is None:
            self._A_w = np.nansum(self.s_we, axis=1)
        return self._A_w

    @property
    def A_wGG(self):
        """Generate the spectrum from the eigenvalues and eigenvectors."""
        A_wGG = np.empty((self.wblocks.nlocal, self.nG, self.nG),
                         dtype=complex)
        for w, (s_e, v_Ge) in enumerate(zip(self.s_we, self.v_wGe)):
            emask = ~np.isnan(s_e)
            svinv_eG = s_e[emask][:, np.newaxis] * np.conj(v_Ge.T[emask])
            A_wGG[w] = v_Ge[:, emask] @ svinv_eG
        return A_wGG

    def new_nan_arrays(self, neigs):
        """Allocate new eigenvalue and eigenvector arrays filled with np.nan.
        """
        s_we = np.empty((self.wblocks.nlocal, neigs), dtype=self.s_we.dtype)
        v_wGe = np.empty((self.wblocks.nlocal, self.nG, neigs),
                         dtype=self.v_wGe.dtype)
        s_we[:] = np.nan
        v_wGe[:] = np.nan
        return s_we, v_wGe

    def get_positive_eigenvalue_spectrum(self):
        """Create a new EigendecomposedSpectrum from the positive eigenvalues.

        This is especially useful in order to separate the full spectrum of
        induced excitations, see [PRB 103, 245110 (2021)],

        S_GG'^μν(q,ω) = A_GG'^μν(q,ω) - A_(-G'-G)^νμ(-q,-ω)

        into the ν and μ components of the spectrum. Since the spectral
        function A_GG'^μν(q,ω) is positive definite or zero (in regions without
        excitations), A_GG'^μν(q,ω) simply corresponds to the positive
        eigenvalue contribution to the full spectrum S_GG'^μν(q,ω).
        """
        # Find the maximum number of positive eigenvalues across the entire
        # frequency range
        if self.wblocks.nlocal > 0:
            pos_we = self.s_we > 0.
            npos_max = int(np.max(np.sum(pos_we, axis=1)))
        else:
            npos_max = 0
        npos_max = self.wblocks.blockcomm.max_scalar(npos_max)

        # Allocate new arrays, using np.nan for padding (the number of positive
        # eigenvalues might vary with frequency)
        s_we, v_wGe = self.new_nan_arrays(npos_max)

        # Fill arrays with the positive eigenvalue data
        for w, (s_e, v_Ge) in enumerate(zip(self.s_we, self.v_wGe)):
            pos_e = s_e > 0.
            npos = np.sum(pos_e)
            s_we[w, :npos] = s_e[pos_e]
            v_wGe[w, :, :npos] = v_Ge[:, pos_e]

        return EigendecomposedSpectrum(self.omega_w, self.G_Gc, s_we, v_wGe,
                                       wblocks=self.wblocks)

    def get_negative_eigenvalue_spectrum(self):
        """Create a new EigendecomposedSpectrum from the negative eigenvalues.

        The spectrum is created by reversing and negating the spectrum,

        -S_GG'^μν(q,-ω) = -A_GG'^μν(q,-ω) + A_(-G'-G)^νμ(-q,ω),

        from which the spectral function A_GG'^νμ(q,ω) can be extracted as the
        positive eigenvalue contribution, thanks to the reciprocity relation

                                  ˍˍ
        χ_GG'^μν(q,ω) = χ_(-G'-G)^νμ(-q,ω),
                   ˍ
        in which n^μ(r) denotes the hermitian conjugate [n^μ(r)]^†, and which
        is valid for μν ∊ {00,0z,zz,+-} in collinear systems without spin-orbit
        coupling.
        """
        # Negate the spectral function, its frequencies and reverse the order
        # of eigenvalues
        omega_w = - self.omega_w
        s_we = - self.s_we[:, ::-1]
        v_wGe = self.v_wGe[..., ::-1]
        inverted_spectrum = EigendecomposedSpectrum(omega_w, self.G_Gc,
                                                    s_we, v_wGe,
                                                    wblocks=self.wblocks)

        return inverted_spectrum.get_positive_eigenvalue_spectrum()

    def reduce_number_of_eigenvalues(self, neigs):
        """Create a new spectrum with only the neigs largest eigenvalues.

        The returned EigendecomposedSpectrum is constructed to retain knowledge
        of the full spectral weight of the unreduced spectrum through the A_w
        attribute.
        """
        assert self.nG >= neigs
        # Check that the available eigenvalues are in descending order
        assert all([np.all(np.logical_not(s_e[1:] - s_e[:-1] > 0.))
                    for s_e in self.s_we]), \
            'Eigenvalues needs to be sorted in descending order!'

        # Create new output arrays with the requested number of eigenvalues,
        # using np.nan for padding
        s_we, v_wGe = self.new_nan_arrays(neigs)
        # In reality, there may be less actual eigenvalues than requested,
        # since the user usually does not know how many e.g. negative
        # eigenvalues persist on the positive frequency axis (or vice-versa).
        # Fill in available eigenvalues up to the requested number.
        neigs = min(neigs, self.neigs)
        s_we[:, :neigs] = self.s_we[:, :neigs]
        v_wGe[..., :neigs] = self.v_wGe[..., :neigs]

        return EigendecomposedSpectrum(self.omega_w, self.G_Gc, s_we, v_wGe,
                                       # Keep the full spectral weight
                                       A_w=self.A_w,
                                       wblocks=self.wblocks)

    def get_eigenmode_lineshapes(self, nmodes=1, wmask=None):
        """Extract the spectral lineshapes of the eigenmodes.

        The spectral lineshape is calculated as the inner product

        a^μν_n(q,ω) = <v^μν_n(q)| A^μν(q,ω) |v^μν_n(q)>

        where the eigenvectors |v^μν_n> diagonalize the full spectral function
        at an appropriately chosen frequency ω_m:

        S^μν(q,ω_m) |v^μν_n(q)> = s^μν_n(q,ω_m) |v^μν_n(q)>
        """
        wm = self.get_eigenmode_frequency(nmodes=nmodes, wmask=wmask)
        return self.get_eigenmodes_at_frequency(wm, nmodes=nmodes)

    def get_eigenmode_frequency(self, nmodes=1, wmask=None):
        """Get the frequency at which to extract the eigenmodes.

        Generally, we chose the frequency ω_m to maximize the minimum
        eigenvalue difference

        ω_m(q) = maxmin_ωn[s^μν_n(q,ω) - s^μν_n+1(q,ω)]

        where n only runs over the desired number of modes (and the eigenvalues
        are sorted in descending order).

        However, in the case where only a single mode is extracted, we use the
        frequency at which the eigenvalue is maximal.
        """
        assert nmodes <= self.neigs
        # Use wmask to specify valid eigenmode frequencies
        wblocks = self.wblocks
        if wmask is None:
            wmask = np.ones(self.wblocks.N, dtype=bool)
        if nmodes == 1:
            # Find frequency where the eigenvalue is maximal
            s_w = wblocks.all_gather(self.s_we[:, 0])
            wm = np.nanargmax(s_w * wmask)  # skip np.nan padding
        else:
            # Find frequency with maximum minimal difference between size of
            # eigenvalues
            ds_we = np.array([self.s_we[:, e] - self.s_we[:, e + 1]
                              for e in range(nmodes - 1)]).T
            dsmin_w = np.min(ds_we, axis=1)
            dsmin_w = wblocks.all_gather(dsmin_w)
            wm = np.nanargmax(dsmin_w * wmask)  # skip np.nan padding

        return wm

    def get_eigenmodes_at_frequency(self, wm, nmodes=1):
        """Extract the eigenmodes at a specific frequency index wm."""
        v_Gm = self.get_eigenvectors_at_frequency(wm, nmodes=nmodes)
        return self._get_eigenmode_lineshapes(v_Gm)

    def get_eigenvectors_at_frequency(self, wm, nmodes=1):
        """Extract the eigenvectors at a specific frequency index wm."""
        wblocks = self.wblocks
        root, wmlocal = wblocks.find_global_index(wm)
        if wblocks.blockcomm.rank == root:
            v_Ge = self.v_wGe[wmlocal]
            v_Gm = np.ascontiguousarray(v_Ge[:, :nmodes])
        else:
            v_Gm = np.empty((self.nG, nmodes), dtype=complex)
        wblocks.blockcomm.broadcast(v_Gm, root)

        return v_Gm

    def _get_eigenmode_lineshapes(self, v_Gm):
        """Extract the eigenmode lineshape based on the mode eigenvectors."""
        wblocks = self.wblocks
        A_wGG = self.A_wGG
        a_wm = np.empty((wblocks.nlocal, v_Gm.shape[1]), dtype=float)
        for m, v_G in enumerate(v_Gm.T):
            a_w = np.conj(v_G) @ A_wGG @ v_G
            assert np.allclose(a_w.imag, 0.)
            a_wm[:, m] = a_w.real
        a_wm = wblocks.all_gather(a_wm)

        return a_wm

    def write_full_spectral_weight(self, filename):
        A_w = self.wblocks.gather(self.A_w)
        if self.wblocks.blockcomm.rank == 0:
            write_full_spectral_weight(filename, self.omega_w, A_w)

    def write_eigenmode_lineshapes(self, filename, **kwargs):
        a_wm = self.get_eigenmode_lineshapes(**kwargs)
        if self.wblocks.blockcomm.rank == 0:
            write_eigenmode_lineshapes(filename, self.omega_w, a_wm)


def write_full_spectral_weight(filename, omega_w, A_w):
    """Write the sum of spectral weights A(ω) to a file."""
    with open(filename, 'w') as fd:
        print('# {:>11}, {:>11}'.format('omega [eV]', 'A(w)'), file=fd)
        for omega, A in zip(omega_w, A_w):
            print(f'  {omega:11.6f}, {A:11.6f}', file=fd)


def read_full_spectral_weight(filename):
    """Read a stored full spectral weight file."""
    data = np.loadtxt(filename, delimiter=',')
    omega_w = np.array(data[:, 0], float)
    A_w = np.array(data[:, 1], float)
    return omega_w, A_w


def write_eigenmode_lineshapes(filename, omega_w, a_wm):
    """Write the eigenmode lineshapes a^μν_n(ω) to a file."""
    with open(filename, 'w') as fd:
        # Print header
        header = '# {:>11}'.format('omega [eV]')
        for m in range(a_wm.shape[1]):
            header += ', {:>11}'.format(f'a_{m}(w)')
        print(header, file=fd)
        # Print data
        for omega, a_m in zip(omega_w, a_wm):
            data = f'  {omega:11.6f}'
            for a in a_m:
                data += f', {a:11.6f}'
            print(data, file=fd)


def read_eigenmode_lineshapes(filename):
    """Read a stored eigenmode lineshapes file."""
    data = np.loadtxt(filename, delimiter=',')
    omega_w = np.array(data[:, 0], float)
    a_wm = np.array(data[:, 1:], float)
    return omega_w, a_wm
