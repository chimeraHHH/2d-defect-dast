"""Functionality to calculate the all-electron Fourier components of local
functions of the electon (spin-)density."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import spherical_jn

from ase.units import Bohr

from gpaw.response import ResponseGroundStateAdapter, ResponseContext, timer

from gpaw.spherical_harmonics import Yarr
from gpaw.sphere.rshe import calculate_reduced_rshe
from gpaw.xc import XC
from gpaw.xc.libxc import LibXC


class LocalFTCalculator(ABC):
    r"""Calculator base class for calculators of all-electron plane-wave
    components to arbitrary real-valued real-space functionals f[n](r) which
    can be written as closed form functions of the local ground state
    (spin-)density:

    f[n](r) = f(n(r)).

    Since n(r) is lattice periodic, so is f(r) and the plane-wave components
    can be calculated as (see [PRB 103, 245110 (2021)] for definitions)

           /
    f(G) = |dr f(r) e^(-iG.r),
           /
            V0

    where V0 is the unit-cell volume.
    """

    def __init__(self, gs, context, bg_density=None):
        """Constructor for the LocalFTCalculator

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
            Adapter containing relevant information about the underlying DFT
            ground state
        context : ResponseContext
        bg_density : float
            Spin-neutral background electron density (in Å^-3) to add to the
            actual electron density in order to regularize functions which
            diverge in vacuum.
        """
        assert isinstance(gs, ResponseGroundStateAdapter)
        self.gs = gs
        assert isinstance(context, ResponseContext)
        self.context = context

        if bg_density is None:
            self.bg_density = None
        else:
            assert isinstance(bg_density, float)
            # Convert to atomic units
            self.bg_density = bg_density * Bohr**3.  # Å^-3 -> Bohr^-3

    @staticmethod
    def from_rshe_parameters(gs, context, bg_density=None,
                             rshelmax=-1, rshewmin=None):
        """Construct the LocalFTCalculator based on parameters for the
        expansion of the PAW correction in real spherical harmonics

        Parameters
        ----------
        rshelmax : int or None
            Expand f(r) in real spherical harmonics inside the augmentation
            spheres. If None, the plane-wave components will be calculated
            without augmentation. The value of rshelmax indicates the maximum
            index l to perform the expansion in (l < 6).
        rshewmin : float or None
            If None, the PAW correction will be fully expanded up to the chosen
            lmax. Given as a float (0 < rshewmin < 1), rshewmin indicates what
            coefficients to use in the expansion. If any (l,m) coefficient
            contributes with less than a fraction of rshewmin on average, it
            will not be included.
        """
        if rshelmax is None:
            return LocalGridFTCalculator(gs, context, bg_density=bg_density)
        else:
            return LocalPAWFTCalculator(gs, context, bg_density=bg_density,
                                        rshelmax=rshelmax, rshewmin=rshewmin)

    @timer('LocalFT')
    def __call__(self, qpd, add_f):
        """Calculate the plane-wave components f(G).

        Parameters
        ----------
        qpd : SingleQPWDescriptor
            Defines the plane-wave basis to calculate the components in.
        add_f : method
            Defines the local function of the electron (spin-)density to
            Fourier transform. Should take arguments gd (GridDescriptor),
            n_sR (electron spin-density on the real space grid of gd) and
            f_R (output array) and add the function f(R) to the output array.
            Example:
            >>> def add_total_density(gd, n_sR, f_R):
            ...     f_R += np.sum(n_sR, axis=0)

        Returns
        -------
        f_G : np.array
            Plane-wave components of the function f, indexes by the reciprocal
            lattice vectors G.
        """
        self.context.print('Calculating f(G)')
        f_G = self.calculate(qpd, add_f)
        self.context.print('Finished calculating f(G)')

        return f_G

    @abstractmethod
    def calculate(self, qpd, add_f):
        pass

    @staticmethod
    def equivalent_real_space_grids(gd1, gd2):
        assert gd1.comm.size == 1
        assert gd2.comm.size == 1
        return (gd1.N_c == gd2.N_c).all()

    def get_electron_density(self, gd, pseudo=False):
        """Get the electron density corresponding to a given grid descriptor.
        """
        gridrefinement = self.get_gridrefinement(gd)

        if pseudo:
            _get_electron_density = self.gs.get_pseudo_density
        else:
            _get_electron_density = self.gs.get_all_electron_density

        n_sR, gdref = _get_electron_density(gridrefinement=gridrefinement)

        assert self.equivalent_real_space_grids(gd, gdref)

        if self.bg_density is not None:
            # Add spin-neutral background electron density
            self.context.print('    Adding a background a background electron '
                               f'density of {self.bg_density / Bohr**3.} Å^-3')
            n_sR = n_sR.copy()  # Make a copy in order not to modify gs
            n_sR += self.bg_density / n_sR.shape[0]

        return n_sR

    def get_gridrefinement(self, gd):
        if self.equivalent_real_space_grids(gd, self.gs.gd):
            gridrefinement = 1
        elif self.equivalent_real_space_grids(gd, self.gs.finegd):
            gridrefinement = 2
        else:
            raise ValueError('The supplied gd is neither compatible with the '
                             'coarse nor the fine real-space grid of the '
                             'underlying ground state')

        return gridrefinement


class LocalGridFTCalculator(LocalFTCalculator):

    def calculate(self, qpd, add_f):
        """Calculate f(G) directly from the all-electron density on the cubic
        real-space grid."""
        n_sR = self.get_all_electron_density(qpd.gd)
        f_G = self._calculate(qpd, n_sR, add_f)

        return f_G

    def _calculate(self, qpd, n_sR, add_f):
        """In-place calculation of the plane-wave components."""
        # Calculate f(r)
        gd = qpd.gd
        f_R = gd.zeros()
        add_f(gd, n_sR, f_R)

        # FFT to reciprocal space
        f_G = fft_from_grid(f_R, qpd)  # G = 1D grid of |G|^2/2 < ecut

        return f_G

    @timer('Calculate the all-electron density')
    def get_all_electron_density(self, gd):
        """Calculate the all-electron (spin-)density."""
        self.context.print('    Calculating the all-electron density')
        return self.get_electron_density(gd)


class LocalPAWFTCalculator(LocalFTCalculator):

    def __init__(self, gs, context, bg_density=None,
                 rshelmax=-1, rshewmin=None):
        super().__init__(gs, context, bg_density=bg_density)

        self.engine = LocalPAWFTEngine(self.context, rshelmax, rshewmin)

    def calculate(self, qpd, add_f):
        """Calculate f(G) with an expansion of f(r) in real spherical harmonics
        inside the augmentation spheres."""
        # Retrieve the pseudo (spin-)density on the real-space grid
        nt_sR = self.get_pseudo_density(qpd.gd)  # R = 3D real-space grid

        # Retrieve the pseudo and all-electron atomic centered densities inside
        # the augmentation spheres
        R_av, micro_setups = self.extract_atom_centered_quantities()

        # Let the engine perform the in-place calculation
        f_G = self.engine.calculate(qpd, nt_sR, R_av, micro_setups, add_f)

        return f_G

    def get_pseudo_density(self, gd):
        """Get the pseudo (spin-)density of the ground state."""
        return self.get_electron_density(gd, pseudo=True)

    def extract_atom_centered_quantities(self):
        """Extract all relevant atom centered quantities that the engine needs
        in order to calculate PAW corrections. Most of the information is
        bundled as a list of MicroSetups for each atom."""
        R_av = self.gs.atoms.positions / Bohr
        micro_setups = self.gs.micro_setups
        return R_av, micro_setups


class MicroSetup:

    def __init__(self, rgd, Y_nL, n_sLg, nt_sLg):
        self.rgd = rgd
        self.Y_nL = Y_nL
        self.n_sLg = n_sLg
        self.nt_sLg = nt_sLg

    def evaluate_function(self, add_f):
        """Evaluate a given function f(r) on the angular and radial grids."""
        f_ng = np.array([self.rgd.zeros() for n in range(self.Y_nL.shape[0])])
        for n, Y_L in enumerate(self.Y_nL):
            n_sg = Y_L @ self.n_sLg
            add_f(self.rgd, n_sg, f_ng[n])
        return f_ng

    def evaluate_paw_correction(self, add_f):
        r"""Evaluate Δf_a[n_a,ñ_a](r) for a given function f(r).

        Returns
        -------
        df_ng : nd.array
            (f_ng - ft_ng) where (n=Lebedev index, g=radial grid index)
        """
        rgd = self.rgd
        f_g = rgd.zeros()
        ft_g = rgd.zeros()
        df_ng = np.array([rgd.zeros() for n in range(self.Y_nL.shape[0])])
        for n, Y_L in enumerate(self.Y_nL):
            f_g[:] = 0.
            n_sg = Y_L @ self.n_sLg
            add_f(rgd, n_sg, f_g)

            ft_g[:] = 0.
            nt_sg = Y_L @ self.nt_sLg
            add_f(rgd, nt_sg, ft_g)

            df_ng[n, :] = f_g - ft_g

        return df_ng

    def expand_function(self, add_f, **kwargs):
        f_ng = self.evaluate_function(add_f)
        return self.expand(f_ng, **kwargs)

    def expand_paw_correction(self, add_f, **kwargs):
        df_ng = self.evaluate_paw_correction(add_f)
        return self.expand(df_ng, **kwargs)

    def expand(self, f_ng, **kwargs):
        """Expand into real spherical harmonics."""
        return calculate_reduced_rshe(self.rgd, f_ng, self.Y_nL, **kwargs)


def extract_micro_setup(pawdata, D_sp) -> MicroSetup:
    """Extract the a.e. and pseudo (spin-)densities as a MicroSetup."""
    # Radial grid descriptor:
    rgd = pawdata.xc_correction.rgd
    # Spherical harmonics on the Lebedev quadrature:
    Y_nL = pawdata.xc_correction.Y_nL
    n_sLg, nt_sLg = calculate_atom_centered_densities(pawdata,
                                                      # atomic density matrix
                                                      D_sp)
    return MicroSetup(rgd, Y_nL, n_sLg, nt_sLg)


def calculate_atom_centered_densities(pawdata, D_sp):
    """Calculate the AE and pseudo densities inside the augmentation sphere.

    Returns
    -------
    n_sLg : nd.array
        all-electron density
    nt_sLg : nd.array
        pseudo density
    (s=spin, L=(l,m) spherical harmonic index, g=radial grid index)
    """
    n_qg = pawdata.xc_correction.n_qg
    nt_qg = pawdata.xc_correction.nt_qg
    nc_g = pawdata.xc_correction.nc_g
    nct_g = pawdata.xc_correction.nct_g

    B_pqL = pawdata.xc_correction.B_pqL
    D_sLq = np.inner(D_sp, B_pqL.T)
    nspins = len(D_sp)

    n_sLg = D_sLq @ n_qg
    nt_sLg = D_sLq @ nt_qg

    # Add core density
    n_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nc_g
    nt_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nct_g

    return n_sLg, nt_sLg


class LocalPAWFTEngine:

    def __init__(self, context, rshelmax=-1, rshewmin=None):
        """Construct the engine."""
        self.context = context
        self.rshelmax = rshelmax
        self.rshewmin = rshewmin

        self._add_f = None

    def calculate(self, qpd, nt_sR, R_av, micro_setups, add_f):
        r"""Calculate the Fourier transform f(G) by splitting up the
        calculation into a pseudo density contribution and a PAW correction
        accounting for the difference

        Δf[n,ñ](r) = f(n(r)) - f(ñ(r)),

        such that:

        f(G) = f[ñ](G) + Δf[n,ñ](G).

        See [PRB 103, 245110 (2021)] for definitions and notation details."""
        self._add_f = add_f

        ft_G = self.calculate_pseudo_contribution(qpd, nt_sR)
        fPAW_G = self.calculate_paw_corrections(qpd, R_av, micro_setups)

        return ft_G + fPAW_G

    def calculate_pseudo_contribution(self, qpd, nt_sR):
        """Calculate the pseudo density contribution by performing a FFT of
        f(ñ(r)) on the cubic real-space grid.

        NB: This operation assumes that the function f is a slowly varrying
        function of the pseudo density ñ(r) everywhere in space, such that
        f(ñ(r)) is accurately described on the cubic real-space grid."""
        # Calculate ft(r) (t=tilde=pseudo)
        gd = qpd.gd
        ft_R = gd.zeros()
        self._add_f(gd, nt_sR, ft_R)

        # FFT to reciprocal space
        ft_G = fft_from_grid(ft_R, qpd)  # G = 1D grid of |G|^2/2 < ecut

        return ft_G

    @timer('Calculate PAW corrections')
    def calculate_paw_corrections(self, qpd, R_av, micro_setups):
        r"""Calculate the PAW corrections to f(G), for each augmentation sphere
        at a time:
                      __
                      \   /
        Δf[n,ñ](G) =  /   |dr Δf_a[n_a,ñ_a](r - R_a) e^(-iG.r)
                      ‾‾  /
                      a    V0

        where Δf_a is the atom centered difference between the all electron
        and pseudo quantities inside augmentation sphere a:

        Δf_a[n_a,ñ_a](r) = f(n_a(r)) - f(ñ_a(r)).
        """
        self.context.print('    Calculating PAW corrections\n')

        # Extract reciprocal lattice vectors
        nG = qpd.ngmax
        G_Gv = qpd.get_reciprocal_vectors(add_q=False)
        assert G_Gv.shape[0] == nG

        # Allocate output array
        fPAW_G = np.zeros(nG, dtype=complex)

        # Distribute plane waves
        G_myG = self._distribute_correction(nG)
        G_myGv = G_Gv[G_myG]

        # Calculate and add the PAW corrections from each augmentation sphere
        for a, (R_v, micro_setup) in enumerate(zip(R_av, micro_setups)):
            self._add_paw_correction(a, R_v, micro_setup,
                                     G_myG, G_myGv, fPAW_G)

        self.context.comm.sum(fPAW_G)

        return fPAW_G

    def _distribute_correction(self, nG):
        comm = self.context.comm
        nGpr = (nG + comm.size - 1) // comm.size
        Ga = min(comm.rank * nGpr, nG)
        Gb = min(Ga + nGpr, nG)

        return range(Ga, Gb)

    def _add_paw_correction(self, a, R_v, micro_setup, G_myG, G_myGv, fPAW_G):
        r"""Calculate the PAW correction of augmentation sphere a,

                              /
        Δf_a(G) = e^(-iG.R_a) |dr Δf_a[n_a,ñ_a](r) e^(-iG.r),
                              /
                              V0

        by expanding both the atom centered correction and the plane wave in
        real spherical harmonics, see [PRB 103, 245110 (2021)]:

                                 l               a
                              __ __             R_c
                              \  \      l    ^  /                   a
        Δf_a(G) = e^(-iG.R_a) /  /  (-i)  Y (G) |4πr^2 dr j(|G|r) Δf (r)
                              ‾‾ ‾‾        lm   /          l        lm
                              l m=-l            0

        The calculated atomic correction is then added to the output array."""
        rgd = micro_setup.rgd

        # Expand Δf_a[n_a,ñ_a](r) in real spherical harmonics
        rshe, info_string = micro_setup.expand_paw_correction(
            self._add_f, lmax=self.rshelmax, wmin=self.rshewmin)
        self.print_rshe_info(a, info_string)

        # Expand the plane waves in real spherical harmonics (and spherical
        # Bessel functions)
        (ii_MmyG,
         j_gMmyG,
         Y_MmyG) = self._expand_plane_waves(
             G_myGv, rgd.r_g, rshe.L_M, rshe.l_M)

        # Calculate the PAW correction as an integral over the radial grid
        # and rshe coefficients
        with self.context.timer('Integrate PAW correction'):
            angular_coef_MmyG = ii_MmyG * Y_MmyG
            # Radial integral, dv = 4πr^2
            df_gM = rshe.f_gM
            radial_coef_MmyG = np.tensordot(j_gMmyG * df_gM[..., np.newaxis],
                                            rgd.dv_g, axes=([0, 0]))
            # Angular integral (sum over l,m)
            atomic_corr_myG = np.sum(angular_coef_MmyG * radial_coef_MmyG,
                                     axis=0)

            position_prefactor_myG = np.exp(-1j * np.inner(G_myGv, R_v))

            # Add to output array
            fPAW_G[G_myG] += position_prefactor_myG * atomic_corr_myG

    def print_rshe_info(self, a, info_string):
        """Print information about the expansion at atom a."""
        info_string = f'    RSHE of atom {a}\n' + info_string
        self.context.print(
            info_string.replace('\n', '\n      ') + '\n')

    @timer('Expand plane waves in real spherical harmonics')
    def _expand_plane_waves(self, G_myGv, r_g, L_M, l_M):
        r"""Expand plane waves in spherical Bessel functions and real spherical
        harmonics:
                        l
                    __  __
         -iG.r      \   \      l             ^     ^
        e      = 4π /   /  (-i)  j (|G|r) Y (G) Y (r)
                    ‾‾  ‾‾        l        lm    lm
                    l  m=-l

        Returns
        -------
        ii_MmyG : nd.array
            (-i)^l for used (l,m) coefficients M
        j_gMmyG : nd.array
            j_l(|G|r) for used (l,m) coefficients M
        Y_MmyG : nd.array
                 ^
            Y_lm(K) for used (l,m) coefficients M
        """
        nmyG = G_myGv.shape[0]
        Gnorm_myG, Gdir_myGv = self._calculate_norm_and_direction(G_myGv)

        # Setup arrays to fully vectorize computations
        nM = len(L_M)
        (r_gMmyG, l_gMmyG,
         Gnorm_gMmyG) = (a.reshape(len(r_g), nM, nmyG)
                         for a in np.meshgrid(r_g, l_M, Gnorm_myG,
                                              indexing='ij'))

        with self.context.timer('Compute spherical bessel functions'):
            # Slow step
            j_gMmyG = spherical_jn(l_gMmyG, Gnorm_gMmyG * r_gMmyG)

        Y_MmyG = Yarr(L_M, Gdir_myGv)
        ii_MmyG = (-1j) ** np.repeat(l_M, nmyG).reshape((nM, nmyG))

        return ii_MmyG, j_gMmyG, Y_MmyG

    @staticmethod
    def _calculate_norm_and_direction(G_myGv):
        """Calculate the length and direction of reciprocal lattice vectors."""
        Gnorm_myG = np.linalg.norm(G_myGv, axis=1)
        Gdir_myGv = np.zeros_like(G_myGv)
        mask0 = np.where(Gnorm_myG != 0.)
        Gdir_myGv[mask0] = G_myGv[mask0] / Gnorm_myG[mask0][:, np.newaxis]

        return Gnorm_myG, Gdir_myGv


def fft_from_grid(f_R, qpd):
    r"""Perform a FFT to reciprocal space:
                                    __
           /                    V0  \
    f(G) = |dr f(r) e^(-iG.r) ≃ ‾‾  /  f(r) e^(-iG.r)
           /                    N   ‾‾
           V0                       r

    where N is the number of grid points."""
    Q_G = qpd.Q_qG[0]

    # Perform the FFT
    N = np.prod(qpd.gd.N_c)
    f_Q123 = qpd.gd.volume / N * np.fft.fftn(f_R)  # Q123 = 3D grid in Q-rep

    # Change the view of the plane-wave components from the 3D grid in the
    # Q-representation that numpy spits out to the 1D grid in the
    # G-representation, that GPAW relies on internally
    f_G = f_Q123.ravel()[Q_G]

    return f_G


# ---------- Local functions of the (spin-)density ---------- #


def add_total_density(gd, n_sR, n_R):
    n_R += np.sum(n_sR, axis=0)


def add_spin_polarization(gd, n_sR, nz_R):
    nz_R += calculate_spin_polarization(n_sR)


def calculate_spin_polarization(n_sR):
    return n_sR[0] - n_sR[1]


def add_LSDA_Wxc(gd, n_sR, Wxc_R):
    Wxc_R += calculate_LSDA_Wxc(gd, n_sR)


def calculate_LSDA_Wxc(gd, n_sR):
    """Calculate W_xc^z in the local spin-density approximation.

    For a collinear system:

                δE_xc[n,n^z]   1
    W_xc^z(r) = ‾‾‾‾‾‾‾‾‾‾‾‾ = ‾ [V_LSDA^↑(r) - V_LSDA^↓(r)]
                   δn^z(r)     2
    """
    # Allocate an array for the spin-dependent xc potential on the real
    # space grid
    v_sR = np.zeros(np.shape(n_sR))

    # Calculate the spin-dependent potential
    xc = XC('LDA')
    xc.calculate(gd, n_sR, v_sg=v_sR)

    return (v_sR[0] - v_sR[1]) / 2


def add_LSDA_zeeman_energy(gd, n_sR, EZ_R):
    """Calculate and add the LSDA Zeeman energy to the output array.

    The Zeeman energy is defined as:

    E_Z(r) = - B^(xc)(r) m(r) = - W_xc^z n^z(r).
    """
    EZ_R += - calculate_LSDA_Wxc(gd, n_sR) * calculate_spin_polarization(n_sR)


def add_LDA_dens_fxc(gd, n_sR, fxc_R, *, fxc):
    r"""Calculate the LDA density kernel and add it to the output array fxc_R.

    The LDA density kernel is given by:

                    ∂^2[ϵ_xc(n,m)n] |
    f_LDA^(00)(r) = ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ |
                         ∂n^2       |n=n(r),m=m(r)
    """
    assert len(n_sR) == 1, \
        'The density kernel is untested for spin-polarized systems'

    if fxc == 'ALDA_x':
        fxc_R += -1. / 3. * (3. / np.pi)**(1. / 3.) * n_sR[0]**(-2. / 3.)
    else:
        assert fxc in ['ALDA_X', 'ALDA']
        kernel = LibXC(fxc[1:])
        fxc_sR = np.zeros_like(n_sR)
        kernel.xc.calculate_fxc_spinpaired(n_sR.ravel(), fxc_sR)

        fxc_R += fxc_sR[0]


def add_LSDA_trans_fxc(gd, n_sR, fxc_R, *, fxc):
    r"""Calculate the transverse LDA kernel and add it to the output arr. fxc_R

    The transverse LDA kernel is given by:

                     2  ∂[ϵ_xc(n,n^z)n] |
    f_LDA^(+-)(r) = ‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾  |
                    n^z      ∂n^z       |n=n(r),n^z=n^z(r)

                    V_LSDA^↑(r) - V_LSDA^↓(r)
                  = ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                              n^z(r)
    """
    assert len(n_sR) == 2  # nspins
    nz_R = n_sR[0] - n_sR[1]

    if fxc == 'ALDA_x':
        fxc_R += - (6. / np.pi)**(1. / 3.) \
            * (n_sR[0]**(1. / 3.) - n_sR[1]**(1. / 3.)) / nz_R
    else:
        assert fxc in ['ALDA_X', 'ALDA']
        v_sR = np.zeros(np.shape(n_sR))
        xc = XC(fxc[1:])
        xc.calculate(gd, n_sR, v_sg=v_sR)

        fxc_R += (v_sR[0] - v_sR[1]) / nz_R
