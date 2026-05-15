"""Test functionality to compute Fourier Transforms with PAW corrections"""

# General modules
import numpy as np
import pytest

# Script modules
from ase.units import Ha

from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.pw.descriptor import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LFC
from gpaw.atom.radialgd import AERadialGridDescriptor

from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.localft import (LocalFTCalculator, MicroSetup,
                                   add_total_density, add_LSDA_Wxc)
from gpaw.response.pair_functions import get_pw_coordinates
from gpaw.test.response.test_site_kernels import get_pw_descriptor


# ---------- Test parametrization ---------- #

# 1s orbital radii
testa_a = np.linspace(0.5, 1.5, 10)  # a.u.


def ae_1s_density(r_g, a=1.0):
    """Construct the radial dependence of the density from a 1s orbital on the
    radial grid r_g."""
    assert np.all(r_g >= 0)
    prefactor = 1 / (np.pi * a**3.)
    n_g = prefactor * np.exp(-2. * r_g / a)

    return n_g


def ae_1s_density_plane_waves(pd, R_v, a=1.0):
    """Calculate the plane-wave components of the density from a 1s
    orbital centered at a given position analytically."""
    # List of all plane waves
    G_Gv = np.array([pd.G_Qv[Q] for Q in pd.Q_qG[0]])
    Gnorm_G = np.linalg.norm(G_Gv, axis=1)

    position_prefactor_G = np.exp(-1.j * np.dot(G_Gv, R_v))
    atomcentered_n_G = 1 / (1 + (Gnorm_G * a / 2.)**2.)**2.

    n_G = position_prefactor_G * atomcentered_n_G

    return n_G


# ---------- Actual tests ---------- #

@pytest.mark.response
@pytest.mark.parametrize("a", testa_a)
def test_localft_grid_calculator(a):
    """Test that the LocalGridFTCalculator is able to correctly Fourier
    transform the all-electron density of an 1s orbital."""
    # ---------- Inputs ---------- #

    # Real-space grid
    lc_to_a_ratio = 10  # lattice constant to orbital radii
    N_grid_points = 50**3

    # Plane-wave cutoff
    relative_ecut = 20  # eV relative to a=1

    # Test tolerance
    rtol = 1e-3

    # ---------- Script ---------- #

    # Set up atomic position at the center of the unit cell
    lattice_constant = lc_to_a_ratio * a  # a.u.
    R_v = np.array([lattice_constant, lattice_constant,
                    lattice_constant]) / 2.  # Place atom at the center

    # Set up grid descriptor
    cell_cv = np.array([[lattice_constant, 0., 0.],
                        [0., lattice_constant, 0.],
                        [0., 0., lattice_constant]])
    N_c = np.array([int(N_grid_points**(1 / 3.))] * 3)
    gd = GridDescriptor(N_c, cell_cv=cell_cv, comm=mpi.serial_comm)

    # Set up plane-wave descriptor
    qd = KPointDescriptor(np.array([[0., 0., 0.]]))
    pd = PWDescriptor(relative_ecut / Ha / a**2., gd, complex, qd)

    # Calculate the plane-wave components analytically
    ntest_G = ae_1s_density_plane_waves(pd, R_v, a=a)

    # Calculate the atomic radius at all grid points
    r_vR = gd.get_grid_point_coordinates()
    r_R = np.linalg.norm(r_vR - R_v[:, np.newaxis, np.newaxis, np.newaxis],
                         axis=0)

    # Calculate the all-electron density on the real-space grid
    n_sR = np.array([ae_1s_density(r_R, a=a)])

    # Initialize the LocalGridFTCalculator with an empty ground state adapter
    gs = EmptyGSAdapter()  # hack to pass isinstance in constructor
    context = ResponseContext()
    localft_calc = LocalFTCalculator.from_rshe_parameters(gs, context,
                                                          rshelmax=None)

    # Compute the plane-wave components numerically
    n_G = localft_calc._calculate(pd, n_sR, add_total_density)

    # Check validity of results
    assert np.allclose(n_G, ntest_G, rtol=rtol)


@pytest.mark.response
@pytest.mark.parametrize("a", testa_a)
def test_localft_paw_engine(a):
    """Test that the LocalPAWFTEngine is able to correctly Fourier
    transform the all-electron density of an 1s orbital."""
    # ---------- Inputs ---------- #

    # Real-space grid
    lc_to_a_ratio = 10  # lattice constant to orbital radii
    N_grid_points = 50**3

    # Radial grid (using standard parameters from Li)
    rgd_a = 0.0023570226039551583
    rgd_b = 0.0004528985507246377
    rcut = 2.0  # a.u.

    # Plane-wave cutoff
    relative_ecut = 20  # eV relative to a=1

    # Settings for the expansion in real spherical harmonics
    rshe_params_p = [{},
                     {'rshelmax': 0},  # test that only l=0 contributes
                     {'rshewmin': 1e-8}]  # test coefficient filter

    # Test tolerance
    rtol = 5e-4

    # ---------- Script ---------- #

    # Set up atomic position at the center of the unit cell
    lattice_constant = lc_to_a_ratio * a  # a.u.
    R_v = np.array([lattice_constant, lattice_constant,
                    lattice_constant]) / 2.
    pos_ac = np.array([[0.5, 0.5, 0.5]])  # Relative atomic positions

    # Set up grid descriptor
    cell_cv = np.array([[lattice_constant, 0., 0.],
                        [0., lattice_constant, 0.],
                        [0., 0., lattice_constant]])
    N_c = np.array([int(N_grid_points**(1 / 3.))] * 3)
    gd = GridDescriptor(N_c, cell_cv=cell_cv, comm=mpi.serial_comm)

    # Set up radial grid descriptor extending all the way to the edge of the
    # unit cell
    redge = np.sqrt(3) * lattice_constant / 2.  # center-corner distance
    Ng = int(np.floor(redge / (rgd_a + rgd_b * redge)) + 1)
    rgd = AERadialGridDescriptor(rgd_a, rgd_b, N=Ng)

    # Set up plane-wave descriptor
    qd = KPointDescriptor(np.array([[0., 0., 0.]]))
    pd = PWDescriptor(relative_ecut / Ha / a**2., gd, complex, qd)

    # Calculate the plane-wave components analytically
    ntest_G = ae_1s_density_plane_waves(pd, R_v, a=a)

    # Calculate the pseudo and ae densities on the radial grid
    n_g = ae_1s_density(rgd.r_g, a=a)
    gcut = rgd.floor(rcut)
    nt_g, _ = rgd.pseudize(n_g, gcut)

    # Set up pseudo and ae densities on the Lebedev quadrature
    from gpaw.sphere.lebedev import Y_nL
    Y_nL = Y_nL[:, :9]  # include only s, p and d
    nL = Y_nL.shape[1]
    n_sLg = np.zeros((1, nL, Ng), dtype=float)
    nt_sLg = np.zeros((1, nL, Ng), dtype=float)
    # 1s <=> (l,m) = (0,0) <=> L = 0
    n_sLg[0, 0, :] += np.sqrt(4. * np.pi) * n_g  # Y_0 = 1 / sqrt(4pi)
    nt_sLg[0, 0, :] += np.sqrt(4. * np.pi) * nt_g

    # Calculate the pseudo density on the real-space grid
    # ------------------------------------------------- #
    # Generate splines on for the pseudo density on the radial grid
    spline = rgd.spline(nt_g, l=0, rcut=redge)
    # Use the LocalizedFunctionsCollection to generate pseudo density
    # on the cubic real space grid
    nt_R = gd.zeros()
    lfc = LFC(gd, [[spline]])
    lfc.set_positions(pos_ac)
    lfc.add(nt_R, c_axi=np.sqrt(4. * np.pi))  # Y_0 = 1 / sqrt(4pi)
    nt_sR = np.array([nt_R])

    # Create MicroSetup(s)
    micro_setup = MicroSetup(rgd, Y_nL, n_sLg, nt_sLg)
    micro_setups = [micro_setup]

    gs = EmptyGSAdapter()  # hack to pass isinstance in constructor
    context = ResponseContext()
    for rshe_params in rshe_params_p:
        # Initialize the LocalPAWFTCalculator with an empty gs adapter
        localft_calc = LocalFTCalculator.from_rshe_parameters(gs, context,
                                                              **rshe_params)

        # Compute the plane-wave components numerically
        n_G = localft_calc.engine.calculate(pd, nt_sR, [R_v], micro_setups,
                                            add_total_density)

        # Check validity of numerical results
        assert np.allclose(n_G, ntest_G, rtol=rtol)


@pytest.mark.response
def test_Fe_bxc(gpw_files):
    """Test the symmetry relation

    W_xc^z(G)^* = W_xc^z(-G)

    for a real life system with d-electrons (bcc-Fe)."""
    # ---------- Inputs ---------- #

    # Bxc calculation
    ecut = 100

    # ---------- Script ---------- #

    # Bxc calculation

    # Set up calculator and plane-wave descriptor
    calc = GPAW(gpw_files['fe_pw'], parallel=dict(domain=1))
    atoms = calc.atoms
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    localft_calc = LocalFTCalculator.from_rshe_parameters(gs, context)
    pd0 = get_pw_descriptor(atoms, calc, [0., 0., 0.],
                            ecut=ecut,
                            gammacentered=True)

    Wxc_G = localft_calc(pd0, add_LSDA_Wxc)

    # Part 3: Check symmetry relation
    G1_G, G2_G = get_inversion_pairs(pd0)

    assert np.allclose(np.conj(Wxc_G[G1_G]), Wxc_G[G2_G])


# ---------- Test functionality ---------- #


class EmptyGSAdapter(ResponseGroundStateAdapter):
    # Make an empty subclass to pass isinstance in constructor
    # In a future where the response code has been liberated from GPAW
    # calculator objects, the
    # >>> assert isinstance(gs, ResponseGroundStateAdapter)
    # statements can be deleted, making this class redundant.

    def __init__(self):
        pass


def get_inversion_pairs(pd0):
    """Get all pairs of G-indices which correspond to inverted reciprocal
    lattice vectors G and -G."""
    G_Gc = get_pw_coordinates(pd0)

    G1_G = []
    G2_G = []
    paired_indices = []
    for G1, G1_c in enumerate(G_Gc):
        if G1 in paired_indices:
            continue  # Already paired

        for G2, G2_c in enumerate(G_Gc):
            if np.all(G2_c == -G1_c):
                G1_G.append(G1)
                G2_G.append(G2)
                paired_indices += [G1, G2]
                break

    assert len(np.unique(paired_indices)) == len(G_Gc)

    return G1_G, G2_G
