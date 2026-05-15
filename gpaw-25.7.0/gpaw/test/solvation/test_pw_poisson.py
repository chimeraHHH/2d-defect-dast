import numpy as np
import pytest
import time
from ase.units import Bohr
from gpaw.core import PWDesc, UGDesc, PWArray
from gpaw.new.pw.poisson import ConjugateGradientPoissonSolver
from ase.parallel import parprint

nn = 3
accuracy = 2e-10
BOX = 12.0


def generate_grid(box, h=0.4, pbc=[True, True, True], dtype=float):
    """
    Create a grid for testing.

    Parameters:
    ----------
    box : float
        Box size in Ångström
    h : float
        Grid spacing in Ångström
    pbc : list
        Periodic boundary conditions
    dtype : type
        Data type for the grid (float or complex)

    Returns:
    -------
    grid : UGDesc
        Grid descriptor
    """
    diag = np.array([box] * 3)
    cell = np.diag(diag)
    grid_shape = tuple((diag / h * 2).astype(int))

    return UGDesc(cell=cell / Bohr, size=grid_shape, pbc=pbc, dtype=dtype)


def generate_dipole_charges(grid, pw, box):
    """
    Generate a dipole charge distribution (positive and negative charges).

    Parameters:
    -----------
    grid : UGDesc
        Grid descriptor
    pw : PWDesc
        Plane wave descriptor
    box : float
        Box size

    Returns:
    ---------
    rho : PWArray
        Charge density for testing
    """
    xyz = grid.xyz()
    x = xyz[:, :, :, 0]
    y = xyz[:, :, :, 1]
    z = xyz[:, :, :, 2]
    center = box / (2 * Bohr)

    radius = 1.0 / Bohr
    pos_center = np.array([center - 2.0, center, center])
    neg_center = np.array([center + 2.0, center, center])

    r_pos = np.sqrt(
        (x - pos_center[0])**2 +
        (y - pos_center[1])**2 +
        (z - pos_center[2])**2)
    r_neg = np.sqrt(
        (x - neg_center[0])**2 +
        (y - neg_center[1])**2 +
        (z - neg_center[2])**2)

    shape = tuple(grid.size_c)
    charge_array = np.zeros(shape)

    sigma = 0.5 / Bohr
    pos_charge = 0.5 * (1.0 - np.tanh((r_pos - radius) / sigma))
    neg_charge = -0.5 * (1.0 - np.tanh((r_neg - radius) / sigma))

    charge_array = pos_charge + neg_charge

    total_charge = np.sum(charge_array) * grid.dv
    charge_array -= total_charge / (grid.dv * np.prod(shape))

    rho_r = grid.zeros()
    rho_r.data[:] = charge_array

    rho = rho_r.fft(pw=pw)

    return rho


def generate_quadrupole_charges(grid, pw, box, seed=None):
    """
    Generate a quadrupole-like charge distribution with four charges
    arranged in a tetrahedral pattern.

    Parameters:
    -----------
    grid : UGDesc
        Grid descriptor
    pw : PWDesc
        Plane wave descriptor
    box : float
        Box size
    seed : int, optional
        Seed for random number generator

    Returns:
    ---------
    rho : PWArray
        Charge density for testing
    """
    if seed is not None:
        np.random.seed(seed)

    xyz = grid.xyz()
    x = xyz[:, :, :, 0]
    y = xyz[:, :, :, 1]
    z = xyz[:, :, :, 2]
    center = box / (2 * Bohr)

    radius = box / (4 * Bohr)
    charge_radius = 1.0 / Bohr

    vertices = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / np.sqrt(3)

    centers = vertices * radius
    centers += np.array([center, center, center])

    charges = [1.0, -1.0, 1.0, -1.0]

    shape = tuple(grid.size_c)
    charge_array = np.zeros(shape)
    sigma = 0.5 / Bohr

    for q, pos in zip(charges, centers):
        r = np.sqrt((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2)
        charge = q * 0.5 * (1.0 - np.tanh((r - charge_radius) / sigma))
        charge_array += charge

    total_charge = np.sum(charge_array) * grid.dv
    charge_array -= total_charge / (grid.dv * np.prod(shape))

    rho_r = grid.zeros()
    rho_r.data[:] = charge_array

    rho = rho_r.fft(pw=pw)

    return rho


def spherical_dielectric_function(grid, box, epsinf=80.0, eps1=1.0,
                                  radius=None):
    """
    Generate a spherical dielectric function with high dielectric inside
    a sphere and low dielectric outside, with a smooth transition at the
    boundary.

    Parameters:
    ----------
    grid : UGDesc
        Grid descriptor
    box : float
        Box size in Ångström
    epsinf : float
        Dielectric constant inside the sphere (typically water = 80)
    eps1 : float
        Dielectric constant outside the sphere (typically vacuum = 1)
    radius : float, optional
        Radius of the dielectric sphere in Ångström. Defaults to 1/3
        of box size.

    Returns:
    --------
    eps : UGArray
        Dielectric function for testing
    eps_gradeps : list
        List containing eps and its gradient as UGArray objects
    """
    xyz = grid.xyz()
    x = xyz[:, :, :, 0]
    y = xyz[:, :, :, 1]
    z = xyz[:, :, :, 2]
    center = box / (2 * Bohr)

    if radius is None:
        radius = box / 3.0
    radius_bohr = radius / Bohr

    r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)

    width = 0.5 / Bohr
    transition = 0.5 * (1.0 - np.tanh((r - radius_bohr) / width))

    eps_np = eps1 + (epsinf - eps1) * transition

    deps_dr_np = -(epsinf - eps1) * 0.5 * (
        1.0 - np.tanh((r - radius_bohr) / width)**2) / width

    grad_x_np = np.zeros_like(r)
    grad_y_np = np.zeros_like(r)
    grad_z_np = np.zeros_like(r)

    mask = r > 1e-10
    grad_x_np[mask] = deps_dr_np[mask] * (x[mask] - center) / r[mask]
    grad_y_np[mask] = deps_dr_np[mask] * (y[mask] - center) / r[mask]
    grad_z_np[mask] = deps_dr_np[mask] * (z[mask] - center) / r[mask]

    eps = grid.zeros()
    eps.data[:] = eps_np

    grad_x = grid.zeros()
    grad_x.data[:] = grad_x_np

    grad_y = grid.zeros()
    grad_y.data[:] = grad_y_np

    grad_z = grid.zeros()
    grad_z.data[:] = grad_z_np

    eps_gradeps = [eps, grad_x, grad_y, grad_z]

    return eps, eps_gradeps


@pytest.mark.parametrize("density_generator", [
    generate_dipole_charges,
    lambda grid, pw, box: generate_quadrupole_charges(grid, pw, box, seed=0),
    lambda grid, pw, box: generate_quadrupole_charges(grid, pw, box, seed=1),
])
def test_cg_poisson_solver_constant_dielectric(density_generator):
    """Test the conjugate gradient Poisson solver with constant dielectric."""
    box = BOX
    grid = generate_grid(box, dtype=complex)

    pw = PWDesc(ecut=20.0, cell=grid.cell, kpt=grid.kpt, comm=grid.comm,
                dtype=complex)

    epsinf = 80.0

    shape = tuple(grid.size_c)
    eps_np = np.ones(shape) * epsinf

    eps = grid.zeros()
    eps.data[:] = eps_np

    grad_zeros_np = np.zeros(shape)

    grad_x = grid.zeros()
    grad_x.data[:] = grad_zeros_np
    grad_y = grid.zeros()
    grad_y.data[:] = grad_zeros_np
    grad_z = grid.zeros()
    grad_z.data[:] = grad_zeros_np

    eps_gradeps = [eps, grad_x, grad_y, grad_z]

    rho = density_generator(grid, pw, box)

    class MockDielectric:
        def __init__(self, eps_gradeps):
            self.eps_gradeps = [e.data for e in eps_gradeps]

    solver = ConjugateGradientPoissonSolver(
        pw, grid, MockDielectric(eps_gradeps),
        eps=1e-6, maxiter=100)

    phi = PWArray(pw=pw)
    start = time.time()
    solver.solve(phi, rho)
    end = time.time()
    parprint(f'CG Solver timing: {end - start}')

    laplacian_phi = PWArray(pw=pw)
    G2_q = pw.ekin_G.copy()
    laplacian_phi.data[:] = G2_q * phi.data.copy()

    expected_laplacian = PWArray(pw=pw)
    factor = 2 * np.pi / float(epsinf)
    expected_laplacian.data[:] = factor * rho.data.copy()

    laplacian_phi_r = laplacian_phi.ifft(grid=grid).data
    expected_laplacian_r = expected_laplacian.ifft(grid=grid).data

    assert laplacian_phi_r == pytest.approx(expected_laplacian_r, abs=1e-3)


@pytest.mark.parametrize("density_generator", [
    generate_dipole_charges,
    lambda grid, pw, box: generate_quadrupole_charges(grid, pw, box, seed=0),
])
def test_cg_poisson_solver_variable_dielectric(density_generator):
    """Test the conjugate gradient Poisson solver with variable dielectric."""
    box = BOX
    grid = generate_grid(box, dtype=complex)

    pw = PWDesc(ecut=20.0, cell=grid.cell, kpt=grid.kpt, comm=grid.comm,
                dtype=complex)

    eps, eps_gradeps = spherical_dielectric_function(grid, box,
                                                     epsinf=80.0, eps1=1.0)

    rho = density_generator(grid, pw, box)

    class MockDielectric:
        def __init__(self, eps_gradeps):
            self.eps_gradeps = [e.data for e in eps_gradeps]

    solver = ConjugateGradientPoissonSolver(
        pw=pw, grid=grid,
        dielectric=MockDielectric(eps_gradeps),
        eps=1e-6, maxiter=100)

    phi = PWArray(pw=pw)
    start = time.time()
    solver.solve(phi, rho)
    end = time.time()
    parprint(f'CG Solver timing: {end - start}')

    phi_operator = solver.operator(phi.data.copy())

    expected = rho.data.copy()
    expected *= 4 * np.pi

    assert np.allclose(phi_operator, expected, atol=1e-3)
