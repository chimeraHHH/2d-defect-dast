import pytest


@pytest.mark.response
def test_pair_density_paw_correction():
    import numpy as np
    from gpaw.lfc import LocalizedFunctionsCollection as LFC
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
    from gpaw.spline import Spline
    from gpaw.response.paw import (calculate_pair_density_correction,
                                   LeanPAWDataset)
    # Initialize s, p, d (9 in total) wave and put them on grid
    rc = 2.0
    a = 2.5 * rc
    n = 64
    lmax = 2
    b = 8.0
    m = (lmax + 1)**2
    gd = GridDescriptor([n, n, n], [a, a, a])
    r = np.linspace(0, rc, 200)
    g = np.exp(-(r / rc * b)**2)
    splines = [Spline.from_data(l=l, rmax=rc, f_g=g) for l in range(lmax + 1)]
    c = LFC(gd, [splines])
    c.set_positions([(0.5, 0.5, 0.5)])
    psi = gd.zeros(m)
    d0 = c.dict(m)
    if 0 in d0:
        d0[0] = np.identity(m)
    c.add(psi, d0)

    # Calculate on 3d-grid < phi_i | e**(-ik.r) | phi_j >
    R_a = np.array([a / 2, a / 2, a / 2])
    rr = gd.get_grid_point_coordinates()
    for dim in range(3):
        rr[dim] -= R_a[dim]

    k_G = np.array([[0.0, 0.0, 0.0], [1., 0.2, 0.1], [10., 0., 10.]])
    nkpt = k_G.shape[0]

    d0 = np.zeros((nkpt, m, m), dtype=complex)
    for i in range(m):
        for j in range(m):
            for ik in range(nkpt):
                k = k_G[ik]
                # kk = np.sqrt(np.inner(k, k))
                kr = np.inner(k, rr.T).T
                expkr = np.exp(-1j * kr)
                d0[ik, i, j] = gd.integrate(psi[i] * psi[j] * expkr)

    # Calculate on 1d-grid < phi_i | e**(-ik.r) | phi_j >
    rgd = EquidistantRadialGridDescriptor(r[1], len(r))
    g = [np.exp(-(r / rc * b)**2) * r**l for l in range(lmax + 1)]
    l_j = range(lmax + 1)
    rcut_j = [rc] * (lmax + 1)
    d1 = calculate_pair_density_correction(k_G, pawdata=LeanPAWDataset(
        rgd=rgd, phi_jg=g, phit_jg=np.zeros_like(g), l_j=l_j, rcut_j=rcut_j))

    assert d0 == pytest.approx(d1, abs=1e-8)
