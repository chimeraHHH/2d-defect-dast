import pytest
from gpaw.atom.aeatom import AllElectronAtom, c


@pytest.mark.ci
@pytest.mark.serial
def test_aeatom():
    Z = 79  # gold atom
    kwargs = dict(ngpts=5000, alpha2=1000 * Z**2, ngauss=200)

    # Test Schroedinger equation:
    aea = AllElectronAtom(Z, log=None)
    aea.initialize(**kwargs)

    errors = []
    for channel in aea.channels:
        vr_g = channel.basis.rgd.empty()
        vr_g[:] = -Z
        # Basis set of Gaussians:
        channel.solve(vr_g)
        for n in range(7):
            e = channel.e_n[n]
            e0 = -0.5 * Z**2 / (n + channel.l + 1)**2
            errors.append(abs(e / e0 - 1))

        # Finite-difference:
        channel.solve2(vr_g, scalar_relativistic=False, Z=Z)
        for n in range(7):
            e = channel.e_n[n]
            e0 = -0.5 * Z**2 / (n + channel.l + 1)**2
            errors.append(abs(e / e0 - 1))

    print(max(errors))
    assert max(errors) == pytest.approx(0, abs=2.0e-5)

    # Test Dirac equation:
    aea = AllElectronAtom(Z, dirac=True, log=None)
    aea.initialize(**kwargs)

    errors = []
    for channel in aea.channels:
        vr_g = channel.basis.rgd.empty()
        vr_g[:] = -Z
        channel.solve(vr_g)
        for n in range(7):
            e = channel.e_n[n]
            if channel.k > 0:
                n += 1
            e0 = (1 +
                  (Z / c)**2 /
                  ((channel.k**2 - (Z / c)**2)**0.5 + n)**2)**-0.5 - 1
            e0 *= c**2
            errors.append(abs(e / e0 - 1))

    print(max(errors))
    assert max(errors) == pytest.approx(0, abs=4.0e-5)
