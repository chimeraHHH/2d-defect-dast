import pytest
import numpy as np
import numpy.random as ra
from gpaw.setup import create_setup
from gpaw.xc import XC
from gpaw.test import gen


@pytest.mark.libxc
def test_xc_lxc_xcatom(in_tmp_dir):
    setups = {}
    for functional in [
        'LDA_X', 'LDA_X+LDA_C_PW', 'LDA_X+LDA_C_VWN', 'LDA_X+LDA_C_PZ',
        'GGA_X_PBE+GGA_C_PBE', 'GGA_X_PBE_R+GGA_C_PBE',
        'GGA_X_B88+GGA_C_P86', 'GGA_X_B88+GGA_C_LYP',
        'GGA_X_FT97_A+GGA_C_LYP']:
        s = gen('N', xcname=functional)
        setups[functional] = s

    tolerance = 0.000005  # libxc must reproduce old gpaw energies
    # zero Kelvin: in Hartree

    reference = {  # version 0.9.1
        'LDA_X+LDA_C_PW': 2.28836113207,  # 'LDA'
        'GGA_X_PBE+GGA_C_PBE': 2.3366049993,  # 'PBE'
        'GGA_X_PBE_R+GGA_C_PBE': 2.34496288319}  # 'revPBE'

    reference_libxc = {  # svnversion 5252
        'LDA_X': 1.95030600807,
        'LDA_X+LDA_C_PW': 2.23194461135,
        'LDA_X+LDA_C_VWN': 2.23297429824,
        'LDA_X+LDA_C_PZ': 2.23146045547,
        'GGA_X_PBE+GGA_C_PBE': 2.28208665019,
        'GGA_X_PBE_R+GGA_C_PBE': 2.29201920843,
        'GGA_X_B88+GGA_C_P86': 2.30508027546,
        'GGA_X_B88+GGA_C_LYP': 2.28183010548,
        'GGA_X_FT97_A+GGA_C_LYP': 2.26846048873}

    libxc_set = [
        'LDA_X', 'LDA_X+LDA_C_PW', 'LDA_X+LDA_C_VWN', 'LDA_X+LDA_C_PZ',
        'GGA_X_PBE+GGA_C_PBE', 'GGA_X_PBE_R+GGA_C_PBE',
        'GGA_X_B88+GGA_C_P86', 'GGA_X_B88+GGA_C_LYP',
        'GGA_X_FT97_A+GGA_C_LYP']

    x = 0.000001
    for xcname in libxc_set:
        # note: using ra.default_rng() not compatible with legacy results
        rng = ra.RandomState(8)
        xc = XC(xcname)
        s = create_setup('N', xc, setupdata=setups[xcname])
        ni = s.ni
        nii = ni * (ni + 1) // 2
        D_p = 0.1 * rng.random(nii) + 0.4
        H_p = np.zeros(nii)

        E1 = xc.calculate_paw_correction(s,
                                         D_p.reshape(1, -1),
                                         H_p.reshape(1, -1))
        dD_p = x * rng.random(nii)
        D_p += dD_p
        dE = np.dot(H_p, dD_p) / x
        E2 = xc.calculate_paw_correction(s, D_p.reshape(1, -1))
        print(xcname, dE, (E2 - E1) / x)
        assert dE == pytest.approx((E2 - E1) / x, abs=0.003)

        E2s = xc.calculate_paw_correction(
            s,
            np.array([0.5 * D_p, 0.5 * D_p]),
            np.array([H_p, H_p]))
        print(E2, E2s)
        assert E2 == pytest.approx(E2s, abs=1.0e-12)

        if xcname in reference:  # compare with old gpaw
            print('A:', E2, reference[xcname])
            assert E2 == pytest.approx(reference[xcname], abs=tolerance)

        if xc in reference_libxc:  # compare with reference libxc
            print('B:', E2, reference_libxc[xcname])
            assert E2 == pytest.approx(reference_libxc[xcname], abs=tolerance)

        D_sp = 0.1 * rng.random((2, nii)) + 0.2
        H_sp = np.zeros((2, nii))

        E1 = xc.calculate_paw_correction(s, D_sp, H_sp)
        dD_sp = x * rng.random((2, nii))
        D_sp += dD_sp
        dE = np.dot(H_sp.ravel(), dD_sp.ravel()) / x
        E2 = xc.calculate_paw_correction(s, D_sp, H_sp)
        print(dE, (E2 - E1) / x)
        assert dE == pytest.approx((E2 - E1) / x, abs=0.005)
