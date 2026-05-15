import pytest

import numpy as np
from itertools import permutations

from gpaw.spherical_harmonics import Y
from gpaw.gaunt import gaunt, super_gaunt


def test_contraction_rule(lmax: int = 3):
    """Test that two spherical harmonics can be contracted to one."""
    G_LLL = gaunt(lmax)
    L1max, L2max, L3max = G_LLL.shape
    assert L2max == L3max

    x, y, z = unit_sphere_test_coordinates()
    for L1 in range(L1max):
        # In order to include all finite Gaunt coefficients in the expansion,
        # l1 + l2 has to be within the range of available L3 coefficients.
        l1 = int(np.sqrt(L1))
        l3max = int(np.sqrt(L3max)) - 1
        l2max = l3max - l1
        L2max = (l2max + 1)**2
        for L2 in range(L2max):
            product = Y(L1, x, y, z) * Y(L2, x, y, z)
            expansion = 0.
            for L3 in range(L3max):
                expansion += G_LLL[L1, L2, L3] * Y(L3, x, y, z)
            assert expansion == pytest.approx(product)


def test_selection_rules(lmax: int = 3):
    """Test Gaunt coefficient selection rules."""
    G_LLL = gaunt(lmax)
    L1max, L2max, L3max = G_LLL.shape

    for L1 in range(L1max):
        # Convert composit index to (l,m)
        l1, m1 = lm_indices(L1)
        for L2 in range(L2max):
            l2, m2 = lm_indices(L2)
            for L3 in range(L3max):
                l3, m3 = lm_indices(L3)
                # In order for the Gaunt coefficients to be finite, the
                # following conditions should be met
                if abs(G_LLL[L1, L2, L3]) > 1e-10:
                    assert (l1 + l2 + l3) % 2 == 0, \
                        f'l1+l2+l3 = {l1 + l2 + l3} should be an even integer'
                    assert abs(l1 - l2) <= l3 <= l1 + l2, f'{l1, l2, l3}'
                    assert m1 + m2 + m3 == 0 or \
                        m1 + m2 - m3 == 0 or \
                        m3 + m1 - m2 == 0 or \
                        m2 + m3 - m1 == 0, f'{m1, m2, m3}'


def test_permutation_symmetry(lmax: int = 3):
    """Test that the Gaunt coefficients are permutationally invariant"""
    G_LLL = gaunt(lmax)
    L1max, L2max, L3max = G_LLL.shape
    assert L2max == L3max

    for L1 in range(L1max):
        # Permutations between all indices
        for L2 in range(L1max):
            for L3 in range(L1max):
                for Lp1, Lp2, Lp3 in permutations([L1, L2, L3]):
                    assert abs(G_LLL[L1, L2, L3]
                               - G_LLL[Lp1, Lp2, Lp3]) < 1e-10
        # Permutations between L2 and L3
        for L2 in range(L2max):
            for L3 in range(L3max):
                assert abs(G_LLL[L1, L2, L3] - G_LLL[L1, L3, L2]) < 1e-10


def test_super_contraction_rule(lmax: int = 2):
    """Test that three spherical harmonics can be contracted to one."""
    G_LLLL = super_gaunt(lmax)
    L1max, L2max, L3max, L4max = G_LLLL.shape
    assert L1max == L2max

    x, y, z = unit_sphere_test_coordinates()
    for L1 in range(L1max):
        for L2 in range(L2max):
            for L3 in range(L3max):
                product = Y(L1, x, y, z) * Y(L2, x, y, z) * Y(L3, x, y, z)
                expansion = 0.
                for L4 in range(L4max):
                    expansion += G_LLLL[L1, L2, L3, L4] * Y(L4, x, y, z)
                assert expansion == pytest.approx(product)


def test_super_selection_rules(lmax: int = 2):
    """Test selection rules for products of Gaunt coefficients."""
    G_LLLL = super_gaunt(lmax)
    L1max, L2max, L3max, L4max = G_LLLL.shape

    for L1 in range(L1max):
        # Convert composit index to (l,m)
        l1, m1 = lm_indices(L1)
        for L2 in range(L2max):
            l2, m2 = lm_indices(L2)
            for L3 in range(L3max):
                l3, m3 = lm_indices(L3)
                for L4 in range(L4max):
                    l4, m4 = lm_indices(L4)
                    # In order for the Gaunt coefficients to be finite, the
                    # following conditions should be met
                    if abs(G_LLLL[L1, L2, L3, L4]) > 1e-10:
                        assert (l1 + l2 + l3 + l4) % 2 == 0, \
                            f'l1 + l2 + l3 + l4 = {l1 + l2 + l3 + l4} ' \
                            'should be an even integer'
                        # The allowed l' range governs the allowed l4 range
                        lp_range = np.arange(abs(l1 - l2), l1 + l2 + 1)
                        assert np.min(abs(lp_range - l3)) <= l4 <= l1 + l2 + l3
                        # m' needs to be allowed and repeated
                        mpset1 = {-m1 - m2, m1 + m2, -m1 + m2, m1 - m2}
                        mpset2 = {-m3 - m4, m3 + m4, -m3 + m4, m3 - m4}
                        assert len(mpset1 | mpset2) < len(mpset1) + len(mpset2)


def test_super_permutation_symmetry(lmax: int = 2):
    """Test that the super Gaunt coefficients are permutationally invariant"""
    G_LLLL = super_gaunt(lmax)
    L1max, L2max, L3max, L4max = G_LLLL.shape
    assert L1max == L2max

    for L1 in range(L1max):
        for L2 in range(L1max):
            # Permutations between all indices
            for L3 in range(L1max):
                for L4 in range(L1max):
                    for Lp1, Lp2, Lp3, Lp4 in permutations([L1, L2, L3, L4]):
                        assert abs(G_LLLL[L1, L2, L3, L4]
                                   - G_LLLL[Lp1, Lp2, Lp3, Lp4]) < 1e-10
            # Permutations between L3 and L4
            for L3 in range(L3max):
                for L4 in range(L3max):
                    assert abs(G_LLLL[L1, L2, L3, L4]
                               - G_LLLL[L1, L2, L4, L3]) < 1e-10


def unit_sphere_test_coordinates():
    """Unit-sphere coordinates to test"""
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 6),
                             np.linspace(0, 2 * np.pi, 11),
                             indexing='ij')
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def lm_indices(L):
    l = int(np.sqrt(L))
    m = L - l * (l + 1)
    return l, m
