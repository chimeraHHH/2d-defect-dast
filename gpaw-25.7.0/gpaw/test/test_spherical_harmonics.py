from math import pi

import numpy as np
import pytest
from gpaw.spherical_harmonics import (YL, Y, Yl, gam, print_YL_table_code,
                                      write_c_code)


def yLL(L1, L2):
    s = 0.0
    for c1, n1 in YL[L1]:
        for c2, n2 in YL[L2]:
            s += c1 * c2 * gam(n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2])
    return s


def test_yy():
    Lmax = len(YL)
    for L1 in range(Lmax):
        for L2 in range(Lmax):
            r = 0.0
            if L1 == L2:
                r = 1.0
            assert yLL(L1, L2) == pytest.approx(r, abs=1e-14)


def test_y_c_code():
    R = np.zeros(3)
    Y = np.zeros(1)
    Yl(0, R, Y)
    assert Y[0] == pytest.approx((4 * pi)**-0.5)
    Y = np.zeros(2 * 8 + 1)
    with pytest.raises(RuntimeError):
        Yl(8, R, Y)


def test_y_c_code2():
    R_c = np.array([0.1, -0.2, 0.3])
    for l in range(8):
        print('l', l)

        # Using C implementation
        rlY1_m = np.zeros(2 * l + 1) + np.nan
        Yl(l, R_c, rlY1_m)
        print(rlY1_m)

        # Using Python implementation
        rlY2_m = []
        for m in range(-l, l + 1):
            L = l**2 + l + m
            rlY2_m.append(Y(L, *R_c))
        rlY2_m = np.array(rlY2_m)
        print(rlY2_m)

        assert np.allclose(rlY2_m, rlY1_m)


def test_write_c_code(capsys):
    write_c_code(1)
    s = capsys.readouterr().out
    s = s.replace(' ', '')
    s = s.replace('\n', '')
    assert s == (
        'elseif(l==1)' +
        '{Y_m[0]=0.4886025119029199*y;' +
        'Y_m[1]=0.4886025119029199*z;' +
        'Y_m[2]=0.4886025119029199*x;}')


def test_print_YL_table_code():
    pytest.importorskip('sympy')
    print_YL_table_code()
