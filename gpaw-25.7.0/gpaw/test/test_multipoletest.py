from math import sqrt, pi

import numpy as np
import pytest

from gpaw.setup import create_setup
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LFC
from gpaw.xc import XC

n = 60  # 40 /8 * 10
a = 10.0


@pytest.mark.ci
def test_multipole():
    gd = GridDescriptor((n, n, n), (a, a, a))
    c_LL = np.identity(9, float)
    a_Lg = gd.zeros(9)
    xc = XC('LDA')
    for soft in [False]:
        s = create_setup('Cu', xc, lmax=2)
        ghat_l = s.ghat_l
        ghat_Lg = LFC(gd, [ghat_l])
        ghat_Lg.set_positions([(0.54321, 0.5432, 0.543)])
        a_Lg[:] = 0.0
        ghat_Lg.add(a_Lg, {0: c_LL} if ghat_Lg.my_atom_indices else {})
        for l in range(3):
            for m in range(2 * l + 1):
                L = l**2 + m
                a_g = a_Lg[L]
                Q0 = gd.integrate(a_g) / sqrt(4 * pi)
                Q1_m = -gd.calculate_dipole_moment(a_g) / sqrt(4 * pi / 3)
                print(Q0)
                if l == 0:
                    Q0 -= 1.0
                    Q1_m[:] = 0.0
                elif l == 1:
                    Q1_m[(m + 1) % 3] -= 1.0
                print(soft, l, m, Q0, Q1_m)
                assert abs(Q0) < 2e-6
                assert (abs(Q1_m) < 3e-5).all()
        b_Lg = np.reshape(a_Lg, (9, -1))
        S_LL = np.inner(b_Lg, b_Lg)
        gd.comm.sum(S_LL)
        S_LL.ravel()[::10] = 0.0
        print(max(abs(S_LL).ravel()))
        assert max(abs(S_LL).ravel()) < 3e-4
