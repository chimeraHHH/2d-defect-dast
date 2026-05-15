import numpy as np
import numpy.random as ra
import pytest
from gpaw.setup import create_setup
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LFC
from gpaw.spline import Spline
from gpaw.xc import XC
from gpaw.utilities import pack_density
from gpaw.mpi import serial_comm


def test_xc_gga_atom():
    rng = ra.default_rng(8)
    for name in ['LDA', 'PBE']:
        xc = XC(name)
        s = create_setup('N', xc)
        ni = s.ni
        nao = s.nao
        wt0_j = s.basis_functions_J

        rcut = s.xc_correction.rgd.r_g[-1]

        wt_j = []
        for wt0 in wt0_j:
            data = [wt0(r) for r in np.arange(121) * rcut / 100]
            data[-1] = 0.0
            l = wt0.get_angular_momentum_number()
            wt_j.append(Spline.from_data(l, 1.2 * rcut, data))

        a = rcut * 1.2 * 2 + 1.0
        n = 70
        n = 90
        gd = GridDescriptor((n, n, n), (a, a, a), comm=serial_comm)
        pr = LFC(gd, [wt_j])
        pr.set_positions([(0.5, 0.5, 0.5)])

        psit_ig = np.zeros((nao, n, n, n))
        pr.add(psit_ig, {0: np.eye(nao)})

        nii = ni * (ni + 1) // 2
        D_p = np.zeros(nii)

        e_g = np.zeros((n, n, n))
        n_g = np.zeros((1, n, n, n))
        v_g = np.zeros((1, n, n, n))

        P_ni = 0.2 * rng.random((20, ni))
        P_ni[:, nao:] = 0.0
        D_ii = np.dot(np.transpose(P_ni), P_ni)
        D_p = pack_density(D_ii)
        p = 0
        for i1 in range(nao):
            for i2 in range(i1, nao):
                n_g += D_p[p] * psit_ig[i1] * psit_ig[i2]
                p += 1
            p += ni - nao

        p = LFC(gd, [[s.nct]])
        p.set_positions([(0.5, 0.5, 0.5)])
        p.add(n_g[0], 1.0)
        e_g = gd.zeros()
        xc.calculate(gd, n_g, v_g, e_g)

        r2_g = np.sum((np.indices((n, n, n)) - n / 2)**2, axis=0)
        dv_g = gd.dv * np.less(r2_g, (rcut / a * n)**2)

        E2 = -np.dot(e_g.ravel(), dv_g.ravel())

        s.xc_correction.n_qg[:] = 0.0
        s.xc_correction.nc_g[:] = 0.0
        E1 = (xc.calculate_paw_correction(s, D_p.reshape(1, -1))
              + s.xc_correction.e_xc0)

        print(name, E1, E2, E1 - E2)
        assert E1 == pytest.approx(E2, abs=0.0013)
