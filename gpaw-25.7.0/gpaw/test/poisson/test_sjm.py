from types import SimpleNamespace
import numpy as np
from gpaw.core import PWDesc, UGDesc
from gpaw.new.sjm import SJMPWPoissonSolver
from gpaw.new.pw.poisson import ConjugateGradientPoissonSolver


def f(a, z, z0, w):
    return np.exp(-((z - z0) / w)**2) / a**2 / np.pi**0.5 / w


if 0:  # Analytic result
    from sympy import Symbol, exp, integrate, oo, var
    z = var('z')
    w = Symbol('w', positive=True)
    m = integrate(exp(-(z / w)**2), (z, -oo, oo))
    print(m)  # sqrt(pi)*w


def test_sjm():
    a = 1.0
    L = 10.0
    grid = UGDesc.from_cell_and_grid_spacing(cell=[a, a, L], grid_spacing=0.15)
    z = grid.xyz()[0, 0, :, 2]
    c = 0.05
    rhot = f(a, z, 4.0, 1.0) * (1 + c)  # electrons
    rhot -= f(a, z, 4.0, 0.5)  # nucleui
    rhot -= f(a, z, 6.0, 1.0) * c  # jellium
    eps = 1.0 + f(a, z, 5.0, 2.0) * 20  # dielectric
    rhot_r = grid.zeros()
    rhot_r.data[:] = rhot
    eps_r = grid.zeros()
    eps_r.data[:] = eps
    print(rhot_r.integrate())
    pw = PWDesc(ecut=grid.ekin_max(), cell=grid.cell)
    ps = SJMPWPoissonSolver(pw, dielectric=None)
    vt_g = pw.zeros()
    rhot_g = rhot_r.fft(pw=pw)
    ps.solve(vt_g, rhot_g)
    vt_r = vt_g.ifft(grid=grid)
    dielectric = SimpleNamespace(eps_gradeps=[eps_r.data])
    ps2 = ConjugateGradientPoissonSolver(pw, grid, dielectric,
                                         zero_vacuum=True)
    vt2_g = pw.zeros()
    ps2.solve(vt2_g, rhot_g)
    vt2_r = vt2_g.ifft(grid=grid)
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(z, rhot_r.data[0, 0])
        plt.plot(z, vt_r.data[0, 0])
        plt.plot(z, vt2_r.data[0, 0])
        plt.show()


if __name__ == '__main__':
    test_sjm()
