import warnings

import numpy as np

from ase.units import Bohr, Hartree

from gpaw.fftw import get_efficient_fft_size
from gpaw.utilities import h2gpts
from gpaw.wavefunctions.fd import FD


def get_number_of_grid_points(cell_cv,
                              h=None,
                              mode=None,
                              realspace=None,
                              symmetry=None):
    if mode is None:
        mode = FD()

    if realspace is None:
        realspace = mode.name != 'pw'

    if h is None:
        if mode.name == 'pw':
            h = np.pi / (4 * mode.ecut)**0.5
        elif mode.name == 'lcao' and not realspace:
            h = np.pi / (4 * 340 / Hartree)**0.5
        else:
            h = 0.2 / Bohr

    if realspace or mode.name == 'fd':
        N_c = h2gpts(h, cell_cv, 4)
    else:
        N_c = np.ceil((cell_cv**2).sum(1)**0.5 / h).astype(int)
        if symmetry is None:
            N_c = np.array([get_efficient_fft_size(N) for N in N_c])
        else:
            N_c = np.array([get_efficient_fft_size(N, n)
                            for N, n in zip(N_c, symmetry.gcd_c)])

    if symmetry is not None:
        ok = symmetry.check_grid(N_c)
        if not ok:
            warnings.warn(
                'Initial realspace grid '
                '({},{},{}) inconsistent with symmetries.'.format(*N_c))
            # The grid is not symmetric enough. The essential problem
            # is that we should start at some other Nmin_c and possibly with
            # other gcd_c when getting the most efficient fft grids
            gcd_c = symmetry.gcd_c.copy()
            Nmin_c = N_c.copy()
            for i in range(3):
                for op_cc in symmetry.op_scc:
                    for i, o in enumerate((op_cc.T).flat):
                        i1, i2 = np.unravel_index(i, (3, 3))
                        if i1 == i2:
                            continue
                        if o:
                            # The axes are related and therefore they share
                            # lowest common multiple of gcd
                            gcd = np.lcm.reduce(gcd_c[[i1, i2]])
                            gcd_c[[i1, i2]] = gcd
                            # We just take the maximum of the two axes to make
                            # sure that they are divisible always
                            Nmin = np.max([Nmin_c[i1], Nmin_c[i2]])
                            Nmin_c[i1] = Nmin
                            Nmin_c[i2] = Nmin

            N_c = np.array([get_efficient_fft_size(N, n)
                            for N, n in zip(Nmin_c, gcd_c)])
            warnings.warn('Using symmetrized grid: ({},{},{}).\n'.format(*N_c))
            ok = symmetry.check_grid(N_c)
            assert ok, ('Grid still not constistent with symmetries '
                        '({},{},{})'.format(*N_c))

    return N_c
