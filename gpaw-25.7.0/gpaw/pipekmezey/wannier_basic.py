""" Maximally localized Wannier Functions

    Find the set of maximally localized Wannier functions
    using the spread functional of Marzari and Vanderbilt
    (PRB 56, 1997 page 12847).

    this code is as in ASE but modified to use it with gpaw's wfs.
"""

from time import time
from math import pi
import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.dft.wannier import calculate_weights, gram_schmidt
from ase.transport.tools import dagger
from ase.parallel import parprint

dag = dagger


def random_orthogonal_matrix(dim, rng, real=False):
    """Generate a random orthogonal matrix"""

    H = rng.random((dim, dim))
    np.add(dag(H), H, H)
    np.multiply(.5, H, H)

    if real:
        gram_schmidt(H)
        return H
    else:
        val, vec = np.linalg.eig(H)
        return np.dot(vec * np.exp(1.j * val), dag(vec))


def md_min(func, step=.25, tolerance=1e-6, verbose=False, **kwargs):
    if verbose:
        parprint('Localize with step =', step,
                 'and tolerance =', tolerance)
    t = -time()
    fvalueold = 0.
    fvalue = fvalueold + 10
    count = 0
    V = np.zeros(func.get_gradients().shape, dtype=complex)

    while abs((fvalue - fvalueold) / fvalue) > tolerance:
        fvalueold = fvalue
        dF = func.get_gradients()
        V *= (dF * V.conj()).real > 0
        V += step * dF
        func.step(V, **kwargs)
        fvalue = func.get_function_value()

        if fvalue < fvalueold:
            step *= 0.5
        count += 1
        func.niter = count

        if verbose:
            parprint('MDmin: iter=%s, step=%s, value=%s'
                     % (count, step, fvalue))
    t += time()
    if verbose:
        parprint('%d iterations in %0.2f seconds(%0.2f ms/iter),'
                 ' endstep = %s'
                 % (count, t, t * 1000. / count, step))


def get_atoms_object_from_wfs(wfs):
    from ase.units import Bohr
    from ase import Atoms

    spos_ac = wfs.spos_ac
    cell_cv = wfs.gd.cell_cv
    positions = spos_ac * cell_cv.diagonal() * Bohr

    string = ''
    for a, atoms in enumerate(wfs.setups):
        string += atoms.symbol

    atoms = Atoms(string)
    atoms.positions = positions
    atoms.cell = cell_cv * Bohr

    return atoms


class WannierLocalization:
    """Maximally localized Wannier Functions
       for n_occ only - for ODD calculations
    """

    def __init__(self, wfs, calc=None, spin=0, seed=None, verbose=False):
        from ase.dft.wannier import get_kklst, get_invkklst

        # Bloch phase sign convention
        sign = -1
        self.wfs = wfs
        self.gd = self.wfs.gd
        self.ns = self.wfs.nspins
        self.dtype = wfs.dtype

        if hasattr(self.wfs, 'mode'):
            self.mode = self.wfs.mode
        else:
            self.mode = None

        if calc is not None:
            self.atoms = calc.atoms
        else:
            self.atoms = get_atoms_object_from_wfs(self.wfs)

        # Determine nocc: integer occupations only
        k_rank, u = divmod(0 + len(self.wfs.kd.ibzk_kc) * spin,
                           len(self.wfs.kpt_u))

        f_n = self.wfs.kpt_u[u].f_n
        self.nwannier = int(np.rint(f_n.sum()) /
                            (3 - self.ns))  # No fractional occ

        self.spin = spin
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.kpt_kc = self.wfs.kd.bzk_kc
        assert len(self.wfs.kd.ibzk_kc) == len(self.kpt_kc)

        self.kptgrid = \
            get_monkhorst_pack_size_and_offset(self.kpt_kc)[0]
        self.kpt_kc *= sign

        self.Nk = len(self.kpt_kc)
        self.unitcell_cc = self.atoms.get_cell()
        self.largeunitcell_cc = (self.unitcell_cc.T * self.kptgrid).T
        self.weight_d, self.Gdir_dc = \
            calculate_weights(self.largeunitcell_cc)
        self.Ndir = len(self.weight_d)  # Number of directions

        # Get neighbor kpt list and inverse kpt list
        self.kklst_dk, k0_dkc = get_kklst(self.kpt_kc, self.Gdir_dc)
        self.invkklst_dk = get_invkklst(self.kklst_dk)

        Nw = self.nwannier
        Z_dknn = np.zeros((self.Ndir, self.Nk, Nw, Nw),
                          dtype=complex)
        self.Z_dkww = np.empty((self.Ndir, self.Nk, Nw, Nw),
                               dtype=complex)

        if self.mode == 'lcao' and self.wfs.kpt_u[0].psit_nG is None:
            self.wfs.initialize_wave_functions_from_lcao()

        for d, dirG in enumerate(self.Gdir_dc):
            for k in range(self.Nk):
                k1 = self.kklst_dk[d, k]
                k0_c = k0_dkc[d, k]
                k_kc = self.wfs.kd.bzk_kc
                Gc = k_kc[k1] - k_kc[k] - k0_c
                # Det. kpt/spin
                kr, u = divmod(k + len(self.wfs.kd.ibzk_kc) * spin,
                               len(self.wfs.kpt_u))
                kr1, u1 = divmod(k1 + len(self.wfs.kd.ibzk_kc) * spin,
                                 len(self.wfs.kpt_u))

                if self.wfs.mode == 'pw':
                    cmo = self.gd.zeros(Nw, dtype=self.wfs.dtype)
                    cmo1 = self.gd.zeros(Nw, dtype=self.wfs.dtype)
                    for i in range(Nw):
                        cmo[i] = self.wfs._get_wave_function_array(u, i)
                        cmo1[i] = self.wfs._get_wave_function_array(u1, i)
                else:
                    cmo = self.wfs.kpt_u[u].psit_nG[:Nw]
                    cmo1 = self.wfs.kpt_u[u1].psit_nG[:Nw]

                e_G = np.exp(-2.j * pi *
                             np.dot(np.indices(self.gd.n_c).T +
                                    self.gd.beg_c,
                                    Gc / self.gd.N_c).T)
                pw = (e_G * cmo.conj()).reshape((Nw, -1))

                Z_dknn[d, k] += \
                    np.inner(pw, cmo1.reshape((Nw, -1))) * self.gd.dv
                # PAW corrections
                P_ani1 = self.wfs.kpt_u[u1].P_ani
                spos_ac = self.atoms.get_scaled_positions()

                for A, P_ni in self.wfs.kpt_u[u].P_ani.items():
                    dS_ii = self.wfs.setups[A].dO_ii
                    P_n = P_ni[:Nw]
                    P_n1 = P_ani1[A][:Nw]
                    e = np.exp(-2.j * pi * np.dot(Gc, spos_ac[A]))

                    Z_dknn[d, k] += e * P_n.conj().dot(
                        dS_ii.dot(P_n1.T))

        self.gd.comm.sum(Z_dknn)
        self.Z_dknn = Z_dknn.copy()

        self.initialize()

    def initialize(self):
        """Re-initialize current rotation matrix.

        Keywords are identical to those of the constructor.
        """
        Nw = self.nwannier

        # Set U to random (orthogonal) matrix
        self.U_kww = np.zeros((self.Nk, Nw, Nw), self.dtype)

        # for k in range(self.Nk):
        if self.dtype == float:
            real = True
        else:
            real = False
        self.U_kww[:] = random_orthogonal_matrix(Nw, self.rng, real=real)

        self.update()

    def update(self):

        # Calculate the Zk matrix from the rotation matrix:
        # Zk = U^d[k] Zbloch U[k1]
        for d in range(self.Ndir):
            for k in range(self.Nk):
                k1 = self.kklst_dk[d, k]
                self.Z_dkww[d, k] = np.dot(dag(self.U_kww[k]), np.dot(
                    self.Z_dknn[d, k], self.U_kww[k1]))

        # Update the new Z matrix
        self.Z_dww = self.Z_dkww.sum(axis=1) / self.Nk

    def get_centers(self, scaled=False):
        """Calculate the Wannier centers

        ::

          pos =  L / 2pi * phase(diag(Z))
        """
        coord_wc = \
            np.angle(self.Z_dww[:3].diagonal(0, 1, 2)).T / \
            (2.0 * pi) % 1
        if not scaled:
            coord_wc = np.dot(coord_wc, self.largeunitcell_cc)
        return coord_wc

    def localize(self, step=0.25, tolerance=1e-08,
                 updaterot=True):
        """Optimize rotation to give maximal localization"""
        md_min(self, step, tolerance, verbose=self.verbose,
               updaterot=updaterot)

    def get_function_value(self):
        """Calculate the value of the spread functional.

        ::

          Tr[|ZI|^2]=sum(I)sum(n) w_i|Z_(i)_nn|^2,

        where w_i are weights."""
        a_d = np.sum(np.abs(self.Z_dww.diagonal(0, 1, 2)) ** 2,
                     axis=1)
        return np.dot(a_d, self.weight_d).real

    def get_gradients(self):

        Nw = self.nwannier
        dU = []
        for k in range(self.Nk):
            Utemp_ww = np.zeros((Nw, Nw), complex)

            for d, weight in enumerate(self.weight_d):
                if abs(weight) < 1.0e-6:
                    continue

                diagZ_w = self.Z_dww[d].diagonal()
                Zii_ww = np.repeat(diagZ_w, Nw).reshape(Nw, Nw)
                k2 = self.invkklst_dk[d, k]
                Z_kww = self.Z_dkww[d]

                temp = Zii_ww.T * Z_kww[k].conj() - \
                    Zii_ww * Z_kww[k2].conj()
                Utemp_ww += weight * (temp - dag(temp))
            dU.append(Utemp_ww.ravel())

        return np.concatenate(dU)

    def step(self, dX, updaterot=True):
        Nw = self.nwannier
        Nk = self.Nk
        if updaterot:
            A_kww = dX[:Nk * Nw ** 2].reshape(Nk, Nw, Nw)
            for U, A in zip(self.U_kww, A_kww):
                H = -1.j * A.conj()
                epsilon, Z = np.linalg.eigh(H)
                # Z contains the eigenvectors as COLUMNS.
                # Since H = iA, dU = exp(-A) = exp(iH) = ZDZ^d
                dU = np.dot(Z * np.exp(1.j * epsilon), dag(Z))
                if U.dtype == float:
                    U[:] = np.dot(U, dU).real
                else:
                    U[:] = np.dot(U, dU)

        self.update()
