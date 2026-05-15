from math import pi, sqrt
from warnings import warn
from typing import Optional, List, Dict

import numpy as np
from ase.units import Bohr, Ha

from gpaw.calculator import GPAW
from gpaw.atom.shapefunc import shape_functions
from gpaw.fftw import get_efficient_fft_size
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.utilities import h2gpts
from gpaw.pw.descriptor import PWDescriptor
from gpaw.mpi import serial_comm
from gpaw.setup import Setup
from gpaw.spline import Spline
from gpaw.typing import Array3D


class Interpolator:
    def __init__(self, gd1, gd2, dtype=float):
        self.pd1 = PWDescriptor(0.0, gd1, dtype)
        self.pd2 = PWDescriptor(0.0, gd2, dtype)

    def interpolate(self, a_r):
        return self.pd1.interpolate(a_r, self.pd2)[0]


POINTS = 200


class PS2AE:
    """Transform PS to AE wave functions.

    Interpolates PS wave functions to a fine grid and adds PAW
    corrections in order to obtain true AE wave functions.
    """
    def __init__(self,
                 calc: GPAW,
                 grid_spacing: float = 0.05,
                 n: int = 2,
                 h=None  # deprecated
                 ):
        """Create transformation object.

        calc: GPAW calculator object
            The calcalator that has the wave functions.
        grid_spacing: float
            Desired grid-spacing in Angstrom.
        n: int
            Force number of points to be a mulitiple of n.
        """
        if h is not None:
            warn('Please use grid_spacing=... instead of h=...')
            grid_spacing = h

        self.calc = calc
        gd = calc.wfs.gd

        gd1 = GridDescriptor(gd.N_c, gd.cell_cv, comm=serial_comm)

        # Descriptor for the final grid:
        N_c = h2gpts(grid_spacing / Bohr, gd.cell_cv)
        N_c = np.array([get_efficient_fft_size(N, n) for N in N_c])
        gd2 = self.gd = GridDescriptor(N_c, gd.cell_cv, comm=serial_comm)
        self.interpolator = Interpolator(gd1, gd2, self.calc.wfs.dtype)

        self._dphi: Optional[LFC] = None  # PAW correction

        self.dv = self.gd.dv * Bohr**3

    @property
    def dphi(self) -> LFC:
        if self._dphi is not None:
            return self._dphi

        splines: Dict[Setup, List[Spline]] = {}
        dphi_aj = []
        for setup in self.calc.wfs.setups:
            dphi_j = splines.get(setup)
            if dphi_j is None:
                rcut = max(setup.rcut_j) * 1.1
                gcut = setup.rgd.ceil(rcut)
                dphi_j = []
                for l, phi_g, phit_g in zip(setup.l_j,
                                            setup.data.phi_jg,
                                            setup.data.phit_jg):
                    dphi_g = (phi_g - phit_g)[:gcut]
                    dphi_j.append(setup.rgd.spline(dphi_g, rcut, l,
                                                   points=200))
                splines[setup] = dphi_j
            dphi_aj.append(dphi_j)

        self._dphi = LFC(self.gd, dphi_aj, kd=self.calc.wfs.kd.copy(),
                         dtype=self.calc.wfs.dtype)
        self._dphi.set_positions(self.calc.spos_ac)

        return self._dphi

    def get_wave_function(self,
                          n: int,
                          k: int = 0,
                          s: int = 0,
                          ae: bool = True,
                          periodic: bool = False) -> Array3D:
        """Interpolate wave function.

        Returns 3-d array in units of Ang**-1.5.

        n: int
            Band index.
        k: int
            K-point index.
        s: int
            Spin index.
        ae: bool
            Add PAW correction to get an all-electron wave function.
        periodic:
            Return periodic part of wave-function, u(r), instead of
            psi(r)=exp(ikr)u(r).
        """
        u_r = self.calc.get_pseudo_wave_function(n, k, s,
                                                 periodic=True)
        u_R = self.interpolator.interpolate(u_r * Bohr**1.5)

        k_c = self.calc.wfs.kd.ibzk_kc[k]
        gamma = np.isclose(k_c, 0.0).all()

        if gamma:
            eikr_R = 1.0
        else:
            eikr_R = self.gd.plane_wave(k_c)

        if ae:
            dphi = self.dphi
            wfs = self.calc.wfs
            P_nI = wfs.collect_projections(k, s)

            if wfs.world.rank == 0:
                psi_R = u_R * eikr_R
                P_ai = {}
                I1 = 0
                for a, setup in enumerate(wfs.setups):
                    I2 = I1 + setup.ni
                    P_ai[a] = P_nI[n, I1:I2]
                    I1 = I2
                dphi.add(psi_R, P_ai, k)
                u_R = psi_R / eikr_R

            wfs.world.broadcast(u_R, 0)

        if periodic:
            return u_R * Bohr**-1.5
        else:
            return u_R * eikr_R * Bohr**-1.5

    def get_pseudo_density(self,
                           add_compensation_charges: bool = True) -> Array3D:
        """Interpolate pseudo density."""
        dens = self.calc.density
        gd1 = dens.gd
        assert gd1.comm.size == 1
        interpolator = Interpolator(gd1, self.gd)
        dens_r = dens.nt_sG[:dens.nspins].sum(axis=0)
        dens_R = interpolator.interpolate(dens_r)

        if add_compensation_charges:
            dens.calculate_multipole_moments()
            ghat = LFC(self.gd, [setup.ghat_l for setup in dens.setups],
                       integral=sqrt(4 * pi))
            ghat.set_positions(self.calc.spos_ac)
            Q_aL = {}
            for a, Q_L in dens.Q_aL.items():
                Q_aL[a] = Q_L.copy()
                Q_aL[a][0] += dens.setups[a].Nv / (4 * pi)**0.5
            ghat.add(dens_R, Q_aL)

        return dens_R / Bohr**3

    def get_electrostatic_potential(self,
                                    ae: bool = True,
                                    rcgauss: float = 0.02) -> Array3D:
        """Interpolate electrostatic potential.

        Return value in eV.

        ae: bool
            Add PAW correction to get the all-electron potential.
        rcgauss: float
            Width of gaussian (in Angstrom) used to represent the nuclear
            charge.
        """
        gd = self.calc.hamiltonian.finegd
        v_r = self.calc.get_electrostatic_potential() / Ha
        gd1 = GridDescriptor(gd.N_c, gd.cell_cv, comm=serial_comm)
        interpolator = Interpolator(gd1, self.gd)
        v_R = interpolator.interpolate(v_r)

        if ae:
            self.add_potential_correction(v_R, rcgauss / Bohr)

        return v_R * Ha

    def add_potential_correction(self,
                                 v_R: Array3D,
                                 rcgauss: float) -> None:
        dens = self.calc.density
        dens.D_asp.redistribute(dens.atom_partition.as_serial())
        dens.Q_aL.redistribute(dens.atom_partition.as_serial())

        dv_a1 = []
        for a, D_sp in dens.D_asp.items():
            setup = dens.setups[a]
            c = setup.xc_correction
            rgd = c.rgd
            params = setup.data.shape_function.copy()
            params['lmax'] = 0
            ghat_g = shape_functions(rgd, **params)[0]
            Z_g = shape_functions(rgd, 'gauss', rcgauss, lmax=0)[0] * setup.Z
            D_q = np.dot(D_sp.sum(0), c.B_pqL[:, :, 0])
            dn_g = np.dot(D_q, (c.n_qg - c.nt_qg)) * sqrt(4 * pi)
            dn_g += 4 * pi * (c.nc_g - c.nct_g)
            dn_g -= Z_g
            dn_g -= dens.Q_aL[a][0] * ghat_g * sqrt(4 * pi)
            dv_g = rgd.poisson(dn_g) / sqrt(4 * pi)
            dv_g[1:] /= rgd.r_g[1:]
            dv_g[0] = dv_g[1]
            dv_g[-1] = 0.0
            dv_a1.append([rgd.spline(dv_g, points=POINTS)])

        dens.D_asp.redistribute(dens.atom_partition)
        dens.Q_aL.redistribute(dens.atom_partition)

        if dv_a1:
            dv = LFC(self.gd, dv_a1)
            dv.set_positions(self.calc.spos_ac)
            dv.add(v_R)
        dens.gd.comm.broadcast(v_R, 0)


def interpolate_weight(calc, weight, h=0.05, n=2):
    """interpolates cdft weight function, gd is the fine grid."""
    gd = calc.density.finegd

    weight = gd.collect(weight, broadcast=True)
    weight = gd.zero_pad(weight)

    w = np.zeros_like(weight)
    gd1 = GridDescriptor(gd.N_c, gd.cell_cv, comm=serial_comm)
    gd1.distribute(weight, w)

    N_c = h2gpts(h / Bohr, gd.cell_cv)
    N_c = np.array([get_efficient_fft_size(N, n) for N in N_c])
    gd2 = GridDescriptor(N_c, gd.cell_cv, comm=serial_comm)

    interpolator = Interpolator(gd1, gd2)
    W = interpolator.interpolate(w)

    return W
