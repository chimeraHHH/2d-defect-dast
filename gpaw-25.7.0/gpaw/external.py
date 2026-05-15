"""This module defines different external potentials."""
import copy
import warnings
from typing import Callable, Dict, Optional

import numpy as np
from ase.units import Bohr, Ha

import gpaw.cgpaw as cgpaw
from gpaw.typing import Array3D

__all__ = ['ConstantPotential', 'ConstantElectricField', 'CDFTPotential',
           'PointChargePotential', 'StepPotentialz',
           'PotentialCollection']

known_potentials: Dict[str, Callable] = {}


def _register_known_potentials():
    from gpaw.bfield import BField
    known_potentials['CDFTPotential'] = lambda: None  # ???
    known_potentials['BField'] = BField
    for name in __all__:
        known_potentials[name] = globals()[name]


def create_external_potential(name, **kwargs):
    """Construct potential from dict."""
    if not known_potentials:
        _register_known_potentials()
    return known_potentials[name](**kwargs)


class ExternalPotential:
    vext_g: Optional[Array3D] = None
    vext_q: Optional[Array3D] = None

    def get_potential(self, gd):
        """Get the potential on a regular 3-d grid.

        Will only call calculate_potential() the first time."""

        if self.vext_g is None:
            self.calculate_potential(gd)
            self.vext_g.flags.writeable = False
        return self.vext_g

    def get_potentialq(self, gd, pd3):
        """Get the potential on a regular 3-d grid in real space.

        Will only call calculate_potential() the first time."""

        if self.vext_q is None:
            vext_g = self.get_potential(gd)
            self.vext_q = pd3.fft(vext_g)
            self.vext_q.flags.writeable = False

        return self.vext_q

    def calculate_potential(self, gd) -> None:
        raise NotImplementedError

    def get_name(self) -> str:
        return self.__class__.__name__

    def update_potential_pw(self, ham, dens) -> float:
        v_q = self.get_potentialq(ham.finegd, ham.pd3).copy()
        eext = ham.pd3.integrate(v_q, dens.rhot_q, global_integral=False)
        dens.map23.add_to1(ham.vt_Q, v_q)
        ham.vt_sG[:] = ham.pd2.ifft(ham.vt_Q)
        if not ham.collinear:
            ham.vt_xG[1:] = 0.0
        return eext

    def update_atomic_hamiltonians_pw(self, ham, W_aL, dens) -> None:
        vext_q = self.get_potentialq(ham.finegd, ham.pd3)
        dens.ghat.integrate(ham.vHt_q + vext_q, W_aL)

    def paw_correction(self, Delta_p, dH_sp) -> None:
        pass

    def derivative_pw(self, ham, ghat_aLv, dens) -> None:
        vext_q = self.get_potentialq(ham.finegd, ham.pd3)
        dens.ghat.derivative(ham.vHt_q + vext_q, ghat_aLv)


class NoExternalPotential(ExternalPotential):
    vext_g = np.zeros((0, 0, 0))

    def __str__(self):
        return 'NoExternalPotential'

    def update_potential_pw(self, ham, dens) -> float:
        ham.vt_sG[:] = ham.pd2.ifft(ham.vt_Q)
        if not ham.collinear:
            ham.vt_xG[1:] = 0.0
        return 0.0

    def update_atomic_hamiltonians_pw(self, ham, W_aL, dens):
        dens.ghat.integrate(ham.vHt_q, W_aL)

    def derivative_pw(self, ham, ghat_aLv, dens):
        dens.ghat.derivative(ham.vHt_q, ghat_aLv)


class ConstantPotential(ExternalPotential):
    """Constant potential for tests."""
    def __init__(self, constant=1.0):
        self.constant = constant / Ha
        self.name = 'ConstantPotential'

    def __str__(self):
        return f'Constant potential: {(self.constant * Ha):.3f} V'

    def calculate_potential(self, gd):
        self.vext_g = gd.zeros() + self.constant

    def todict(self):
        return {'name': self.name,
                'constant': self.constant * Ha}


class ConstantElectricField(ExternalPotential):
    def __init__(self, strength, direction=[0, 0, 1], tolerance=1e-7):
        """External constant electric field.

        strength: float
            Field strength in V/Ang.
        direction: vector
            Polarization direction.
        """
        self.strength = strength * Bohr / Ha
        self.direction_v = np.array(direction, dtype=float)
        self.direction_v /= np.linalg.norm(self.direction_v)
        self.field_v = self.strength * self.direction_v
        self.tolerance = tolerance
        self.name = 'ConstantElectricField'

    def __str__(self):
        return ('Constant electric field: '
                '({:.3f}, {:.3f}, {:.3f}) V/Ang'
                .format(*(self.field_v * Ha / Bohr)))

    def calculate_potential(self, gd):
        # Note that PW-mode is periodic in all directions!
        L_c = abs(gd.cell_cv @ self.direction_v)
        # eps = self.tolerance
        # assert (L_c < eps).sum() == 2

        center_v = 0.5 * gd.cell_cv.sum(0)
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        f_g = (r_gv - center_v) @ self.direction_v

        # Set potential to zero at boundary of box (important for PW-mode).
        # Say we have 8 grid points.  Instead of a potential like this:
        #
        # -4 -3 -2 -1  0  1  2  3
        #
        # we want:
        #
        #  0 -3 -2 -1  0  1  2  3

        L = L_c.sum()
        f_g[abs(abs(f_g) - L / 2) < 1e-5] = 0.0

        self.vext_g = f_g * self.strength

    def todict(self):
        return {'name': self.name,
                'strength': self.strength * Ha / Bohr,
                'direction': self.direction_v}


class ProductPotential(ExternalPotential):
    def __init__(self, ext_i):
        self.ext_i = ext_i

    def calculate_potential(self, gd):
        self.vext_g = self.ext_i[0].get_potential(gd).copy()
        for ext in self.ext_i[1:]:
            self.vext_g *= ext.get_potential(gd)

    def __str__(self):
        return '\n'.join(['Product of potentials:'] +
                         [ext.__str__() for ext in self.ext_i])

    def todict(self):
        return {'name': self.__class__.__name__,
                'ext_i': [ext.todict() for ext in self.ext_i]}


class PointChargePotential(ExternalPotential):
    def __init__(self, charges, positions=None,
                 rc=0.2, rc2=np.inf, width=1.0):
        """Point-charge potential.

        charges: list of float
            Charges in units of `|e|`.
        positions: (N, 3)-shaped array-like of float
            Positions of charges in Angstrom.  Can be set later.
        rc: float
            Inner cutoff for Coulomb potential in Angstrom.
        rc2: float
            Outer cutoff for Coulomb potential in Angstrom.
        width: float
            Width for cutoff function for Coulomb part.

        For r < rc, 1 / r is replaced by a third order polynomial in r^2 that
        has matching value, first derivative, second derivative and integral.

        For rc2 - width < r < rc2, 1 / r is multiplied by a smooth cutoff
        function (a third order polynomium in r).

        You can also give rc a negative value.  In that case, this formula
        is used::

            (r^4 - rc^4) / (r^5 - |rc|^5)

        for all values of r - no cutoff at rc2!
        """
        self._dict = dict(name=self.__class__.__name__,
                          charges=charges, positions=positions,
                          rc=rc, rc2=rc2, width=width)
        self.q_p = np.ascontiguousarray(charges, float)
        self.rc = rc / Bohr
        self.rc2 = rc2 / Bohr
        self.width = width / Bohr
        if positions is not None:
            self.set_positions(positions)
        else:
            self.R_pv = None

        if abs(self.q_p).max() < 1e-14:
            warnings.warn('No charges!')
        if self.rc < 0. and self.rc2 < np.inf:
            warnings.warn('Long range cutoff chosen but will not be applied\
                           for negative inner cutoff values!')

    def todict(self):
        return copy.deepcopy(self._dict)

    def __str__(self):
        return ('Point-charge potential '
                '(points: {}, cutoffs: {:.3f}, {:.3f}, {:.3f} Ang)'
                .format(len(self.q_p),
                        self.rc * Bohr,
                        (self.rc2 - self.width) * Bohr,
                        self.rc2 * Bohr))

    def set_positions(self, R_pv, com_pv=None):
        """Update positions."""
        if com_pv is not None:
            self.com_pv = np.asarray(com_pv) / Bohr
        else:
            self.com_pv = None

        self.R_pv = np.asarray(R_pv) / Bohr
        self.vext_g = None

    def _molecule_distances(self, gd):
        if self.com_pv is not None:
            return self.com_pv - gd.cell_cv.sum(0) / 2

    def calculate_potential(self, gd):
        assert gd.orthogonal
        self.vext_g = gd.zeros()

        dcom_pv = self._molecule_distances(gd)

        cgpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.q_p, self.R_pv,
                           self.rc, self.rc2, self.width,
                           self.vext_g, dcom_pv)

    def get_forces(self, calc):
        """Calculate forces from QM charge density on point-charges."""
        dens = calc.density
        F_pv = np.zeros_like(self.R_pv)
        gd = dens.finegd
        dcom_pv = self._molecule_distances(gd)

        cgpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.q_p, self.R_pv,
                           self.rc, self.rc2, self.width,
                           self.vext_g, dcom_pv, dens.rhot_g, F_pv)
        gd.comm.sum(F_pv)
        return F_pv * Ha / Bohr


class CDFTPotential(ExternalPotential):
    # Dummy class to make cDFT compatible with new external
    # potential class ClassName(object):
    def __init__(self, regions, constraints, n_charge_regions,
                 difference):

        self.name = 'CDFTPotential'
        self.regions = regions
        self.constraints = constraints
        self.difference = difference
        self.n_charge_regions = n_charge_regions

    def todict(self):
        return {'name': 'CDFTPotential',
                # 'regions': self.indices_i,
                'constraints': self.v_i * Ha,
                'n_charge_regions': self.n_charge_regions,
                'difference': self.difference,
                'regions': self.regions}


class StepPotentialz(ExternalPotential):
    def __init__(self, zstep, value_left=0, value_right=0):
        """Step potential in z-direction

        zstep: float
            z-value that splits space into left and right [Angstrom]
        value_left: float
            Left side (z < zstep) potentential value [eV]. Default: 0
        value_right: float
            Right side (z >= zstep) potentential value [eV]. Default: 0
       """
        self.value_left = value_left
        self.value_right = value_right
        self.name = 'StepPotentialz'
        self.zstep = zstep

    def __str__(self):
        return f'Step potentialz: {self.value_left:.3f} V to '\
               f'{self.value_right:.3f} V at z={self.zstep}'

    def calculate_potential(self, gd):
        r_vg = gd.get_grid_point_coordinates()
        self.vext_g = np.where(r_vg[2] < self.zstep / Bohr,
                               gd.zeros() + self.value_left / Ha,
                               gd.zeros() + self.value_right / Ha)

    def todict(self):
        return {'name': self.name,
                'value_left': self.value_left,
                'value_right': self.value_right,
                'zstep': self.zstep}


class PotentialCollection(ExternalPotential):
    def __init__(self, potentials):
        """Collection of external potentials to be applied

        potentials: list
            List of potentials
        """
        self.potentials = []
        for potential in potentials:
            if isinstance(potential, dict):
                potential = create_external_potential(
                    potential.pop('name'), **potential)
            self.potentials.append(potential)

    def __str__(self):
        text = 'PotentialCollection:\n'
        for pot in self.potentials:
            text += '  ' + pot.__str__() + '\n'
        return text

    def calculate_potential(self, gd):
        self.potentials[0].calculate_potential(gd)
        self.vext_g = self.potentials[0].vext_g.copy()
        for pot in self.potentials[1:]:
            pot.calculate_potential(gd)
            self.vext_g += pot.vext_g

    def todict(self):
        return {'name': 'PotentialCollection',
                'potentials': [pot.todict() for pot in self.potentials]}


def static_polarizability(atoms, strength=0.01):
    """Calculate polarizability tensor

    atoms: Atoms object
    strength: field strength in V/Ang

    Returns
    -------
    polarizability tensor:
        Unit (e^2 Angstrom^2 / eV).
        Multiply with Bohr * Ha to get (Angstrom^3)
    """
    atoms.get_potential_energy()
    calc = atoms.calc
    assert calc.parameters.external is None
    dipole_gs = calc.get_dipole_moment()

    alpha = np.zeros((3, 3))
    for c in range(3):
        axes = np.zeros(3)
        axes[c] = 1
        calc.set(external=ConstantElectricField(strength, axes))
        calc.get_potential_energy()
        alpha[c] = (calc.get_dipole_moment() - dipole_gs) / strength
    calc.set(external=None)

    return alpha.T
