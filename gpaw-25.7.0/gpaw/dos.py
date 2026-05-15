from __future__ import annotations
from pathlib import Path
from typing import Union, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
from ase.dft.dos import linear_tetrahedron_integration as lti

from gpaw.setup import Setup
from gpaw.spinorbit import soc_eigenstates, BZWaveFunctions
from gpaw.typing import Array1D, Array2D, Array3D, ArrayLike1D

if TYPE_CHECKING:
    from gpaw.calculator import GPAW
    from gpaw.new.ase_interface import ASECalculator


class IBZWaveFunctions:
    """Container for eigenvalues and PAW projections (only IBZ)."""
    def __init__(self, calc: ASECalculator | GPAW):
        self.calc = calc
        self.fermi_level = self.calc.get_fermi_level()
        self.size = calc.wfs.kd.N_c
        self.bz2ibz_map = calc.wfs.kd.bz2ibz_k

    def weights(self) -> Array1D:
        """Weigths of IBZ k-points (adds to 1.0)."""
        return self.calc.wfs.kd.weight_k

    def eigenvalues(self) -> Array3D:
        """All eigenvalues."""
        kd = self.calc.wfs.kd
        eigs = np.array([[self.calc.get_eigenvalues(kpt=k, spin=s)
                          for k in range(kd.nibzkpts)]
                         for s in range(kd.nspins)])
        return eigs

    def pdos_weights(self,
                     a: int,
                     indices: List[int]
                     ) -> Array3D:
        """Projections for PDOS.

        Returns (nibzkpts, nbands, nspins)-shaped ndarray
        of the square of absolute value of the projections.  The *indices*
        list contains projector-indices.
        """
        kd = self.calc.wfs.kd
        dos_kns = np.zeros((kd.nibzkpts,
                            self.calc.wfs.bd.nbands,
                            kd.nspins))
        bands = self.calc.wfs.bd.get_slice()

        for wf in self.calc.wfs.kpt_u:
            P_ani = wf.projections
            if a in P_ani:
                P_ni = P_ani[a][:, indices]
                dos_kns[wf.k, bands, wf.s] = (abs(P_ni)**2).sum(1)

        self.calc.world.sum(dos_kns)
        return dos_kns


def get_projector_numbers(setup: Setup, ell: int) -> List[int]:
    """Find indices of bound-state PAW projector functions.

    >>> from gpaw.setup import create_setup
    >>> setup = create_setup('Li')
    >>> get_projector_numbers(setup, 0)
    [0]
    >>> get_projector_numbers(setup, 1)
    [1, 2, 3]
    """
    indices = []
    i1 = 0
    for n, l in zip(setup.n_j, setup.l_j):
        i2 = i1 + 2 * l + 1
        if l == ell and n >= 0:
            indices += list(range(i1, i2))
        i1 = i2
    return indices


def gaussian_dos(eig_kn,
                 weight_kn,
                 weight_k,
                 energies,
                 width: float) -> Array1D:
    """Simple broadening with a Gaussian."""
    dos = np.zeros_like(energies)
    if weight_kn is None:
        for e_n, w in zip(eig_kn, weight_k):
            for e in e_n:
                dos += w * np.exp(-((energies - e) / width)**2)
    else:
        for e_n, w, w_n in zip(eig_kn, weight_k, weight_kn):
            for e, w2 in zip(e_n, w_n):
                dos += w * w2 * np.exp(-((energies - e) / width)**2)
    return dos / (np.pi**0.5 * width)


def linear_tetrahedron_dos(eig_kn,
                           weight_kn,
                           energies,
                           cell,
                           size,
                           bz2ibz_map=None) -> Array1D:
    """Linear-tetrahedron method."""
    if len(eig_kn) != np.prod(size):
        eig_kn = eig_kn[bz2ibz_map]
        if weight_kn is not None:
            weight_kn = weight_kn[bz2ibz_map]

    shape = tuple(size) + (-1,)
    eig_kn = eig_kn.reshape(shape)
    if weight_kn is not None:
        weight_kn = weight_kn.reshape(shape)

    dos = lti(cell, eig_kn, energies, weight_kn)
    return dos


class DOSCalculator:
    def __init__(self,
                 wfs,
                 setups=None,
                 cell=None,
                 shift_fermi_level=True):
        self.wfs = wfs
        self.setups = setups
        self.cell = cell

        self.eig_skn = wfs.eigenvalues()
        self.fermi_level = wfs.fermi_level

        if shift_fermi_level:
            self.eig_skn -= wfs.fermi_level

        self.collinear = (self.eig_skn.ndim == 3)
        if self.collinear:
            self.degeneracy = 2 / len(self.eig_skn)
        else:
            self.eig_skn = np.array([self.eig_skn, self.eig_skn])
            self.degeneracy = 0.5

        self.nspins = len(self.eig_skn)
        self.weight_k = wfs.weights()

    def get_energies(self,
                     emin: Optional[float] = None,
                     emax: Optional[float] = None,
                     npoints: int = 100):
        emin = emin if emin is not None else self.eig_skn.min()
        emax = emax if emax is not None else self.eig_skn.max()
        return np.linspace(emin, emax, npoints)

    @classmethod
    def from_calculator(cls,
                        filename: ASECalculator | GPAW | Path | str,
                        soc=False, theta=0.0, phi=0.0,
                        shift_fermi_level=True) -> DOSCalculator:
        """Create DOSCalculator from a GPAW calculation.

        filename: str
            Name of restart-file or GPAW calculator object.
        """
        from gpaw.calculator import GPAW
        if not isinstance(filename, (str, Path)):
            calc = filename
        else:
            calc = GPAW(filename, txt=None)

        wfs: Union[BZWaveFunctions, IBZWaveFunctions]
        if soc:
            wfs = soc_eigenstates(calc, theta=theta, phi=phi)
        else:
            wfs = IBZWaveFunctions(calc)

        return DOSCalculator(wfs,
                             calc.setups, calc.atoms.cell,
                             shift_fermi_level)

    def calculate(self,
                  energies: ArrayLike1D,
                  eig_kn: Array2D,
                  weight_kn: Array2D = None,
                  width: float = 0.1):
        energies = np.asarray(energies)
        if width > 0.0:
            return gaussian_dos(eig_kn, weight_kn,
                                self.weight_k, energies, width)
        else:
            return linear_tetrahedron_dos(
                eig_kn, weight_kn, energies,
                self.cell, self.wfs.size, self.wfs.bz2ibz_map)

    def raw_dos(self,
                energies: Sequence[float],
                spin: Optional[int] = None,
                width: float = 0.1) -> Array1D:
        """Calculate density of states.

        width: float
            Width of Gaussians in eV.  Use width=0.0 to use the
            linear-tetrahedron-interpolation method.
        """
        if spin is None:
            dos = sum(self.calculate(energies, eig_kn, width=width)
                      for eig_kn in self.eig_skn)
            dos *= self.degeneracy
        else:
            dos = self.calculate(energies, self.eig_skn[spin], width=width)

        return dos

    def raw_pdos(self,
                 energies: Sequence[float],
                 a: int,
                 l: int,
                 m: Optional[int] = None,
                 spin: Optional[int] = None,
                 width: float = 0.1) -> Array1D:
        """Calculate projected density of states.

        a:
            Atom index.
        l:
            Angular momentum quantum number.
        m:
            Magnetic quantum number.  Default is None meaning sum over all m.
            For p-orbitals, m=0,1,2 translates to y, z and x.
            For d-orbitals, m=0,1,2,3,4 translates to xy, yz, 3z2-r2,
            zx and x2-y2.
        spin:
            Must be 0, 1 or None meaning spin-up, down or total respectively.
        width: float
            Width of Gaussians in eV.  Use width=0.0 to use the
            linear-tetrahedron-interpolation method.
        """
        indices = get_projector_numbers(self.setups[a], l)
        if m is not None:
            indices = indices[m::(2 * l) + 1]
        weight_kns = self.wfs.pdos_weights(a, indices)

        if spin is None:
            dos = sum(self.calculate(energies,
                                     eig_kn,
                                     weight_nk.T,
                                     width=width)
                      for eig_kn, weight_nk
                      in zip(self.eig_skn, weight_kns.T))
            dos *= self.degeneracy
        else:
            dos = self.calculate(energies,
                                 self.eig_skn[spin],
                                 weight_kns[:, :, spin],
                                 width=width)

        return dos
