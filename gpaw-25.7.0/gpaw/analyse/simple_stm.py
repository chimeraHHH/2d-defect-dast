from math import sqrt

import numpy as np
from ase.atoms import Atoms
from ase.units import Bohr, Hartree
from ase.io.cube import read_cube_data, write_cube
from ase.dft.stm import STM

import gpaw.mpi as mpi
from gpaw.grid_descriptor import GridDescriptor


class SimpleStm(STM):

    """Simple STM object to simulate STM pictures.

    The simulation uses either a single pseudo-wavefunction (PWF)
    or the PWFs inside the given bias range."""

    def __init__(self, atoms):

        self.filename = None
        self.is_wf = False
        self.bias = None
        self.ldos = None
        self.heights = None

        if isinstance(atoms, str):
            self.read_3D(atoms)
            self.calc = None
            self.atoms = None
        else:
            if isinstance(atoms, Atoms):
                self.atoms = atoms
                self.calc = atoms.calc
            else:
                self.calc = atoms
                self.atoms = self.calc.get_atoms()
            self.calc.converge_wave_functions()

            self.gd = self.calc.wfs.gd
            self.offset_c = [int(not a) for a in self.gd.pbc_c]

            super().__init__(self.atoms)

    def calculate_ldos(self, bias):
        """bias is the n, k, s list/tuple."""
        if self.calc is None:
            return

        self.bias = bias
        self.is_wf = True
        self.ldos = self.gd.zeros()

        if hasattr(bias, '__len__') and len(bias) == 3:
            n, k, s = bias
            # only a single wf requested
            kd = self.calc.wfs.kd
            rank, q = kd.who_has(k)
            if kd.comm.rank == rank:
                u = q * kd.nspins + s
                self.add_wf_to_ldos(n, u, weight=1)

        else:
            # energy bias
            try:
                efermi_s = self.calc.get_fermi_levels()
            except ValueError:
                efermi_s = np.array([self.calc.get_fermi_level()] * 2)

            if isinstance(bias, (int, float)):
                # bias given
                if bias > 0:
                    # positive bias = negative tip
                    # -> probe unoccupied states
                    emin_s = efermi_s
                    emax_s = efermi_s + bias
                    occupied = False
                else:
                    # negative bias = positive tip
                    # -> probe occupied states
                    emin_s = efermi_s + bias
                    emax_s = efermi_s
                    occupied = True
            else:
                # emin and emax given
                emin, emax = bias
                if abs(emin) > abs(emax):
                    occupied = True
                else:
                    occupied = False
                emin_s = np.array([emin + efermi_s] * 2)
                emax_s = np.array([emax + efermi_s] * 2)

            emin_s /= Hartree
            emax_s /= Hartree

            for u in range(len(self.calc.wfs.kpt_u)):
                kpt = self.calc.wfs.kpt_u[u]
                emin = emin_s[kpt.s]
                emax = emax_s[kpt.s]
                for n, eps in enumerate(kpt.eps_n):
                    if eps > emin and eps < emax:
                        if occupied:
                            weight = kpt.f_n[n]
                        else:
                            weight = kpt.weight - kpt.f_n[n]
                        self.add_wf_to_ldos(n, u, weight)

    def add_wf_to_ldos(self, n, u, weight=None):
        """Add the wf with given kpoint and spin to the ldos"""
        kpt = self.calc.wfs.kpt_u[u]
        psi = kpt.psit_nG[n]
        w = weight
        if w is None:
            w = kpt.weight
# print "w=", w, kpt.weight
        self.ldos += w * (psi * np.conj(psi)).real

    def write_3D(self, bias, filename, filetype=None):
        """Write the density as a 3D file.

        Units: [e/A^3]"""
        self.calculate_ldos(bias)
        self.calc.wfs.kd.comm.sum(self.ldos)
        ldos = self.gd.collect(self.ldos)
# print "write: integrated =", self.gd.integrate(self.ldos)

        if mpi.rank != 0:
            return

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype = filetype.lower()

        if filetype == 'cube':
            with open(filename, 'w') as fd:
                write_cube(fd, self.calc.get_atoms(), ldos / Bohr ** 3)
        else:
            raise NotImplementedError('unknown file type "' + filetype + '"')

    def read_3D(self, filename, filetype=None):
        """Read the density from a 3D file"""

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]

        if filetype == 'cube':
            data, atoms = read_cube_data(filename)
            self.cell = atoms.cell.array

            pbc_c = [True, True, True]
            N_c = np.array(data.shape)
            for c in range(3):
                if N_c[c] % 2 == 1:
                    pbc_c[c] = False
                    N_c[c] += 1
            self.gd = GridDescriptor(N_c, self.cell.diagonal() / Bohr, pbc_c)
            self.offset_c = [int(not a) for a in self.gd.pbc_c]

        else:
            raise NotImplementedError('unknown file type "' + filetype + '"')

        self.filename = filename
        self.ldos = np.array(data * Bohr ** 3)
# print "read: integrated =", self.gd.integrate(self.ldos)

    def current_to_density(self, current):
        """The connection between density n and current I

        n [e/Angstrom^3] = 0.0002 sqrt(I [nA])

        as given in Hofer et al., RevModPhys 75 (2003) 1287
        """
        return 0.0002 * sqrt(current)

    def linescan(self, bias, current, p1, p2, npoints=50, z0=None):
        """Line scan for a given current [nA]"""
        return STM.linescan(self, bias,
                            self.current_to_density(current),
                            p1, p2, npoints, z0)

    def density_to_current(self, density):
        return 5000. * density ** 2

    def scan_const_current(self, current, bias):
        """Get the height image for constant current I [nA].
        """
        return self.scan_const_density(self.current_to_density(current),
                                       bias)

    def scan_const_density(self, density, bias):
        """Get the height image for constant density [e/Angstrom^3].
        """
        x, y, heights = self.scan(bias, density)
        self.heights = heights / Bohr

        return self.heights
