import json
import re
from typing import Tuple

import numpy as np
from ase import Atoms
from ase.units import Bohr
from ase.utils import IOContext
from gpaw.fd_operators import Gradient
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.typing import Vector
from gpaw.utilities.tools import coordinates


def calculate_magnetic_moment_on_grid(wfs, grad_v, r_vG, dM_vaii, *,
                                      only_pseudo=False):
    """Calculate magnetic moment on grid.

    Parameters
    ----------
    wfs
        Wave functions object
    grad_v
        List of gradient operators
    r_vG
        Grid point coordinates
    dM_vaii
        Atomic PAW corrections for magnetic moment
    only_pseudo
        If true, do not add atomic corrections

    Returns
    -------
    Magnetic moment vector
    """
    gd = wfs.gd
    mode = wfs.mode
    bd = wfs.bd
    kpt_u = wfs.kpt_u

    rxnabla_v = np.zeros(3, dtype=complex)
    if mode == 'lcao':
        psit_G = gd.empty(dtype=complex)
    nabla_psit_vG = gd.empty(3, dtype=complex)
    for kpt in kpt_u:
        for n, f in enumerate(kpt.f_n):
            if mode == 'lcao':
                psit_G[:] = 0.0
                wfs.basis_functions.lcao_to_grid(kpt.C_nM[n], psit_G, kpt.q)
            else:
                psit_G = kpt.psit_nG[n]

            for v in range(3):
                grad_v[v].apply(psit_G, nabla_psit_vG[v], kpt.phase_cd)

            # rxnabla   = <psi1| r x nabla |psi2>
            # rxnabla_x = <psi1| r_y nabla_z - r_z nabla_y |psi2>
            # rxnabla_y = <psi1| r_z nabla_x - r_x nabla_z |psi2>
            # rxnabla_z = <psi1| r_x nabla_y - r_y nabla_x |psi2>
            for v in range(3):
                v1 = (v + 1) % 3
                v2 = (v + 2) % 3
                rnabla_psit_G = (r_vG[v1] * nabla_psit_vG[v2]
                                 - r_vG[v2] * nabla_psit_vG[v1])
                rxnabla_v[v] += f * gd.integrate(psit_G.conj() * rnabla_psit_G)

    if not only_pseudo:
        paw_rxnabla_v = np.zeros(3, dtype=complex)
        for kpt in kpt_u:
            for v in range(3):
                for a, P_ni in kpt.P_ani.items():
                    paw_rxnabla_v[v] += np.einsum('n,ni,ij,nj',
                                                  kpt.f_n, P_ni.conj(),
                                                  dM_vaii[v][a], P_ni,
                                                  optimize=True)
        gd.comm.sum(paw_rxnabla_v)
        rxnabla_v += paw_rxnabla_v

    bd.comm.sum(rxnabla_v)
    return -0.5 * rxnabla_v.imag


def calculate_magnetic_moment_atomic_corrections(R_av, setups, partition):
    """Calculate atomic PAW augmentation corrections for magnetic moment.

    Parameters
    ----------
    R_av
        Atom positions
    setups
        PAW setups object
    partition
        Atom partition object

    Returns
    -------
    Atomic correction matrices
    """
    # augmentation contributions to magnetic moment
    # <psi1| r x nabla |psi2> = <psi1| (r - Ra + Ra) x nabla |psi2>
    #                         = <psi1| (r - Ra) x nabla |psi2>
    #                             + Ra x <psi1| nabla |psi2>

    def shape(a):
        ni = setups[a].ni
        return ni, ni

    dM_vaii = []
    for _ in range(3):
        dM_aii = partition.arraydict(shapes=shape, dtype=complex)
        dM_vaii.append(dM_aii)

    for a in partition.my_indices:
        Ra_v = R_av[a]
        rxnabla_iiv = setups[a].rxnabla_iiv
        nabla_iiv = setups[a].nabla_iiv

        # rxnabla   = <psi1| (r - Ra) x nabla |psi2>
        # Rxnabla   = Ra x <psi1| nabla |psi2>
        # Rxnabla_x = Ra_y nabla_z - Ra_z nabla_y
        # Rxnabla_y = Ra_z nabla_x - Ra_x nabla_z
        # Rxnabla_z = Ra_x nabla_y - Ra_y nabla_x
        for v in range(3):
            v1 = (v + 1) % 3
            v2 = (v + 2) % 3
            Rxnabla_ii = (Ra_v[v1] * nabla_iiv[:, :, v2]
                          - Ra_v[v2] * nabla_iiv[:, :, v1])
            dM_vaii[v][a][:] = Rxnabla_ii + rxnabla_iiv[:, :, v]

    return dM_vaii


def calculate_magnetic_moment_matrix(kpt_u, bfs, correction, r_vG, dM_vaii, *,
                                     only_pseudo=False):
    """Calculate magnetic moment matrix in LCAO basis.

    Parameters
    ----------
    kpt_u
        K-points
    bfs
        Basis functions object
    correction
        Correction object
    r_vG
        Grid point coordinates
    dM_vaii
        Atomic PAW corrections for magnetic moment
    only_pseudo
        If true, do not add PAW corrections

    Returns
    -------
    Magnetic moment matrix
    """
    Mstart = correction.Mstart
    Mstop = correction.Mstop
    mynao = Mstop - Mstart
    nao = bfs.Mmax

    assert bfs.Mstart == Mstart
    assert bfs.Mstop == Mstop

    M_vmM = np.zeros((3, mynao, nao), dtype=complex)
    rnabla_vmM = np.empty((3, mynao, nao), dtype=complex)

    for v in range(3):
        v1 = (v + 1) % 3
        v2 = (v + 2) % 3
        rnabla_vmM[:] = 0.0
        bfs.calculate_potential_matrix_derivative(r_vG[v], rnabla_vmM, 0)
        M_vmM[v1] += rnabla_vmM[v2]
        M_vmM[v2] -= rnabla_vmM[v1]

    if not only_pseudo:
        for kpt in kpt_u:
            assert kpt.k == 0

        for v in range(3):
            correction.calculate(kpt_u[0].q, dM_vaii[v], M_vmM[v],
                                 Mstart, Mstop)

    # The matrices should be real
    assert np.max(np.absolute(M_vmM.imag)) == 0.0
    M_vmM = M_vmM.real.copy()
    return -0.5 * M_vmM


def calculate_magnetic_moment_in_lcao(ksl, rho_mm, M_vmm):
    """Calculate magnetic moment in LCAO.

    Parameters
    ----------
    ksl
        Kohn-Sham Layouts object
    rho_mm
        Density matrix in LCAO basis
    M_vmm
        Magnetic moment matrix in LCAO basis

    Returns
    -------
    Magnetic moment vector
    """
    assert M_vmm.dtype == float
    mm_v = np.sum(rho_mm.imag * M_vmm, axis=(1, 2))
    if ksl.using_blacs:
        ksl.mmdescriptor.blacsgrid.comm.sum(mm_v)
    return mm_v


def get_origin_coordinates(atoms: Atoms,
                           origin: str,
                           origin_shift: Vector) -> np.ndarray:
    """Get origin coordinates.

    Parameters
    ----------
    atoms
        Atoms object
    origin
        See :class:`~gpaw.tddft.MagneticMomentWriter`
    origin_shift
        See :class:`~gpaw.tddft.MagneticMomentWriter`

    Returns
    -------
    Origin coordinates in atomic units
    """
    if origin == 'COM':
        origin_v = atoms.get_center_of_mass()
    elif origin == 'COC':
        origin_v = 0.5 * atoms.get_cell().sum(0)
    elif origin == 'zero':
        origin_v = np.zeros(3, dtype=float)
    else:
        raise ValueError('unknown origin')
    origin_v += np.asarray(origin_shift, dtype=float)
    return origin_v / Bohr


def parse_header(line: str) -> Tuple[str, int, dict]:
    """Parse header line.

    Example header line (keyword arguments as json):

        NameOfWriter[version=1](**{"arg1": "abc", ...})

    Parameters
    ----------
    line
        Header line

    Returns
    -------
    name
        Name
    version
        Version
    kwargs
        Keyword arguments

    Raises
    ------
    ValueError
        Line cannot be parsed
    """
    regexp = r"^(?P<name>\w+)\[version=(?P<ver>\d+)\]\(\*\*(?P<args>.*)\)$"
    m = re.match(regexp, line)
    if m is None:
        raise ValueError('unable parse header')
    name = m.group('name')
    version = int(m.group('ver'))
    try:
        kwargs = json.loads(m.group('args'))
    except json.decoder.JSONDecodeError:
        raise ValueError('unable parse keyword arguments')
    return name, version, kwargs


class MagneticMomentWriter(TDDFTObserver):
    """Observer for writing time-dependent magnetic moment data.

    The data is written in atomic units.

    The observer attaches to the TDDFT calculator during creation.

    Parameters
    ----------
    paw
        TDDFT calculator
    filename
        File for writing magnetic moment data
    origin
        Origin of the coordinate system used in calculation.
        Possible values:
        ``'COM'``: center of mass (default),
        ``'COC'``: center of cell,
        ``'zero'``: (0, 0, 0)
    origin_shift
        Vector in Å shifting the origin from the position defined
        by *origin*.
    dmat
        Density matrix object.
        Define this for LCAO calculations to avoid
        expensive recalculations of the density matrix.
    calculate_on_grid
        Parameter for testing.
        In LCAO mode, if true, calculation is performed on real-space grid.
        In fd mode, calculation is always performed on real-space grid
        and this parameter is neglected.
    only_pseudo
        Parameter for testing.
        If true, PAW corrections are neglected.
    interval
        Update interval. Value of 1 corresponds to evaluating and
        writing data after every propagation step.
    """
    version = 5

    def __init__(self, paw, filename: str, *,
                 origin: str = None,
                 origin_shift: Vector = None,
                 dmat: DensityMatrix = None,
                 calculate_on_grid: bool = None,
                 only_pseudo: bool = None,
                 interval: int = 1):
        super().__init__(paw, interval)
        self.ioctx = IOContext()
        mode = paw.wfs.mode
        assert mode in ['fd', 'lcao'], f'unknown mode: {mode}'
        if paw.niter == 0:
            if origin is None:
                origin = 'COM'
            if origin_shift is None:
                origin_shift = [0., 0., 0.]
            if calculate_on_grid is None:
                calculate_on_grid = mode == 'fd'
            if only_pseudo is None:
                only_pseudo = False
            _kwargs = dict(origin=origin,
                           origin_shift=origin_shift,
                           calculate_on_grid=calculate_on_grid,
                           only_pseudo=only_pseudo)

            # Initialize
            self.fd = self.ioctx.openfile(filename, comm=paw.world, mode='w')
            self._write_header(paw, _kwargs)
        else:
            if origin is not None:
                raise ValueError('Do not set origin in restart')
            if origin_shift is not None:
                raise ValueError('Do not set origin_shift in restart')
            if calculate_on_grid is not None:
                raise ValueError('Do not set calculate_on_grid in restart')
            if only_pseudo is not None:
                raise ValueError('Do not set only_pseudo in restart')

            # Read and continue
            _kwargs = self._read_header(filename)
            origin = _kwargs['origin']  # type: ignore
            origin_shift = _kwargs['origin_shift']  # type: ignore
            calculate_on_grid = _kwargs['calculate_on_grid']  # type: ignore
            only_pseudo = _kwargs['only_pseudo']  # type: ignore
            self.fd = self.ioctx.openfile(filename, comm=paw.world, mode='a')

        atoms = paw.atoms
        gd = paw.wfs.gd
        self.timer = paw.timer

        assert isinstance(origin, str)
        assert isinstance(origin_shift, list)
        origin_v = get_origin_coordinates(atoms, origin, origin_shift)
        R_av = atoms.positions / Bohr - origin_v[np.newaxis, :]
        r_vG, _ = coordinates(gd, origin=origin_v)

        dM_vaii = calculate_magnetic_moment_atomic_corrections(
            R_av, paw.setups, paw.hamiltonian.dH_asp.partition)

        self.calculate_on_grid = calculate_on_grid
        if self.calculate_on_grid:
            self.only_pseudo = only_pseudo
            self.r_vG = r_vG
            self.dM_vaii = dM_vaii

            grad_v = []
            for v in range(3):
                grad_v.append(Gradient(gd, v, dtype=complex, n=2))
            self.grad_v = grad_v
        else:
            M_vmM = calculate_magnetic_moment_matrix(
                paw.wfs.kpt_u, paw.wfs.basis_functions,
                paw.wfs.atomic_correction, r_vG, dM_vaii,
                only_pseudo=only_pseudo)

            # TODO: All observers recalculate density matrix
            # unless dmat is given.
            # Calculator itself could have a density matrix object to avoid
            # this expensive recalculation in a clean way.
            if dmat is None:
                self.dmat = DensityMatrix(paw)
            else:
                self.dmat = dmat
            ksl = paw.wfs.ksl
            if ksl.using_blacs:
                self.M_vmm = ksl.distribute_overlap_matrix(M_vmM)
            else:
                gd.comm.sum(M_vmM)
                self.M_vmm = M_vmM

    def _write(self, line):
        self.fd.write(line)
        self.fd.flush()

    def _write_header(self, paw, kwargs):
        origin_v = get_origin_coordinates(
            paw.atoms, kwargs['origin'], kwargs['origin_shift'])
        lines = [f'{self.__class__.__name__}[version={self.version}]'
                 f'(**{json.dumps(kwargs)})',
                 'origin_v = [%.6f, %.6f, %.6f] Å' % tuple(origin_v * Bohr)]
        self._write('# ' + '\n# '.join(lines) + '\n')
        self._write(f'# {"time":>15} {"mmx":>17} {"mmy":>22} {"mmz":>22}\n')

    def _read_header(self, filename):
        with open(filename, encoding='utf-8') as fd:
            line = fd.readline()
        try:
            name, version, kwargs = parse_header(line[2:])
        except ValueError as e:
            raise ValueError(f'File {filename} cannot be parsed: {e}')
        if name != self.__class__.__name__:
            raise ValueError(f'File {filename} is not '
                             f'for {self.__class__.__name__}')
        if version != self.version:
            raise ValueError(f'File {filename} is not '
                             f'of version {self.version}')
        return kwargs

    def _write_init(self, paw):
        time = paw.time
        line = '# Start; Time = %.8lf\n' % time
        self._write(line)

    def _write_kick(self, paw):
        time = paw.time
        kick = paw.kick_strength
        gauge = paw.kick_gauge
        line = '# Kick = [%22.12le, %22.12le, %22.12le]; ' % tuple(kick)
        line += 'Gauge = %s; ' % gauge
        line += 'Time = %.8lf\n' % time
        self._write(line)

    def _calculate_mm(self, paw):
        if self.calculate_on_grid:
            self.timer.start('Calculate magnetic moment on grid')
            mm_v = calculate_magnetic_moment_on_grid(
                paw.wfs, self.grad_v, self.r_vG, self.dM_vaii,
                only_pseudo=self.only_pseudo)
            self.timer.stop('Calculate magnetic moment on grid')
        else:
            self.timer.start('Calculate magnetic moment in LCAO')

            mm_v = 0.0
            for kpt in paw.wfs.kpt_u:
                assert kpt.q == 0
            for rho_mm in self.dmat.get_density_matrix((paw.niter,
                                                        paw.action)):
                mm_v += calculate_magnetic_moment_in_lcao(
                    paw.wfs.ksl, rho_mm, self.M_vmm)
            self.timer.stop('Calculate magnetic moment in LCAO')
        assert mm_v.shape == (3,)
        assert mm_v.dtype == float
        return mm_v

    def _write_mm(self, paw):
        time = paw.time
        mm_v = self._calculate_mm(paw)
        line = ('%20.8lf %22.12le %22.12le %22.12le\n'
                % (time, mm_v[0], mm_v[1], mm_v[2]))
        self._write(line)

    def _update(self, paw):
        if paw.action == 'init':
            self._write_init(paw)
        elif paw.action == 'kick':
            self._write_kick(paw)
        self._write_mm(paw)

    def __del__(self):
        self.ioctx.close()
