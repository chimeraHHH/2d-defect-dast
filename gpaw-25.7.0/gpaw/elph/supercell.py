"""Module for electron-phonon supercell properties."""

import numpy as np
from typing import Tuple

from ase import Atoms
from ase.parallel import parprint
from ase.units import Bohr
from ase.utils.filecache import MultiFileJSONCache

from gpaw.calculator import GPAW
from gpaw.lcao.tightbinding import TightBinding
from gpaw.typing import ArrayND
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.tools import tri2full

from .filter import fourier_filter

sc_version = 1
# v1: saves natom, supercell, g_sNNMM.shape and dtype


class VersionError(Exception):
    """Error raised for wrong cache versions."""
    pass


class Supercell:
    """Class for supercell-related stuff."""

    def __init__(self, atoms: Atoms, supercell_name: str = "supercell",
                 supercell: tuple = (1, 1, 1), indices=None) -> None:
        """Initialize supercell class.

        Parameters
        ----------
        atoms: Atoms
            The atoms to work on. Primitive cell.
        supercell_name: str
            User specified name of the generated JSON cache.
            Default is 'supercell'.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        """
        self.atoms = atoms
        self.supercell_name = supercell_name
        self.supercell = supercell
        if indices is None:
            self.indices = np.arange(len(atoms))
        else:
            self.indices = indices

    def _calculate_supercell_entry(self, a, v, V1t_sG, dH1_asp, wfs,
                                   dH_asp) -> ArrayND:
        kpt_u = wfs.kpt_u
        setups = wfs.setups
        nao = setups.nao
        bfs = wfs.basis_functions
        dtype = wfs.dtype
        nspins = wfs.nspins

        # Array for different k-point components
        g_sqMM = np.zeros((nspins, len(kpt_u) // nspins, nao, nao), dtype)

        # 1) Gradient of effective potential
        for kpt in kpt_u:
            # Matrix elements
            # Note: somehow this part does not work with gd-parallelisation
            geff_MM = np.zeros((nao, nao), dtype)
            bfs.calculate_potential_matrix(V1t_sG[kpt.s], geff_MM, q=kpt.q)
            tri2full(geff_MM, "L")
            # wfs.gd.comm.sum(geff_MM)
            # print(world.rank, a, v, kpt.k, geff_MM)
            g_sqMM[kpt.s, kpt.q] += geff_MM
            # print(wfs.kd.comm.rank, wfs.gd.comm.rank, wfs.bd.comm.rank,
            #       "\n", geff_MM)

        # 2) Gradient of non-local part (projectors)
        P_aqMi = getattr(wfs, 'P_aqMi', None)
        # 2a) dH^a part has contributions from all atoms
        for kpt in kpt_u:
            # Matrix elements
            gp_MM = np.zeros((nao, nao), dtype)
            for a_, dH1_sp in dH1_asp.items():
                if a_ not in bfs.my_atom_indices:
                    continue
                dH1_ii = unpack_hermitian(dH1_sp[kpt.s])
                if P_aqMi is None:
                    P_Mi = kpt.P_aMi[a_]
                else:
                    P_Mi = P_aqMi[a_][kpt.q]
                gp_MM += P_Mi.conj() @ dH1_ii @ P_Mi.T
            # wfs.gd.comm.sum(gp_MM)
            g_sqMM[kpt.s, kpt.q] += gp_MM

        # 2b) dP^a part has only contributions from the same atoms
        # For the contribution from the derivative of the projectors
        manytci = wfs.manytci
        dPdR_aqvMi = manytci.P_aqMi(bfs.my_atom_indices, derivative=True)
        dH_ii = unpack_hermitian(dH_asp[a][kpt.s])
        for kpt in kpt_u:
            gp_MM = np.zeros((nao, nao), dtype)
            if a in bfs.my_atom_indices:
                if P_aqMi is None:
                    P_Mi = kpt.P_aMi[a]
                else:
                    P_Mi = P_aqMi[a][kpt.q]
                dP_Mi = dPdR_aqvMi[a][kpt.q][v]
                P1HP_MM = dP_Mi.conj() @ dH_ii @ P_Mi.T
                # Matrix elements
                gp_MM += P1HP_MM + P1HP_MM.T.conjugate()
            # wfs.gd.comm.sum(gp_MM)
            # print(world.rank, a,v, kpt.k, bfs.my_atom_indices, gp_MM)
            g_sqMM[kpt.s, kpt.q] += gp_MM

        return g_sqMM

    def calculate_supercell_matrix(
        self, calc: GPAW, fd_name: str = "elph", filter: str = None
    ) -> None:
        """Calculate matrix elements of the el-ph coupling in the LCAO basis.

        This function calculates the matrix elements between LCAOs and local
        atomic gradients of the effective potential. The matrix elements are
        calculated for the supercell used to obtain finite-difference
        approximations to the derivatives of the effective potential wrt to
        atomic displacements.

        The resulting g_xsNNMM is saved into a JSON cache.

        Parameters
        ----------
        calc: GPAW
            LCAO calculator for the calculation of the supercell matrix.
        fd_name: str
            User specified name of the finite difference JSON cache.
            Default is 'elph'.
        filter: str
            Fourier filter atomic gradients of the effective potential. The
            specified components (``normal`` or ``umklapp``) are removed
            (default: None).
        """

        assert calc.wfs.mode == "lcao", "LCAO mode required."
        assert not calc.symmetry.point_group, \
            "Point group symmetry not supported"

        # JSON cache
        supercell_cache = MultiFileJSONCache(self.supercell_name)

        # Supercell atoms
        atoms_N = self.atoms * self.supercell

        # Initialize calculator if required and extract useful quantities
        if (not hasattr(calc.wfs, "S_qMM") or
            not hasattr(calc.wfs.basis_functions, "M_a")):
            calc.initialize(atoms_N)
            calc.initialize_positions(atoms_N)

        # Extract useful objects from the calculator
        wfs = calc.wfs
        gd = calc.wfs.gd
        kd = calc.wfs.kd
        bd = calc.wfs.bd
        nao = wfs.setups.nao
        nspins = wfs.nspins
        # FIXME: Domain parallelisation broken
        assert gd.comm.size == 1
        # FIXME: Band parallelisation broken - M is band parallel
        assert bd.comm.size == 1

        # Calculate finite-difference gradients (in Hartree / Bohr)
        V1t_xsG, dH1_xasp = self.calculate_gradient(fd_name, self.indices)

        # Equilibrium atomic Hamiltonian matrix (projector coefficients)
        fd_cache = MultiFileJSONCache(fd_name)
        dH_asp = fd_cache["eq"]["dH_all_asp"]

        # Check that the grid is the same as in the calculator
        assert np.all(V1t_xsG.shape[-3:] == (gd.N_c + gd.pbc_c - 1)), \
            "Mismatch in grids."

        # Save basis information, after we checked the data is kosher
        with supercell_cache.lock("basis") as handle:
            if handle is not None:
                basis_info = self.set_basis_info(calc)
                handle.save(basis_info)

        # Fourier filter the atomic gradients of the effective potential
        if filter is not None:
            for s in range(nspins):
                fourier_filter(self.atoms, self.supercell, V1t_xsG[:, s],
                               components=filter)

        if kd.gamma:
            print("WARNING: Gamma-point calculation. \
                   Overlap with neighboring cell cannot be removed")
        else:
            # Bloch to real-space converter
            tb = TightBinding(atoms_N, calc)

        # Calculate < i k | grad H | j k >, i.e. matrix elements in LCAO basis

        # Do each cartesian component separately
        for i, a in enumerate(self.indices):
            for v in range(3):
                # Corresponding array index
                xoutput = 3 * a + v
                xinput = 3 * i + v

                # If exist already, don't recompute
                with supercell_cache.lock(str(xoutput)) as handle:
                    if handle is None:
                        continue

                    parprint("%s-gradient of atom %u" %
                             (["x", "y", "z"][v], a))

                    g_sqMM = self._calculate_supercell_entry(
                        a, v, V1t_xsG[xinput], dH1_xasp[xinput], wfs, dH_asp
                    )

                    # Extract R_c=(0, 0, 0) block by Fourier transforming
                    if kd.gamma or kd.N_c is None:
                        g_sMM = g_sqMM[:, 0]
                    else:
                        # Convert to array
                        g_sMM_tmp = []
                        for s in range(nspins):
                            g_MM = tb.bloch_to_real_space(g_sqMM[s],
                                                          R_c=(0, 0, 0))
                            g_sMM_tmp.append(g_MM[0])  # [0] because of above
                        g_sMM = np.array(g_sMM_tmp)
                        del g_sMM_tmp

                    # Reshape to global unit cell indices
                    N = np.prod(self.supercell)
                    # Number of basis function in the primitive cell
                    assert (nao % N) == 0, "Alarm ...!"
                    nao_cell = nao // N
                    g_sNMNM = g_sMM.reshape((nspins, N, nao_cell, N, nao_cell))
                    g_sNNMM = g_sNMNM.swapaxes(2, 3).copy()
                    handle.save(g_sNNMM)
                if xinput == 0:
                    with supercell_cache.lock("info") as handle:
                        if handle is not None:
                            info = {
                                "sc_version": sc_version,
                                "natom": len(self.atoms),
                                "supercell": self.supercell,
                                "gshape": g_sNNMM.shape,
                                "gtype": g_sNNMM.dtype.name,
                            }
                            handle.save(info)

    def set_basis_info(self, *args) -> dict:
        """Store LCAO basis info for atoms in reference cell in attribute.

        Parameters
        ----------
        args: tuple
            If the LCAO calculator is not available (e.g. if the supercell is
            loaded from file), the ``load_supercell_matrix`` member function
            provides the required info as arguments.

        """
        assert len(args) in (1, 2)
        if len(args) == 1:
            calc = args[0]
            setups = calc.wfs.setups
            bfs = calc.wfs.basis_functions
            nao_a = [setups[a].nao for a in range(len(self.atoms))]
            M_a = [bfs.M_a[a] for a in range(len(self.atoms))]
        else:
            M_a = args[0]
            nao_a = args[1]
        return {"M_a": M_a, "nao_a": nao_a}

    @classmethod
    def calculate_gradient(cls, fd_name: str,
                           indices=None) -> Tuple[ArrayND, list]:
        """Calculate gradient of effective potential and projector coefs.

        This function loads the generated json files and calculates
        finite-difference derivatives.

        Parameters
        ----------
        fd_name: str
            name of finite difference JSON cache
        """
        cache = MultiFileJSONCache(fd_name)
        if "dr_version" not in cache["info"]:
            print("Cache created with old version. Use electronphonon.py")
            raise VersionError
        natom = cache["info"]["natom"]
        delta = cache["info"]["delta"]

        # Array and dict for finite difference derivatives
        V1t_xsG = []
        dH1_xasp = []

        if indices is None:
            indices = np.arange(natom)

        x = 0
        for a in indices:
            for v in "xyz":
                name = "%d%s" % (a, v)
                # Potential and atomic density matrix for atomic displacement
                Vtm_sG = cache[name + "-"]["Vt_sG"]
                dHm_asp = cache[name + "-"]["dH_all_asp"]
                Vtp_sG = cache[name + "+"]["Vt_sG"]
                dHp_asp = cache[name + "+"]["dH_all_asp"]

                # FD derivatives in Hartree / Bohr
                V1t_sG = (Vtp_sG - Vtm_sG) / (2 * delta / Bohr)
                V1t_xsG.append(V1t_sG)

                dH1_asp = {}
                for atom in dHm_asp.keys():
                    dH1_asp[atom] = (dHp_asp[atom] - dHm_asp[atom]) / (
                        2 * delta / Bohr
                    )
                dH1_xasp.append(dH1_asp)
                x += 1
        return np.array(V1t_xsG), dH1_xasp

    @classmethod
    def load_supercell_matrix(cls, name: str = "supercell"
                              ) -> Tuple[ArrayND, dict]:
        """Load supercell matrix from cache.

        Parameters
        ----------
        name: str
            User specified name of the cache.
        """
        # TODO: load by indices?
        supercell_cache = MultiFileJSONCache(name)
        if "sc_version" not in supercell_cache["info"]:
            print("Cache created with old version. Use electronphonon.py")
            raise VersionError
        shape = supercell_cache["info"]["gshape"]
        dtype = supercell_cache["info"]["gtype"]
        natom = supercell_cache["info"]["natom"]
        nx = natom * 3
        g_xsNNMM = np.empty([nx, ] + list(shape), dtype=dtype)
        for x in range(nx):
            g_xsNNMM[x] = supercell_cache[str(x)]
        basis_info = supercell_cache["basis"]
        return g_xsNNMM, basis_info
