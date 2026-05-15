import numpy as np
import numpy.fft as fft
import numpy.linalg as la

from ase import Atoms
import ase.units as units

from gpaw.typing import ArrayND


def fourier_filter(atoms: Atoms, supercell: tuple, V1t_xG: ArrayND,
                   components: str, criteria=0) -> None:
    """Fourier filter atomic gradients of the effective potential.

    ... or any other property defined the same way.

    This method is not tested.

    Parameters
    ----------
    atoms: Atoms
        Atoms object, primitive cell
    supercell: tuple
        Size of supercell given by the number of repetitions (l, m, n) of
        the small unit cell in each direction.
    V1t_xG: ndarray
        Array representation of atomic gradients of the effective potential
        in the supercell grid.
    components: str
        Fourier components to filter out (``normal`` or ``umklapp``).
    """
    assert components in ['normal', 'umklapp']
    # Grid shape
    shape = V1t_xG.shape[-3:]

    # Primitive unit cells in Bohr/Bohr^-1
    cell_cv = atoms.get_cell() / units.Bohr
    reci_vc = 2 * np.pi * la.inv(cell_cv)
    norm_c = np.sqrt(np.sum(reci_vc**2, axis=0))
    # Periodic BC array
    pbc_c = np.array(atoms.get_pbc(), dtype=bool)

    # Supercell atoms and cell
    atoms_N = atoms * supercell
    supercell_cv = atoms_N.get_cell() / units.Bohr

    # q-grid in units of the grid spacing (FFT ordering)
    q_cG = np.indices(shape).reshape(3, -1)
    q_c = np.array(shape)[:, np.newaxis]
    q_cG += q_c // 2
    q_cG %= q_c
    q_cG -= q_c // 2

    # Locate q-points inside the Brillouin zone
    if criteria == 0:
        # Works for all cases
        # Grid spacing in direction of reciprocal lattice vectors
        h_c = np.sqrt(np.sum((2 * np.pi * la.inv(supercell_cv))**2,
                             axis=0))
        # XXX Why does a "*=" operation on q_cG not work here ??
        q1_cG = q_cG * h_c[:, np.newaxis] / (norm_c[:, np.newaxis] / 2)
        mask_G = np.ones(np.prod(shape), dtype=bool)
        for i, pbc in enumerate(pbc_c):
            if not pbc:
                continue
            mask_G &= (-1. < q1_cG[i]) & (q1_cG[i] <= 1.)
    else:
        # 2D hexagonal lattice
        # Projection of q points onto the periodic directions. Only in
        # these directions do normal and umklapp processees make sense.
        q_vG = np.dot(q_cG[pbc_c].T,
                      2 * np.pi * la.inv(supercell_cv).T[pbc_c]).T.copy()
        # Parametrize the BZ boundary in terms of the angle theta
        theta_G = np.arctan2(q_vG[1], q_vG[0]) % (np.pi / 3)
        phi_G = np.pi / 6 - np.abs(theta_G)
        qmax_G = norm_c[0] / 2 / np.cos(phi_G)
        norm_G = np.sqrt(np.sum(q_vG**2, axis=0))
        # Includes point on BZ boundary with +1e-2
        mask_G = (norm_G <= qmax_G + 1e-2)

    if components == 'umklapp':
        mask_G = ~mask_G

    # Reshape to grid shape
    mask_G.shape = shape

    for V1t_G in V1t_xG:
        # Fourier transform atomic gradient
        V1tq_G = fft.fftn(V1t_G)
        # Zero normal/umklapp components
        V1tq_G[mask_G] = 0.0
        # Fourier transform back
        V1t_G[:] = fft.ifftn(V1tq_G).real
