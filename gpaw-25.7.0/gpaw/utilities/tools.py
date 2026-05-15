import numpy as np

from ase.units import Hartree, Bohr


def split_formula(formula):
    """Count elements in a chemical formula.

    E.g. split_formula('C2H3Mg') -> ['C', 'C', 'H', 'H', 'H', 'Mg']
    """
    res = []
    for c in formula:
        if c.isupper():
            res.append(c)
        elif c.islower():
            res[-1] += c
        else:
            res.extend([res[-1]] * (eval(c) - 1))
    return res


def construct_reciprocal(gd, q_c=None):
    """Construct the reciprocal lattice from ``GridDescriptor`` instance.

    The generated reciprocal lattice has lattice vectors corresponding to the
    real-space lattice defined in the input grid. Note that it is the squared
    length of the reciprocal lattice vectors that are returned.

    The ordering of the reciprocal lattice agrees with the one typically used
    in fft algorithms, i.e. positive k-values followed by negative.

    Note that the G=(0,0,0) entry is set to one instead of zero. This
    bit should probably be moved somewhere else ...

    Parameters
    ----------
    q_c: ndarray
        Offset for the reciprocal lattice vectors (in scaled coordinates of the
        reciprocal lattice vectors, i.e. array with index ``c``). When
        specified, the returned array contains the values of (q+G)^2 where G
        denotes the reciprocal lattice vectors.

    """

    assert gd.pbc_c.all(), 'Works only with periodic boundary conditions!'

    q_c = np.zeros((3, 1), dtype=float) if q_c is None else q_c.reshape((3, 1))

    # Calculate reciprocal lattice vectors
    N_c1 = gd.N_c[:, None]
    i_cq = np.indices(gd.n_c, dtype=float).reshape((3, -1))  # offsets....
    i_cq += gd.beg_c[:, None]
    i_cq += N_c1 // 2
    i_cq %= N_c1
    i_cq -= N_c1 // 2

    i_cq += q_c

    # Convert from scaled to absolute coordinates
    B_vc = 2.0 * np.pi * gd.icell_cv.T
    k_vq = np.dot(B_vc, i_cq)

    k_vq *= k_vq
    k2_Q = k_vq.sum(axis=0).reshape(gd.n_c)

    # Avoid future divide-by-zero by setting k2_Q[G=(0,0,0)] = 1.0 if needed
    if k2_Q[0, 0, 0] < 1e-10:
        k2_Q[0, 0, 0] = 1.0           # Only make sense iff
        assert gd.comm.rank == 0      # * on rank 0 (G=(0,0,0) is only there)
        assert abs(q_c).sum() < 1e-8  # * q_c is (almost) zero

    assert k2_Q.min() > 0.0       # Now there should be no zero left

    # Determine N^3
    #
    # Why do we need to calculate and return this?  The caller already
    # has access to gd and thus knows how many points there are.
    N3 = gd.n_c[0] * gd.n_c[1] * gd.n_c[2]
    return k2_Q, N3


def coordinates(gd, origin=None, tiny=1e-12):
    """Constructs and returns matrices containing cartesian coordinates,
       and the square of the distance from the origin.

       The origin can be given explicitely (in Bohr units, not Anstroms).
       Otherwise the origin is placed in the center of the box described
       by the given grid-descriptor 'gd'.
    """

    if origin is None:
        origin = 0.5 * gd.cell_cv.sum(0)
    r0_v = np.array(origin)

    r_vG = gd.get_grid_point_distance_vectors(r0_v)
    r2_G = np.sum(r_vG**2, axis=0)
    # Remove singularity at origin and replace with small number
    r2_G = np.where(r2_G < tiny, tiny, r2_G)

    # Return r^2 matrix
    return r_vG, r2_G


def pick(a_ix, i):
    """Take integer index of a, or a linear combination of the elements of a"""
    if isinstance(i, int):
        return a_ix[i]
    shape = a_ix.shape
    a_x = np.dot(i, a_ix[:].reshape(shape[0], -1))
    return a_x.reshape(shape[1:])


def dagger(a, copy=True):
    """Return Hermitian conjugate of input

    If copy is False, the original array might be overwritten. This is faster,
    but use with care.
    """
    if copy:
        return np.conj(a.T)
    else:
        a = a.T
        if a.dtype == complex:
            a.imag *= -1
        return a


def project(a, b):
    """Return the projection of b onto a."""
    return a * (np.dot(a.conj(), b) / np.linalg.norm(a))


def normalize(U):
    """Normalize columns of U."""
    for col in U.T:
        col /= np.linalg.norm(col)


def gram_schmidt(U):
    """Orthonormalize columns of U according to the Gram-Schmidt procedure."""
    for i, col in enumerate(U.T):
        for col2 in U.T[:i]:
            col -= col2 * np.dot(col2.conj(), col)
        col /= np.linalg.norm(col)


def lowdin(U, S=None):
    """Orthonormalize columns of U according to the Lowdin procedure.

    If the overlap matrix is know, it can be specified in S.
    """
    if S is None:
        S = np.dot(dagger(U), U)
    eig, rot = np.linalg.eigh(S)
    rot = np.dot(rot / np.sqrt(eig), dagger(rot))
    U[:] = np.dot(U, rot)


def symmetrize(matrix):
    """Symmetrize input matrix."""
    np.add(dagger(matrix), matrix, matrix)
    np.multiply(.5, matrix, matrix)
    return matrix


def tri2full(H_nn, UL='L', map=np.conj):
    """Fill in values of hermitian or symmetric matrix.

    Fill values in lower or upper triangle of H_nn based on the opposite
    triangle, such that the resulting matrix is symmetric/hermitian.

    UL='U' will copy (conjugated) values from upper triangle into the
    lower triangle.

    UL='L' will copy (conjugated) values from lower triangle into the
    upper triangle.

    The map parameter can be used to specify a different operation than
    conjugation, which should work on 1D arrays.  Example::

      def antihermitian(src, dst):
            np.conj(-src, dst)

      tri2full(H_nn, map=antihermitian)

    """
    N, tmp = H_nn.shape
    assert N == tmp, 'Matrix must be square'
    if UL != 'L':
        H_nn = H_nn.T

    for n in range(N - 1):
        map(H_nn[n + 1:, n], H_nn[n, n + 1:])


def cutoff2gridspacing(E):
    """Convert planewave energy cutoff to a real-space gridspacing."""
    return np.pi / np.sqrt(2 * E / Hartree) * Bohr
