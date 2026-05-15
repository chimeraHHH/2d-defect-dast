"""Contains methods for calculating local LR-TDDFT kernels."""

from functools import partial
from pathlib import Path

import numpy as np

from gpaw.response import timer
from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.dyson import PWKernel
from gpaw.response.localft import (LocalFTCalculator,
                                   add_LDA_dens_fxc, add_LSDA_trans_fxc)


class FXCKernel(PWKernel):
    r"""Adiabatic local exchange-correlation kernel in a plane-wave basis.

    In real-space, the adiabatic local xc-kernel matrix is given by:

    K_xc^μν(r, r', t-t') = f_xc^μν(r) δ(r-r') δ(t-t')

    where the local xc kernel is given by:

                 ∂^2[ϵ_xc(n,m)n] |
    f_xc^μν(r) = ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ |
                   ∂n^μ ∂n^ν     |n=n(r),m=m(r)

    In the plane-wave basis (and frequency domain),

                     1
    K_xc^μν(G, G') = ‾‾ f_xc^μν(G-G'),
                     V0

    where V0 is the cell volume.

    Because the set of unique reciprocal lattice vector differences
    dG = G-G' is much more compact than the full kernel matrix structure
    Kxc_GG', we store data internally in the dG representation and only
    unfold the data into the full kernel matrix when requested.
    """

    def __init__(self, fxc_dG, dG_K, GG_shape, volume):
        """Construct the fxc kernel."""
        assert np.prod(GG_shape) == len(dG_K), \
            "The K index should be a flattened (G,G') composite index'"

        self._fxc_dG = fxc_dG
        self._dG_K = dG_K
        self.GG_shape = GG_shape
        self.volume = volume

    def get_number_of_plane_waves(self):
        assert self.GG_shape[0] == self.GG_shape[1]
        return self.GG_shape[0]

    def _add_to(self, x_GG):
        """Add Kxc_GG to input array."""
        x_GG[:] += self.get_Kxc_GG()

    def get_Kxc_GG(self):
        """Unfold the fxc(G-G') kernel into the Kxc_GG' kernel matrix."""
        # Kxc(G-G') = 1 / V0 * fxc(G-G')
        Kxc_dG = 1 / self.volume * self._fxc_dG

        # Unfold Kxc(G-G') to the kernel matrix structure Kxc_GG'
        Kxc_GG = Kxc_dG[self._dG_K].reshape(self.GG_shape)

        return Kxc_GG

    def save(self, path: Path):
        """Save the fxc kernel in a .npz file."""
        assert path.suffix == '.npz'
        with open(str(path), 'wb') as fd:
            np.savez(fd,
                     fxc_dG=self._fxc_dG,
                     dG_K=self._dG_K,
                     GG_shape=self.GG_shape,
                     volume=self.volume)

    @staticmethod
    def from_file(path: Path):
        """Construct an fxc kernel from a previous calculation."""
        assert path.suffix == '.npz'
        npzfile = np.load(path)

        args = [npzfile[key]
                for key in ['fxc_dG', 'dG_K', 'GG_shape', 'volume']]

        return FXCKernel(*args)


class AdiabaticFXCCalculator:
    """Calculator for adiabatic local exchange-correlation kernels."""

    def __init__(self, localft_calc: LocalFTCalculator):
        """Contruct the fxc calculator based on a local FT calculator."""
        self.localft_calc = localft_calc

        self.gs = localft_calc.gs
        self.context = localft_calc.context

    @staticmethod
    def from_rshe_parameters(*args, **kwargs):
        return AdiabaticFXCCalculator(
            LocalFTCalculator.from_rshe_parameters(*args, **kwargs))

    @timer('Calculate XC kernel')
    def __call__(self, fxc, spincomponent, qpd):
        """Calculate fxc(G-G'), which defines the kernel matrix Kxc_GG'.

        The fxc kernel is calculated for all unique dG = G-G' reciprocal
        lattice vectors and returned as an FXCKernel instance which can unfold
        itself to produce the full kernel matrix Kxc_GG'.
        """
        # Generate a large_qpd to encompass all G-G' in qpd
        large_ecut = 4 * qpd.ecut  # G = 1D grid of |G|^2/2 < ecut
        large_qpd = qpd.copy_with(ecut=large_ecut,
                                  gammacentered=True,
                                  gd=self.gs.finegd)

        # Calculate fxc(Q) on the large plane-wave grid (Q = large grid index)
        add_fxc = create_add_fxc(fxc, spincomponent)
        fxc_Q = self.localft_calc(large_qpd, add_fxc)

        # Create a mapping from the large plane-wave grid to an unfoldable mesh
        # of all unique dG = G-G' reciprocal lattice vector differences on the
        # qpd plane-wave representation
        GG_shape, dG_K, Q_dG = self.create_unfoldable_Q_dG_mapping(
            qpd, large_qpd)

        # Map the calculated kernel fxc(Q) onto the unfoldable grid of unique
        # reciprocal lattice vector differences fxc(dG)
        fxc_dG = fxc_Q[Q_dG]

        # Return the calculated kernel as an fxc kernel object
        fxc_kernel = FXCKernel(fxc_dG, dG_K, GG_shape, qpd.gd.volume)

        return fxc_kernel

    @timer('Create unfoldable Q_dG mapping')
    def create_unfoldable_Q_dG_mapping(self, qpd, large_qpd):
        """Create mapping from Q index to the kernel matrix indeces GG'.

        The mapping is split into two parts:
         * Mapping from the large plane-wave representation index Q (of
           large_qpd) to an index dG representing all unique reciprocal lattice
           vector differences (G-G') of the original plane-wave representation
           qpd
         * A mapping from the dG index to a flattened K = (G, G') composite
           kernel matrix index

        Lastly the kernel matrix shape GG_shape is returned so that an array in
        index K can easily be reshaped to a kernel matrix Kxc_GG'.
        """
        # Calculate all (G-G') reciprocal lattice vectors
        dG_GGv = calculate_dG_GGv(qpd)
        GG_shape = dG_GGv.shape[:2]

        # Reshape to composite K = (G, G') index
        dG_Kv = dG_GGv.reshape(-1, dG_GGv.shape[-1])

        # Find unique dG-vectors
        # We need tight control of the decimals to avoid precision artifacts
        dG_dGv, dG_K = np.unique(dG_Kv.round(decimals=6),
                                 return_inverse=True, axis=0)

        # Create the mapping from Q-index to dG-index
        Q_dG = self.create_Q_dG_map(large_qpd, dG_dGv)

        return GG_shape, dG_K, Q_dG

    @timer('Create Q_dG map')
    def create_Q_dG_map(self, large_qpd, dG_dGv):
        """Create mapping between (G-G') index dG and large_qpd index Q."""
        G_Qv = large_qpd.get_reciprocal_vectors(add_q=False)
        # Make sure to match the precision of dG_dGv
        G_Qv = G_Qv.round(decimals=6)

        # Distribute dG over world
        # This is necessary because the next step is to create a K_QdGv buffer
        # of which the norm is taken. When the number of plane-wave
        # coefficients is large, this step becomes a memory bottleneck, hence
        # the distribution.
        dGblocks = Blocks1D(self.context.comm, dG_dGv.shape[0])
        dG_mydGv = dG_dGv[dGblocks.myslice]

        # Determine Q index for each dG index
        diff_QmydG = np.linalg.norm(G_Qv[:, np.newaxis] - dG_mydGv[np.newaxis],
                                    axis=2)
        Q_mydG = np.argmin(diff_QmydG, axis=0)

        # Check that all the identified Q indices produce identical reciprocal
        # lattice vectors
        assert np.allclose(np.diagonal(diff_QmydG[Q_mydG]), 0.), \
            'Could not find a perfect matching reciprocal wave vector in '\
            'large_qpd for all dG_dGv'

        # Collect the global Q_dG map
        Q_dG = dGblocks.all_gather(Q_mydG)

        return Q_dG


def create_add_fxc(fxc: str, spincomponent: str):
    """Create an add_fxc function according to the requested functional and
    spin component."""
    assert fxc in ['ALDA_x', 'ALDA_X', 'ALDA']

    if spincomponent in ['00', 'uu', 'dd']:
        add_fxc = partial(add_LDA_dens_fxc, fxc=fxc)
    elif spincomponent in ['+-', '-+']:
        add_fxc = partial(add_LSDA_trans_fxc, fxc=fxc)
    else:
        raise ValueError(spincomponent)

    return add_fxc


def calculate_dG_GGv(qpd):
    """Calculate dG_GG' = (G-G') for the plane wave basis in qpd."""
    nG = qpd.ngmax
    G_Gv = qpd.get_reciprocal_vectors(add_q=False)

    dG_GGv = np.zeros((nG, nG, 3))
    for v in range(3):
        dG_GGv[:, :, v] = np.subtract.outer(G_Gv[:, v], G_Gv[:, v])

    return dG_GGv
