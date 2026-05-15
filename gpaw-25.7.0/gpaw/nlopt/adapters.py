from __future__ import annotations
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gpaw.core.atom_arrays import AtomArrays
    from gpaw.core.plane_waves import PWArray
    from gpaw.new.ase_interface import ASECalculator
    from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
    from gpaw.typing import ArrayND


class GSInfo:
    """
    This is the base class for the ground state adapters in the non-linear
    optics module. It is only compatible with GPAW_NEW.

    The class should never be called directly, but should instead be called
    through the CollinearGSInfo or NoncollinearGSInfo classes.

    These subclasses are necessary due to the different ways which the spin
    index is handled in collinear and noncollinear ground state calculations.
    """
    def __init__(self,
                 calc: ASECalculator):
        assert calc.params.mode.name == 'pw', \
            'Calculator must be in plane wave mode.'

        dft = calc.dft
        self.nabla_aiiv = [setup.nabla_iiv for setup in dft.setups]

        ibzwfs = self.ibzwfs = dft.ibzwfs
        if not (ibzwfs.domain_comm.size == 1 and ibzwfs.band_comm.size == 1):
            raise ValueError('Calculator must be initialised with '
                             'only k-point parallelisation.')
        if isinstance(ibzwfs.wfs_qs[0][0].psit_nX, SimpleNamespace):
            raise ValueError('Calculator is missing wfs data. If loading from '
                             'a .gpw file, please recalculate wave functions.')

        density = dft.density
        self.collinear = density.collinear
        self.ndensities = density.ndensities

        grid = density.nt_sR.desc
        self.ucvol = np.abs(np.linalg.det(grid.cell))
        self.bzvol = np.abs(np.linalg.det(2 * np.pi * grid.icell))

    def get_plane_wave_coefficients(self,
                                    wfs: PWFDWaveFunctions,
                                    bands: slice,
                                    spin: int) -> tuple[ArrayND, ArrayND]:
        """
        Returns the plane wave coefficients and reciprocal vectors.

        Output is an array with shape (band index, reciprocal vector index)
        """
        psit_nG = wfs.psit_nX[bands]
        G_plus_k_Gv = psit_nG.desc.G_plus_k_Gv
        return G_plus_k_Gv, self._pw_data(psit_nG, spin)

    def get_wave_function_projections(self,
                                      wfs: PWFDWaveFunctions,
                                      bands: slice,
                                      spin: int):
        """
        Returns the projections of the pseudo wfs onto the partial waves.

        Output is a dictionary with atom index keys and array values with
        shape (band index, partial wave index)
        """
        return self._proj_data(wfs.P_ani, bands, spin)

    def get_wfs(self,
                wfs_s: list[PWFDWaveFunctions],
                spin: int) -> PWFDWaveFunctions:
        raise NotImplementedError

    @staticmethod
    def _pw_data(psit: PWArray,
                 spin: int) -> ArrayND:
        raise NotImplementedError

    @staticmethod
    def _proj_data(P: AtomArrays,
                   bands: slice,
                   spin: int) -> dict[int, ArrayND]:
        raise NotImplementedError


class CollinearGSInfo(GSInfo):
    def __init__(self,
                 calc: ASECalculator):
        super().__init__(calc)
        self.ns = self.ndensities

    def get_wfs(self,
                wfs_s: list[PWFDWaveFunctions],
                spin: int) -> PWFDWaveFunctions:
        return wfs_s[spin]

    @staticmethod
    def _pw_data(psit_nG: PWArray,
                 _: int | None = None) -> ArrayND:
        return psit_nG.data

    @staticmethod
    def _proj_data(P_ani: AtomArrays,
                   bands: slice,
                   _: int | None = None) -> dict[int, ArrayND]:
        return {a: P_ni[bands] for a, P_ni in P_ani.items()}


class NoncollinearGSInfo(GSInfo):
    def __init__(self,
                 calc: ASECalculator):
        super().__init__(calc)
        self.ns = 2

    def get_wfs(self,
                wfs_s: list[PWFDWaveFunctions],
                _: int | None = None) -> PWFDWaveFunctions:
        return wfs_s[0]

    @staticmethod
    def _pw_data(psit_nsG: PWArray,
                 spin: int) -> ArrayND:
        return psit_nsG.data[:, spin]

    @staticmethod
    def _proj_data(P_ansi: AtomArrays,
                   bands: slice,
                   spin: int) -> dict[int, ArrayND]:
        return {a: P_nsi[bands, spin] for a, P_nsi in P_ansi.items()}
