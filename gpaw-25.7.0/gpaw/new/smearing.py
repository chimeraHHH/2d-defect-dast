from __future__ import annotations
from gpaw.occupations import create_occ_calc, ParallelLayout
from gpaw.band_descriptor import BandDescriptor
from gpaw.typing import ArrayLike2D, Array2D


class OccupationNumberCalculator:
    def __init__(self,
                 dct,
                 pbc,
                 ibz,
                 nbands,
                 comms,
                 magmom_v,
                 ncomponents,
                 nelectrons,
                 rcell):
        if not dct:
            if pbc.any():
                dct = {'name': 'fermi-dirac',
                       'width': 0.1}  # eV
            else:
                dct = {'width': 0.0}

        if dct.get('fixmagmom'):
            if ncomponents == 1:
                dct = dct.copy()
                del dct['fixmagmom']
            assert ncomponents == 2

        kwargs = dct.copy()
        name = kwargs.pop('name', '')
        assert name != 'mom'

        bd = BandDescriptor(nbands)  # dummy
        # Note that eigenvalues are not distributed over
        # the band communicator.
        self.occ = create_occ_calc(
            dct,
            parallel_layout=ParallelLayout(bd,
                                           comms['k'],
                                           comms['K']),
            nbands=nbands,
            nkpts=len(ibz),
            nelectrons=nelectrons,
            nspins=ncomponents % 3,
            fixed_magmom_value=magmom_v[2],
            rcell=rcell,
            monkhorst_pack_size=getattr(ibz.bz, 'size_c', None),
            bz2ibzmap=ibz.bz2ibz_K)
        self.extrapolate_factor = self.occ.extrapolate_factor

    def __str__(self):
        return str(self.occ)

    def _set_nbands(self, nbands):
        bd, kpt_comm, domain_comm = self.occ.parallel_layout
        self.occ = self.occ.copy(
            parallel_layout=ParallelLayout(BandDescriptor(nbands),
                                           kpt_comm, domain_comm))

    def calculate(self,
                  nelectrons: float,
                  eigenvalues: ArrayLike2D,
                  weights: list[float],
                  fermi_levels_guess: list[float] = None,
                  fix_fermi_level: bool = False
                  ) -> tuple[Array2D, list[float], float]:
        occs, fls, e = self.occ.calculate(nelectrons, eigenvalues, weights,
                                          fermi_levels_guess, fix_fermi_level)
        return occs, fls, e

    def initialize_reference_orbitals(self):
        try:
            self.occ.initialize_reference_orbitals()
        except AttributeError:
            pass
