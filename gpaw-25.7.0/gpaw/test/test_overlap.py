import numpy as np
import pytest
from ase.build import molecule
from ase.parallel import parprint, world

from gpaw import GPAW
from gpaw.analyse.overlap import Overlap
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.lrtddft import LrTDDFT

"""Evaluate the overlap between two independent calculations

Differences are forced by different eigensolvers and differing number
of Kohn-Sham states.
"""


@pytest.mark.old_gpaw_only
def test_overlap(in_tmp_dir):

    def get_kwargs(nbands, **kwargs) -> dict:
        base_kwargs = dict(
            mode='fd',
            h=h,
            txt=txt,
            nbands=nbands,
            convergence={'eigenstates': nbands})
        return dict(base_kwargs, **kwargs)

    h = 0.4
    box = 2
    nbands = 4
    txt = '-'
    txt = None

    H2 = molecule('H2')
    adjust_cell(H2, box, h)

    c1 = GPAW(**get_kwargs(nbands=nbands))
    c1.calculate(H2)
    lr1 = LrTDDFT(c1)

    parprint('sanity --------')
    ov = Overlap(c1).pseudo(c1)
    parprint('pseudo(normalized):\n', ov)
    ov = Overlap(c1).pseudo(c1, False)
    parprint('pseudo(not normalized):\n', ov)
    ov = Overlap(c1).full(c1)
    parprint('full:\n', ov)
    assert ov[0] == pytest.approx(np.eye(ov[0].shape[0], dtype=ov.dtype),
                                  abs=1e-10)

    def show(c2):
        c2.calculate(H2)
        ov = Overlap(c1).pseudo(c2)
        parprint('wave function overlap (pseudo):\n', ov)
        ov = Overlap(c1).full(c2)
        parprint('wave function overlap (full):\n', ov)
        lr2 = LrTDDFT(c2)
        ovkss = lr1.kss.overlap(ov[0], lr2.kss)
        parprint('KSSingles overlap:\n', ovkss)
        ovlr = lr1.overlap(ov[0], lr2)
        parprint('LrTDDFT overlap:\n', ovlr)

    parprint('cg --------')
    c2 = GPAW(**get_kwargs(eigensolver='cg', nbands=nbands + 1))
    show(c2)

    parprint('spin --------')
    H2.set_initial_magnetic_moments([1, -1])
    c2 = GPAW(
        **get_kwargs(
            spinpol=True, nbands=nbands + 1, parallel={'domain': world.size}))
    H2.set_initial_magnetic_moments([0, 0])
    try:
        show(c2)
    except AssertionError:
        parprint('Not ready')
    if 1:
        ov = Overlap(c1).pseudo(c2, otherspin=1)
        parprint('wave function overlap (pseudo other spin):\n', ov)

    parprint('k-points --------')

    H2.set_pbc([1, 1, 1])
    c1 = GPAW(**get_kwargs(nbands=nbands, kpts=(1, 1, 3)))
    c1.calculate(H2)
    c2 = GPAW(**get_kwargs(nbands=nbands + 1, kpts=(1, 1, 3)))
    try:
        show(c2)
    except (AssertionError, IndexError) as e:
        parprint('Not ready', e)
