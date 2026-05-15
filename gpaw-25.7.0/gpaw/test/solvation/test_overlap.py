import pytest
from ase.build import molecule

from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.analyse.overlap import Overlap
import gpaw.solvation as solv
from gpaw.lrtddft import LrTDDFT
from gpaw.poisson import PoissonSolver


@pytest.mark.skip(reason='TODO')
def test_solvation_overlap():
    """Check whether LrTDDFT in solvation works"""

    h = 0.4
    box = 2
    nbands = 2

    H2 = molecule('H2')
    adjust_cell(H2, box, h)

    c1 = GPAW(mode='fd', h=h, txt=None, nbands=nbands)
    c1.calculate(H2)

    c2 = solv.SolvationGPAW(mode='fd',
                            h=h,
                            txt=None,
                            nbands=nbands + 1,
                            **solv.get_HW14_water_kwargs())
    c2.calculate(H2)
    for poiss in [None, PoissonSolver(nn=c2.hamiltonian.poisson.nn)]:
        lr = LrTDDFT(c2, poisson=poiss)
        print(lr)
    print(Overlap(c1).pseudo(c2))
