import pytest
from ase import Atom, Atoms

from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT
from gpaw.mpi import world


def get_H2(calculator=None):
    """Define H2 and set calculator if given"""
    R = 0.7  # approx. experimental bond length
    a = 3.0
    c = 4.0
    H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))

    if calculator is not None:
        H2.calc = calculator

    return H2


@pytest.mark.lrtddft
def test_io(in_tmp_dir):
    calc = GPAW(mode='fd', xc='PBE', h=0.25, nbands=5, txt=None)
    calc.calculate(get_H2(calc))
    exlst = LrTDDFT(calc, restrict={'eps': 0.4, 'jend': 3})
    assert len(exlst) == 3
    assert exlst.kss.restrict['eps'] == 0.4

    fname = 'lr.dat.gz'
    exlst.write(fname)
    world.barrier()

    lr2 = LrTDDFT.read(fname)
    assert len(lr2) == 3

    lr3 = LrTDDFT.read(fname, restrict={'jend': 2})
    assert len(lr3) == 2
    assert len(lr3.kss) == 2
    assert len(lr3.Om.fullkss) == 3

    lr4 = LrTDDFT.read(fname, restrict={'energy_range': 20})
    assert len(lr4) == 1


@pytest.mark.lrtddft
def test_invocation():
    calc = GPAW(xc='PBE', mode='fd', h=0.25, nbands=5, txt=None)
    H2 = get_H2(calc)
    exlst = LrTDDFT(restrict={'eps': 0.4, 'jend': 3}, txt=None)
    exlst.calculate(H2)
    assert exlst.xc is not None
    assert hasattr(exlst, 'Om')

    # traditional way
    h2 = get_H2(calc)
    h2.get_potential_energy()
    exlst2 = LrTDDFT(calc, restrict={'eps': 0.4, 'jend': 3}, txt=None)
    assert hasattr(exlst2, 'Om')
