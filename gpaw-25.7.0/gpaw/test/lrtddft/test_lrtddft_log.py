import pytest
from ase import Atom, Atoms
from ase.parallel import world

from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT


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


def run_and_delete(txt):
    outfname = 'gpawlog.txt'
    calc = GPAW(mode='fd', xc='PBE', h=0.25, nbands=5, txt=outfname)
    calc.calculate(get_H2(calc))
    exlst = LrTDDFT(calc, restrict={'eps': 0.4, 'jend': 3}, txt=txt)
    del calc
    del exlst
    world.barrier()


@pytest.mark.lrtddft
def test_log(in_tmp_dir):
    defname = 'gpawlog.txt'
    # LrTDDFT outputs to the same log like gpaw
    run_and_delete(txt=defname)
    if world.rank == 0:
        with open(defname) as f:
            string = f.read()
            assert 'Kohn-Sham single transitions' in string
            assert 'Linear response TDDFT calculation' in string
    world.barrier()

    # silent LrTDDFT
    run_and_delete(txt=None)
    if world.rank == 0:
        with open(defname) as f:
            assert 'RPA kss[0]' not in f.read()
    world.barrier()

    # output to own file
    ownfname = 'lrtddftlog.txt'
    run_and_delete(txt=ownfname)

    if world.rank == 0:
        with open(defname) as f:
            assert 'Linear response TDDFT calculation' not in f.read()
        with open(ownfname) as f:
            string = f.read()
            assert 'Kohn-Sham single transitions' in string
            assert 'Linear response TDDFT calculation' in string
    world.barrier()
