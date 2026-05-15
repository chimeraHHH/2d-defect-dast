import pytest
from ase import Atom, Atoms
from ase.parallel import parprint

from gpaw import GPAW, mpi
from gpaw.lrtddft import LrTDDFT
from gpaw.poisson import FDPoissonSolver
from gpaw.pes.dos import DOSPES
from gpaw.pes.tddft import TDDFTPES


@pytest.mark.lrtddft
def test_lrtddft_pes(in_tmp_dir):
    txt = None
    R = 0.7  # approx. experimental bond length
    a = 3.0
    c = 3.0
    H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))

    H2_plus = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                     Atom('H', (a / 2, a / 2, (c + R) / 2))],
                    cell=(a, a, c))

    xc = 'LDA'
    calc = GPAW(mode='fd', gpts=(12, 12, 12), xc=xc, nbands=1,
                poissonsolver=FDPoissonSolver(),
                parallel={'domain': mpi.world.size},
                spinpol=True, txt=txt)
    H2.calc = calc
    e_H2 = H2.get_potential_energy()

    calc_plus = GPAW(mode='fd', gpts=(12, 12, 12), xc=xc, nbands=2,
                     poissonsolver=FDPoissonSolver(),
                     parallel={'domain': mpi.world.size},
                     spinpol=True, txt=txt)
    calc_plus = calc_plus.new(charge=+1)
    H2_plus.calc = calc_plus
    e_H2_plus = H2_plus.get_potential_energy()

    out = 'dospes.dat'
    pes = DOSPES(calc, calc_plus, shift=True)
    pes.save_folded_pes(filename=out, folding=None)
    parprint('DOS:')
    pes.save_folded_pes(filename=None, folding=None)

    # check for correct shift
    VDE = calc_plus.get_potential_energy() - calc.get_potential_energy()
    BE_HOMO = 1.e23
    be_n, f_n = pes.get_energies_and_weights()
    for be, f in zip(be_n, f_n):
        if f > 0.1 and be < BE_HOMO:
            BE_HOMO = be
    assert BE_HOMO == pytest.approx(VDE, abs=0.0)

    lr = LrTDDFT(calc_plus, xc=xc)

    out = 'lrpes.dat'
    pes = TDDFTPES(calc, lr)
    pes.save_folded_pes(filename=out, folding='Gauss')
    parprint('Linear response:')
    pes.save_folded_pes(filename=None, folding=None)

    energy_tolerance = 0.001
    assert e_H2 == pytest.approx(-3.90059, abs=energy_tolerance)
    assert e_H2_plus == pytest.approx(10.5659703, abs=energy_tolerance)

    # io
    out = 'lrpes.dat.gz'
    lr.write(out)
    lr = LrTDDFT.read(out)
    lr.calculator = calc_plus

    pes = TDDFTPES(calc, lr)
    parprint('Linear response:')
    pes.save_folded_pes(filename=None, folding=None)
