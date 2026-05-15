import numpy as np
from ase import Atom, Atoms

import gpaw.mpi as mpi
from gpaw import GPAW, FermiDirac, PoissonSolver
import pytest
from gpaw.utilities.dos import RawLDOS, raw_orbital_LDOS, raw_wignerseitz_LDOS


def test_utilities_ldos(in_tmp_dir, gpaw_new):
    comms = [mpi.world.new_communicator(np.array([r]))
             for r in range(mpi.size)]
    comm = comms[mpi.rank]

    Hnospin = Atoms([Atom('H')], cell=[5, 5, 5], pbc=False)
    Hspin = Atoms([Atom('H', magmom=1)], cell=[5, 5, 5], pbc=False)
    LiH = Atoms([Atom('Li', [.0, .0, .41]),
                 Atom('H', [.0, .0, -1.23])])
    Hnospin.center()
    Hspin.center()
    LiH.center(vacuum=3.0)

    # This is needed for the Wigner-Seitz test to give
    # architecture-independent results:
    LiH.translate(0.003234)

    calc = GPAW(mode='fd', gpts=(24, 24, 24), communicator=comm)
    Hnospin.calc = calc
    e_Hnospin = Hnospin.get_potential_energy()
    energies, sweight = raw_orbital_LDOS(calc, a=0, spin=0, angular='s')
    energies, pdfweight = raw_orbital_LDOS(calc, a=0, spin=0, angular='pdf')

    calc = GPAW(mode='fd',
                gpts=(24, 24, 24),
                occupations=FermiDirac(width=0, fixmagmom=True),
                poissonsolver=PoissonSolver('fd'),
                hund=True,
                communicator=comm)
    Hspin.calc = calc
    e_Hspin = Hspin.get_potential_energy()
    energies, sweight_spin = raw_orbital_LDOS(calc, a=0, spin=0, angular='s')

    calc = GPAW(mode='fd', gpts=(32, 32, 40), nbands=2,
                poissonsolver=PoissonSolver('fd'),
                communicator=comm)
    LiH.calc = calc
    e_LiH = LiH.get_potential_energy()
    energies, Li_orbitalweight = raw_orbital_LDOS(calc, a=0, spin=0,
                                                  angular=None)
    energies, H_orbitalweight = raw_orbital_LDOS(calc, a=1, spin=0,
                                                 angular=None)
    energies, Li_wzweight = raw_wignerseitz_LDOS(calc, a=0, spin=0)
    energies, H_wzweight = raw_wignerseitz_LDOS(calc, a=1, spin=0)

    if not gpaw_new:
        n_a = calc.get_wigner_seitz_densities(spin=0)
        print(n_a)
        assert n_a.sum() == pytest.approx(0.0, abs=1e-5)
        assert n_a[1] == pytest.approx(0.737, abs=0.001)

    print(sweight, pdfweight)
    print(sweight_spin)
    print(Li_wzweight)
    print(H_wzweight)

    assert sweight[0] == pytest.approx(1.0, abs=0.06)
    assert pdfweight[0] == pytest.approx(0.0, abs=0.0001)
    assert sweight_spin[0] == pytest.approx(1.14, abs=0.06)
    assert ((Li_wzweight - [.13, 0.93]).round(2) == 0).all()
    assert ((H_wzweight - [0.87, 0.07]).round(2) == 0).all()
    assert ((Li_wzweight + H_wzweight).round(5) == 1).all()

    print(Li_orbitalweight)
    print(H_orbitalweight)
    #               HOMO    s   py  pz  px  *s
    Li_orbitalweight[0] -= [.5, .0, .6, .0, .0]
    H_orbitalweight[0] -= [.7, .0, .0, .0, .0]

    #              LUMO       s  py   pz  px  *s
    Li_orbitalweight[1] -= [1.0, .0, 0.9, .0, .0]
    H_orbitalweight[1] -= [0.1, .0, 0.0, .0, .0]

    assert not Li_orbitalweight.round(1).any()
    assert not H_orbitalweight.round(1).any()

    ldos = RawLDOS(calc)
    fname = 'ldbe'
    ldos.by_element_to_file(fname + '.dat', shift=False)
    ldos.by_element_to_file(fname + '_2.0.dat', 2.0, shift=False)
    ldos.by_element_to_file(fname + '_indx0.dat', indices=[0])
    # the hydrogen entries are missing for index 0 only
    if mpi.world.rank == 0:
        assert (np.loadtxt(fname + '_indx0.dat').shape[1] + 3 ==
                np.loadtxt(fname + '.dat').shape[1])

    energy_tolerance = 0.001
    assert e_Hnospin == pytest.approx(0.153991, abs=energy_tolerance)
    assert e_Hspin == pytest.approx(-0.782309, abs=energy_tolerance)
    assert e_LiH == pytest.approx(-3.74582, abs=energy_tolerance)
