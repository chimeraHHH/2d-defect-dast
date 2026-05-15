import numpy as np
import pytest

from ase.parallel import world, parprint
from ase.units import Bohr
from gpaw import GPAW
from gpaw.lcao.dipoletransition import get_dipole_transitions
from gpaw.utilities.dipole import dipole_matrix_elements_from_calc
from gpaw.lrtddft.kssingle import KSSingles


@pytest.mark.old_gpaw_only
def test_dipole_transition(gpw_files, tmp_path_factory):
    """Check dipole matrix-elements for H20."""
    calc = GPAW(gpw_files['h2o_lcao'])
    # Initialize calculator if necessary
    if not hasattr(calc.wfs, 'C_nM'):
        calc.initialize_positions(calc.atoms)
    dip_skvnm = get_dipole_transitions(calc.wfs).real
    parprint("Dipole moments calculated")
    assert dip_skvnm.shape == (1, 1, 3, 6, 6)
    dip_vnm = dip_skvnm[0, 0] * Bohr

    print(world.rank, dip_vnm[0, 0, 3])

    # check symmetry: abs(d[i,j]) == abs(d[j,i])
    for v in range(3):
        dip_vnm[v].T == pytest.approx(dip_vnm[v])

    # Check numerical value of a few elements - signs might change!
    assert 0.0693 == pytest.approx(abs(dip_vnm[2, 0, 4]), abs=1e-4)
    assert 0.1014 == pytest.approx(abs(dip_vnm[1, 0, 5]), abs=1e-4)
    assert 0.1709 == pytest.approx(abs(dip_vnm[0, 3, 4]), abs=1e-4)

    # some printout for manual inspection, if wanted
    f = 6 * "{:+.4f} "
    for c in range(3):
        for i in range(6):
            parprint(f.format(*dip_vnm[c, i]))
        parprint("")

    # ------------------------------------------------------------------------
    # compare to utilities implementation
    if world.rank == 0:
        from gpaw.new.ase_interface import GPAW as NewGPAW
        from gpaw.mpi import serial_comm
        refcalc = NewGPAW(gpw_files['h2o_lcao'],
                          communicator=serial_comm)
        uref = dipole_matrix_elements_from_calc(refcalc, 0, 6)
        uref = uref[0]
        assert uref.shape == (6, 6, 3)
    # NOTE: Comparing implementations of r gauge and v gauge is tricky, as they
    # tend to be numerically inequivalent.

    # compare to lrtddft implementation
    kss = KSSingles()
    atoms = calc.atoms
    atoms.calc = calc
    kss.calculate(calc.atoms, 1)
    lrref = []
    lrrefv = []
    for ex in kss:
        lrref.append(-1. * ex.mur * Bohr)
        lrrefv.append(-1. * ex.muv * Bohr)
    lrref = np.array(lrref)
    lrrefv = np.array(lrrefv)

    # Additional benefit: tests equivalence of r gauge implementations
    if world.rank == 0:
        for i, (m, n, v) in enumerate([[4, 0, 2],
                                       [5, 0, 1],
                                       [4, 1, 1],
                                       [5, 1, 2],
                                       [4, 2, 2],
                                       [5, 2, 1],
                                       [4, 3, 0]]):
            assert abs(lrref[i, v]) == pytest.approx(abs(uref[m, n, v]),
                                                     abs=1e-4)

    # some printout for manual inspection, if wanted
    parprint("         r-gauge   lrtddft(v)  raman(v)")
    f = "{} {:+.4f}    {:+.4f}    {:+.4f}"
    parprint(f.format('0->4 (z)', lrref[0, 2], lrrefv[0, 2], dip_vnm[2, 0, 4]))
    parprint(f.format('0->5 (y)', lrref[1, 1], lrrefv[1, 1], dip_vnm[1, 0, 5]))
    parprint(f.format('1->4 (y)', lrref[2, 1], lrrefv[2, 1], dip_vnm[1, 1, 4]))
    parprint(f.format('1->5 (z)', lrref[3, 2], lrrefv[3, 2], dip_vnm[2, 1, 5]))
    parprint(f.format('2->4 (z)', lrref[4, 2], lrrefv[4, 2], dip_vnm[2, 2, 4]))
    parprint(f.format('2->5 (y)', lrref[5, 1], lrrefv[5, 1], dip_vnm[1, 2, 5]))
    parprint(f.format('3->4 (x)', lrref[6, 0], lrrefv[6, 0], dip_vnm[0, 3, 4]))
