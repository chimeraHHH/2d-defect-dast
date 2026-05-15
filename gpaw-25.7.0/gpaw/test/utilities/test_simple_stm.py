import pytest
from ase import Atoms

from gpaw import GPAW, FermiDirac
from gpaw.analyse.simple_stm import SimpleStm
from gpaw.mpi import rank, size


def test_utilities_simple_stm(in_tmp_dir):
    load = False
    txt = '/dev/null'
    txt = '-'

    me = ''
    if size > 1:
        me += 'rank ' + str(rank) + ': '

    BH = Atoms('BH', [[.0, .0, .41], [.0, .0, -1.23]],
               cell=[5, 6, 6.5])
    BH.center()

    f3dname = 'stm3d.cube'

    def testSTM(calc):
        stm = SimpleStm(calc)
        stm.write_3D([1, 0, 0], f3dname)  # single wf
        wf = stm.gd.integrate(stm.ldos)

        if size == 1:  # XXXX might have trouble reading in parallel
            stm2 = SimpleStm(f3dname)
            wf2 = stm2.gd.integrate(stm2.ldos)
            print('Integrals: written, read=', wf, wf2)
            assert wf == pytest.approx(wf2, abs=2.e-6)

        stm.scan_const_current(0.02, 5)
    #    print eigenvalue_string(calc)
        stm.write_3D(3.1, f3dname)
        wf2 = stm.gd.integrate(stm.ldos)
    #    print "wf2=", wf2
        assert wf2 == pytest.approx(2, abs=0.12)

        return wf

    # finite system without spin and width
    fname = 'BH-nospin_wfs.gpw'
    if not load:
        BH.set_pbc(False)
        cf = GPAW(mode='fd', nbands=3, h=.3, txt=txt)
        BH.calc = cf
        e1 = BH.get_potential_energy()
        cf.write(fname, 'all')
    else:
        cf = GPAW(fname, txt=txt)
    wf = testSTM(cf)

    # finite system with spin
    fname = 'BH-spin_Sz2_wfs.gpw'
    BH.set_initial_magnetic_moments([1, 1])
    if not load:
        BH.set_pbc(False)
        cf = GPAW(mode='fd',
                  occupations=FermiDirac(0.1, fixmagmom=True),
                  nbands=5,
                  h=0.3,
                  txt=txt)
        BH.calc = cf
        e2 = BH.get_potential_energy()
        cf.write(fname, 'all')
    else:
        cf = GPAW(fname, txt=txt)
    testSTM(cf)

    # periodic system
    if not load:
        BH.set_pbc(True)
        cp = GPAW(mode='fd', spinpol=True, nbands=3, h=.3,
                  kpts=(2, 1, 1), txt=txt, symmetry='off')
        BH.calc = cp
        e3 = BH.get_potential_energy()
        cp.write('BH-4kpts_wfs.gpw', 'all')
    else:
        cp = GPAW('BH-4kpts_wfs.gpw', txt=txt)

    stmp = SimpleStm(cp)

    stmp.write_3D(-4., f3dname)
    print(me + 'Integrals(occ): 2 * wf, bias=', 2 * wf,
          stmp.gd.integrate(stmp.ldos))
    assert 2 * wf == pytest.approx(stmp.gd.integrate(stmp.ldos), abs=0.02)

    stmp.write_3D(+4., f3dname)
    print(me + 'Integrals(unocc): 2 * wf, bias=', end=' ')
    print(2 * wf, stmp.gd.integrate(stmp.ldos))
    assert 2 * wf == pytest.approx(stmp.gd.integrate(stmp.ldos), abs=0.02)

    energy_tolerance = 0.007
    assert e1 == pytest.approx(-2.54026, abs=energy_tolerance)
    assert e2 == pytest.approx(-1.51101, abs=energy_tolerance)
    assert e3 == pytest.approx(-2.83573, abs=energy_tolerance)
