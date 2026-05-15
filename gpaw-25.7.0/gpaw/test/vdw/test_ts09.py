import ase.io
import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.vdwcorrection import vdWTkatchenko09prl
from ase.parallel import barrier, parprint

from gpaw import GPAW
from gpaw.analyse.hirshfeld import HirshfeldPartitioning
from gpaw.analyse.vdwradii import vdWradii
from gpaw.utilities.adjust_cell import adjust_cell


@pytest.mark.old_gpaw_only
def test_vdw_ts09(in_tmp_dir):
    h = 0.4
    s = molecule('LiH')
    adjust_cell(s, 3., h=h)

    def print_charge_and_check(hp, q=0, label='unpolarized'):
        q_a = np.array(hp.get_charges())
        parprint('Charges ({0})='.format(label), q_a, ', sum=', q_a.sum())
        assert q_a.sum() == pytest.approx(q, abs=0.03)
        return q_a

    # spin unpolarized

    if 1:
        out_traj = 'LiH.traj'
        out_txt = 'LiH.txt'

        cc = GPAW(mode='fd', h=h, xc='PBE', txt=out_txt)

        # this is needed to initialize txt output
        cc.initialize(s)

        hp = HirshfeldPartitioning(cc)
        c = vdWTkatchenko09prl(hp,
                               vdWradii(s.get_chemical_symbols(), 'PBE'))
        s.calc = c
        E = s.get_potential_energy()
        F_ac = s.get_forces()
        s.write(out_traj)
        q_a = print_charge_and_check(hp)

        barrier()

        # test I/O, accuracy due to text output
        accuracy = 1.e-5
        for fname in [out_traj, out_txt]:
            s_out = ase.io.read(fname)
            assert s_out.get_potential_energy() == pytest.approx(E,
                                                                 abs=accuracy)
            for fi, fo in zip(F_ac, s_out.get_forces()):
                assert fi == pytest.approx(fo, abs=accuracy)

    # spin polarized

    if 0:
        ccs = GPAW(mode='fd', h=h, xc='PBE', spinpol=True,
                   txt=None)
        hps = HirshfeldPartitioning(ccs)
        cs = vdWTkatchenko09prl(hps, vdWradii(s.get_chemical_symbols(), 'PBE'))
        s.calc = cs
        Es = s.get_potential_energy()
        Fs_ac = s.get_forces()

        qs_a = print_charge_and_check(hps, label='spin')

        assert q_a == pytest.approx(qs_a, abs=1.e-6)
        assert E == pytest.approx(Es, abs=1.e-4)
        assert F_ac == pytest.approx(Fs_ac, abs=1.e-4)

    # charged

    if 0:
        cc = cc.new(charge=1)
        hpp = HirshfeldPartitioning(cc)
        cp = vdWTkatchenko09prl(hpp,
                                vdWradii(s.get_chemical_symbols(), 'PBE'))
        s.calc = cp
        E = s.get_potential_energy()

        print_charge_and_check(hpp, 1, label='+1')
