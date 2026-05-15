import pytest
from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack
# from ase.units import Bohr

from gpaw import GPAW, FermiDirac, PW
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.chi0 import Chi0Calculator
from gpaw.mpi import serial_comm
from itertools import product


@pytest.mark.response
@pytest.mark.slow
def test_response_chi0(in_tmp_dir):
    # inputs to loop over [k, gamma, center, sym]
    settings = product([2, 3], *[[False, True]] * 3)

    for k, gamma, center, sym in settings:
        if k == 3 and gamma:
            continue
        a = bulk('Si', 'diamond')
        q_c = [0, 0, 1.0 / k]
        kpts = monkhorst_pack((k, k, k))
        if gamma:
            kpts += 0.5 / k
        if center:
            a.center()
        name = 'si.k%d.g%d.c%d.s%d' % (k, gamma, center, bool(sym))

        calc = a.calc = GPAW(
            kpts=kpts,
            symmetry={'point_group': sym},
            mode=PW(150),
            occupations=FermiDirac(width=0.001),
            convergence={'bands': 8},
            txt=name + '.txt')
        a.get_potential_energy()
        calc.write(name, 'all')

        calc = GPAW(name, txt=None, communicator=serial_comm)

        chi0_calc = Chi0Calculator(
            gs=calc, context=name + '.log',
            wd=FrequencyDescriptor.from_array_or_dict([0, 1.0, 2.0]),
            hilbert=False, ecut=100)
        chi0 = chi0_calc.calculate(q_c)
        assert chi0.body.blockdist.blockcomm.size == 1
        chi0_wGG = chi0.chi0_WgG  # no block distribution

        # sym and center: False
        if not sym and not center:
            chi00_w = chi0_wGG[:, 0, 0]
        elif -1 not in calc.wfs.kd.bz2bz_ks:
            assert abs(chi0_wGG[:, 0, 0] - chi00_w).max() < 35e-5

        if not sym:
            chi00_wGG = chi0_wGG
        elif -1 not in calc.wfs.kd.bz2bz_ks:
            assert chi0_wGG == pytest.approx(chi00_wGG, abs=3e-5)

        chi0 = chi0_calc.calculate([0, 0, 0])
        assert chi0.body.blockdist.blockcomm.size == 1
        chi0_wGG = chi0.chi0_WgG  # no block distribution

        if not sym and not center:
            chi000_w = chi0_wGG[:, 0, 0]
        elif -1 not in calc.wfs.kd.bz2bz_ks:
            assert abs(chi0_wGG[:, 0, 0] - chi000_w).max() < 0.0015

        if not sym:
            chi000_wGG = chi0_wGG
        elif -1 not in calc.wfs.kd.bz2bz_ks:
            assert abs(chi0_wGG - chi000_wGG).max() < 0.0015
