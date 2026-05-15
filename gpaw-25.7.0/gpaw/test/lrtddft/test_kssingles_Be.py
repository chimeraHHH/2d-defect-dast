import pytest
import numpy as np

from ase import Atoms
from ase.parallel import parprint
from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.lrtddft.kssingle import KSSingles


@pytest.mark.lrtddft
def test_lrtddft_kssingles_Be(in_tmp_dir):
    Be = Atoms('Be')
    Be.center(vacuum=4)
    if 1:
        # introduce a sligth non-orthgonality
        cell = Be.get_cell()
        cell[1] += 0.001 * cell[0]
        Be.set_cell(cell)

    txt = None
    eigensolver = None

    # modes = ['lcao', 'fd']
    modes = ['fd']

    for mode in modes:
        energy = {}
        osz = {}
        for pbc in [False, True]:
            Be.set_pbc(pbc)
            if pbc:
                name = 'periodic'
                calc = GPAW(h=0.25,
                            nbands=4,
                            kpts=(1, 2, 2),
                            mode=mode,
                            poissonsolver={'name': 'fd'},
                            symmetry='off',
                            eigensolver=eigensolver,
                            txt=txt)
            else:
                name = 'zero_bc'
                calc = GPAW(h=0.25, nbands=4, mode=mode,
                            poissonsolver={'name': 'fd'},
                            eigensolver=eigensolver, txt=txt)
            Be.calc = calc
            Be.get_potential_energy()

            kss = KSSingles(restrict={'eps': 0.9})
            kss.calculate(Be)
            # all s->p transitions at the same energy [Ha] and
            # oscillator_strength
            for ks in kss:
                assert ks.get_energy() == pytest.approx(kss[0].get_energy(),
                                                        abs=5.e-3)
                assert ks.get_oscillator_strength()[0] == pytest.approx(
                    kss[0].get_oscillator_strength()[0], abs=5.e-3)
                assert ks.get_oscillator_strength()[0] == pytest.approx(
                    ks.get_oscillator_strength()[1:].sum() / 3, abs=1.e-15)
                for c in range(3):
                    assert ks.get_oscillator_strength()[1 + c] == (
                        pytest.approx(ks.get_dipole_tensor()[c, c],
                                      abs=1.e-15))
            energy[name] = np.array(
                [ks.get_energy() * Hartree for ks in kss]).mean()
            osz[name] = np.array(
                [ks.get_oscillator_strength()[0] for ks in kss]).sum()

            parprint(name + ':')
            parprint(kss)

            # I/O
            fname = 'kss_' + name + '.dat'
            kss.write(fname)
            mpi.world.barrier()
            kss = KSSingles.read(fname)
            kss1 = KSSingles.read(fname, restrict={'jend': 1})
            assert len(kss1) == calc.wfs.kd.nibzkpts * calc.wfs.nspins

        # periodic and non-periodic should be roughly equal
        assert energy['zero_bc'] == pytest.approx(energy['periodic'],
                                                  abs=5.e-2)
        assert osz['zero_bc'] == pytest.approx(osz['periodic'], abs=2.e-2)
