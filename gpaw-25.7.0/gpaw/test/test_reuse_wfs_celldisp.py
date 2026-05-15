import numpy as np
from ase.build import molecule

from gpaw import GPAW, Mixer
from gpaw.mpi import world

# Place one atom next to cell boundary, then check that reuse_wfs
# works correctly when atom is subsequently displaced across the
# boundary, i.e., that the kpoint phases of the PAW correction
# are handled correctly when unprojecting/reprojecting the wavefunctions.


def test_reuse_wfs_celldisp(in_tmp_dir):
    def check(reuse):
        atoms = molecule('H2')
        atoms.pbc = 1
        atoms.center(vacuum=1.5)
        atoms.positions -= atoms.positions[1]
        dz = 1e-2
        atoms.positions[:, 2] += dz

        calc = GPAW(mode='pw',
                    txt=f'gpaw-{reuse}.txt',
                    nbands=1,
                    eigensolver='davidson',
                    experimental=dict(
                        reuse_wfs_method='paw' if reuse else None),
                    kpts=[[-0.3, 0.4, 0.2]],
                    symmetry='off',
                    mixer=Mixer(0.7, 5, 50.0))
        atoms.calc = calc

        for ctx in calc.icalculate(atoms):
            if ctx.niter == 2:
                # logerr1 = np.log10(calc.wfs.eigensolver.error)
                logerr1 = np.log10(ctx.wfs.eigensolver.error)

        atoms.positions[:, 2] -= 2 * dz

        if not reuse and not calc.old:
            calc.dft.ibzwfs.move_wave_functions = lambda *args: None

        for ctx in calc.icalculate(atoms, system_changes=['positions']):
            if ctx.niter == 2:
                logerr2 = np.log10(ctx.wfs.eigensolver.error)
                break

        if world.rank == 0:
            print(f'reuse={bool(reuse)}')
            print('logerr1', logerr1)
            print('logerr2', logerr2)
            gain = logerr2 - logerr1
            print('gain', gain)
        return logerr2

    noreuse_logerr = check(0)
    reuse_logerr = check(1)
    # Ref values: logerr=-4.8 without reuse_wfs and -6.1 with reuse_wfs
    assert reuse_logerr < -6.0, reuse_logerr
    assert reuse_logerr < noreuse_logerr - 1.2, (reuse_logerr, noreuse_logerr)
