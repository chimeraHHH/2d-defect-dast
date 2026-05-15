import numpy as np

from ase import Atoms

from gpaw import LCAO
from gpaw.calculator import GPAW as old_GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.new.ase_interface import GPAW as new_GPAW
from gpaw.new.rttddft import RTTDDFT
from gpaw.tddft.units import as_to_au, autime_to_asetime


def main():
    atoms = Atoms('H2', positions=[(0, 0, 0), (1, 0, 0)])
    atoms.center(vacuum=5)
    atoms.pbc = False

    kick_v = [1e-5, 0, 0]

    run_old_gs = True
    run_new_gs = True
    run_old_td = True
    run_new_td = True

    def assert_equal(a, b):
        from gpaw.core.matrix import Matrix
        from gpaw.core.atom_arrays import AtomArrays

        def extract(o):
            if isinstance(o, Matrix):
                return o.data
            elif isinstance(o, AtomArrays):
                return o.data
            else:
                return o

        a = extract(a)
        b = extract(b)

        assert np.allclose(a, b), f'{str(a)} != {str(b)}'

    if run_old_gs:
        old_calc = old_GPAW(mode=LCAO(), basis='sz(dzp)', xc='LDA',
                            symmetry={'point_group': False},
                            txt='old.out', convergence={'density': 1e-12})
        atoms.calc = old_calc
        atoms.get_potential_energy()
        old_calc.write('old_gs.gpw', mode='all')

    if run_new_gs:
        new_calc = new_GPAW(mode='lcao', basis='sz(dzp)', xc='LDA',
                            txt='new.out', force_complex_dtype=True,
                            convergence={'density': 1e-12})
        atoms.calc = new_calc
        atoms.get_potential_energy()
        new_calc.write('new_gs.gpw', mode='all')

        new_restart_calc = new_GPAW('new_gs.gpw')

        assert_equal(
            new_calc.dft.ibzwfs.wfs_qs[0][0].P_ain,
            new_restart_calc.dft.ibzwfs.wfs_qs[0][0].P_ain)

        assert_equal(
            new_calc.dft.ibzwfs.wfs_qs[0][0].C_nM,
            new_restart_calc.dft.ibzwfs.wfs_qs[0][0].C_nM)

    if run_old_td:
        old_tddft = LCAOTDDFT('old_gs.gpw', propagator='ecn', txt='/dev/null')
        DipoleMomentWriter(old_tddft, 'old_dm.out')
        old_tddft.absorption_kick(kick_v)
        old_tddft.propagate(10, 10)
        # old_C_nM = old_tddft.wfs.kpt_u[0].C_nM
        # old_f_n = old_tddft.get_occupation_numbers()
        # old_rho_MM = old_C_nM.T.conj() @ (old_f_n[:, None] * old_C_nM)
        # print('rho_MM', old_rho_MM)

    if run_new_td:
        #  new_tddft = RTTDDFT.from_dft_calculation(new_calc)
        new_tddft = RTTDDFT.from_dft_file('new_gs.gpw')

        new_tddft.absorption_kick(kick_v)
        dt = 10 * as_to_au * autime_to_asetime
        with open('new_dm.out', 'w') as fp:
            for result in new_tddft.ipropagate(dt, 10):
                dm = result.dipolemoment
                fp.write('%20.8lf %20.8le %22.12le %22.12le %22.12le\n' %
                         (result.time, 0, dm[0], dm[1], dm[2]))
                print(result)
        # wfs = new_tddft.state.ibzwfs.wfs_qs[0][0]
        # new_rho_MM = wfs.calculate_density_matrix()
        # print('rho_MM', new_rho_MM)


if __name__ == '__main__':
    main()
