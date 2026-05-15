import pytest
import numpy as np
from ase.build import bulk
from gpaw import GPAW


refs = {'GLLBSC':
        [[-6.96783989, 5.3992398, 5.40304598, 5.40304598, 8.50339453,
          8.50339453, 8.50489331, 8.7972915],
         [-5.53899418, -0.14491454, 4.14051678, 4.14051678, 7.89976594,
          10.62148051, 10.62148051, 15.47779579]],
        'GLLBSC:width=0.1':
        [[-6.90967462, 5.45874896, 5.46258594, 5.46258594, 8.56019914,
          8.56019914, 8.56166802, 8.8487109],
         [-5.48079026, -0.08685927, 4.2001473, 4.2001473, 7.95314291,
          10.67741742, 10.67741742, 15.53082277]],
        'GLLBSCM':
        [[-6.55170494, 5.8177223, 5.82167851, 5.82167851, 8.9114778,
          8.9114778, 8.9128251, 9.16145391],
         [-5.12306231, 0.27223217, 4.56254371, 4.56254371, 8.28319508,
          11.02205235, 11.02205235, 15.85416557]],
        'GLLBSC:metallic=1:width=0.1':
        [[-6.51118445, 5.85791198, 5.86187878, 5.86187878, 8.95128038,
          8.95128038, 8.95261673, 9.19621624],
         [-5.08260244, 0.3129949, 4.60331705, 4.60331705, 8.32039946,
          11.06096213, 11.06096213, 15.89026172]],
        'GLLBSC:stencil=1':
        [[-6.96553391, 5.39977866, 5.40347256, 5.40347256, 8.5044520,
          8.50445200, 8.50610892, 8.79785566],
         [-5.53673488, -0.14290236, 4.14161043, 4.14161043, 7.90076722,
          10.62164739, 10.62164739, 15.47627092]]
        }


@pytest.mark.gllb
@pytest.mark.libxc
@pytest.mark.parametrize('xc', ['GLLBSC', 'GLLBSC:width=0.1',
                                'GLLBSCM', 'GLLBSC:metallic=1:width=0.1',
                                'GLLBSC:stencil=1'])
def test_restart_eigenvalues(xc, in_tmp_dir):
    test_kpts = [[0, 0, 0], [1. / 3, 1. / 3, 1. / 3]]

    atoms = bulk('Si')
    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                h=0.3,
                nbands=8,
                xc=xc,
                kpts={'size': (3, 3, 3), 'gamma': True},
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()

    eref_s = calc.hamiltonian.xc.response.eref_s
    eref_source_s = calc.hamiltonian.xc.response.eref_source_s

    kpt_i = [0, 3]
    calc_kpts = calc.get_ibz_k_points()[kpt_i]
    assert np.allclose(calc_kpts, test_kpts, rtol=0, atol=1e-8), \
        "Wrong kpt indices"
    eig_in = [calc.get_eigenvalues(kpt=kpt) for kpt in kpt_i]
    eig_in = np.array(eig_in)
    calc.write('gs.gpw')

    # Check calculation against reference
    ref_eig_in = refs[xc]
    assert np.allclose(eig_in, ref_eig_in, rtol=0, atol=1e-6), \
        "{} error = {}".format(xc, np.max(np.abs(eig_in - ref_eig_in)))

    # Restart
    calc = GPAW('gs.gpw')
    calc = calc.fixed_density(kpts=test_kpts, txt='gs2.out')
    # Check that calculation was triggered
    scf = calc.scf
    assert scf is not None and scf.niter is not None and scf.niter > 0, \
        "Test error: SCF was not run in restart"
    eig2_in = [calc.get_eigenvalues(kpt=kpt) for kpt in range(len(kpt_i))]
    eig2_in = np.array(eig2_in)

    # Check restarted eigenvalues
    assert np.allclose(eig_in, eig2_in, rtol=0, atol=1e-8), \
        "{} restart error = {}".format(xc, np.max(np.abs(eig_in - eig2_in)))

    # Check that reference energy source information is restored
    assert eref_s == calc.hamiltonian.xc.response.eref_s
    assert eref_source_s == calc.hamiltonian.xc.response.eref_source_s
