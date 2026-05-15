from ase import Atoms
from gpaw import GPAW, PW
from gpaw.mpi import world, serial_comm


def test_pw_fulldiag(in_tmp_dir, scalapack):
    a = Atoms('H2',
              [(0, 0, 0), (0, 0, 0.74)],
              cell=(3, 3, 3),
              pbc=1)

    a.calc = GPAW(mode=PW(force_complex_dtype=True),
                  eigensolver='rmm-diis',
                  nbands=8,
                  parallel={'domain': 1},
                  basis='dzp',
                  txt='H2.txt')

    a.get_potential_energy()
    _ = a.calc.get_pseudo_wave_function(0)
    e1 = a.calc.get_eigenvalues()
    w1 = a.calc.get_pseudo_wave_function(0)

    a.calc.write('H2.gpw')

    if world.size == 1:
        scalapack = None
    else:
        scalapack = (2, world.size // 2, 32)

    a.calc.diagonalize_full_hamiltonian(nbands=120, scalapack=scalapack)
    w2 = a.calc.get_pseudo_wave_function(0)
    e2 = a.calc.get_eigenvalues()

    calc = GPAW('H2.gpw', txt=None, parallel={'domain': 1})
    calc.diagonalize_full_hamiltonian(nbands=120, scalapack=scalapack)
    w3 = calc.get_pseudo_wave_function(0)
    e3 = calc.get_eigenvalues()

    calc.write('H2wf.gpw', 'all')

    calc = GPAW('H2wf.gpw', txt=None, communicator=serial_comm)
    w4 = calc.get_pseudo_wave_function(0)
    e4 = calc.get_eigenvalues()

    for w in [w2, w3, w4]:
        err = abs(abs(w[1, 2, 3]) - abs(w1[1, 2, 3]))
        assert err < 5e-7, err

    for e in [e2, e3, e4]:
        err = abs(e[1] - e1[1])
        assert err < 1e-9, err
        err = abs(e[-1] - e2[-1])
        assert err < 1e-10, err
