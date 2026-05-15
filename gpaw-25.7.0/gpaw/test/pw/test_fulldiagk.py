from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import serial_comm, world


def test_pw_fulldiagk(in_tmp_dir, scalapack):
    a = Atoms('H',
              cell=(1, 3, 3),
              pbc=1)

    a.calc = GPAW(mode='pw',
                  h=0.15,
                  kpts=(4, 1, 1),
                  basis='dzp',
                  nbands=4,
                  eigensolver='rmm-diis',
                  parallel={'domain': 1})

    a.get_potential_energy()
    w1 = a.calc.get_pseudo_wave_function(0, 1)
    e1 = a.calc.get_eigenvalues(1)

    a.calc.write('H.gpw')

    if world.size <= 2:
        scalapack = None
    else:
        mb = world.size // 4
        scalapack = (2, mb, 32)

    a.calc.diagonalize_full_hamiltonian(nbands=100, scalapack=scalapack)
    w2 = a.calc.get_pseudo_wave_function(0, 1)
    e2 = a.calc.get_eigenvalues(1)

    calc = GPAW('H.gpw',
                txt=None,
                parallel={'domain': 1})
    calc.diagonalize_full_hamiltonian(nbands=100, scalapack=scalapack)
    w3 = calc.get_pseudo_wave_function(0, 1)
    e3 = calc.get_eigenvalues(1)

    calc.write('Hwf.gpw', 'all')

    calc = GPAW('Hwf.gpw', txt=None, communicator=serial_comm)
    w4 = calc.get_pseudo_wave_function(0, 1)
    e4 = calc.get_eigenvalues(1)

    for w in [w2, w3, w4]:
        err = abs(abs(w[1, 2, 3]) - abs(w1[1, 2, 3]))
        print(err)
        assert err < 5e-7, err

    for e in [e2, e3, e4]:
        err = abs(e[0] - e1[0])
        assert err < 2e-9, err
        err = abs(e[-1] - e2[-1])
        assert err < 1e-10, err

    a.calc = GPAW(mode='pw',
                  parallel={'band': min(2, world.size)},
                  h=0.15,
                  kpts=(4, 1, 1),
                  convergence={'bands': 'CBM+10'},
                  nbands=4)
    a.get_potential_energy()

    e5 = a.calc.get_eigenvalues(0)
    de = e5 - calc.get_eigenvalues(0)[:4]
    cbm = calc.get_fermi_level()
    assert abs(de[e5 < cbm + 10]).max() < 0.001


if __name__ == '__main__':
    test_pw_fulldiagk(1, 2)
