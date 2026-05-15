import pytest
from ase import Atoms
from gpaw.mpi import world
from gpaw import GPAW, FermiDirac, Mixer, MixerSum, Davidson
from gpaw.atom.all_electron import AllElectron
from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters


@pytest.mark.slow
def test_xc_lb94(in_tmp_dir, add_cwd_to_setup_paths):
    ref1 = 'R. v. Leeuwen PhysRevA 49, 2421 (1994)'
    ref2 = 'Gritsenko IntJQuanChem 76, 407 (2000)'
    # HOMO energy in mHa for closed shell atoms
    e_HOMO_cs = {'He': 851, 'Be': 321, 'Ne': 788,
                 'Ar': 577, 'Kr': 529, 'Xe': 474,
                 'Mg': 281 + 8}

    txt = None

    print('--- Comparing LB94 with', ref1)
    print('and', ref2)

    print('**** all electron calculations')
    print('atom [refs] -e_homo diff   all in mHa')
    if world.rank == 0:
        for atom in e_HOMO_cs.keys():
            ae = AllElectron(atom, 'LB94', txt=txt)
            ae.run()
            e_homo = int(ae.e_j[-1] * 10000 + .5) / 10.
            diff = e_HOMO_cs[atom] + e_homo
            print('%2s %8g %6.1f %4.1g' %
                  (atom, e_HOMO_cs[atom], -e_homo, diff))
            assert abs(diff) < 6
    world.barrier()

    setups = {}

    print('**** 3D calculations')
    print('atom [refs] -e_homo diff   all in mHa')

    for atom in sorted(e_HOMO_cs):
        e_ref = e_HOMO_cs[atom]
        # generate setup for the atom
        if world.rank == 0 and atom not in setups:
            g = Generator(atom, 'LB94', nofiles=True, txt=txt)
            g.run(**parameters[atom])
            setups[atom] = 1
        world.barrier()

        SS = Atoms(atom, cell=(7, 7, 7), pbc=False)
        SS.center()
        c = GPAW(mode='fd', h=.3, xc='LB94',
                 eigensolver=Davidson(3),
                 mixer=Mixer(0.5, 7, 50.0), nbands=-2, txt=txt)
        c.calculate(SS)
        # find HOMO energy
        eps_n = c.get_eigenvalues(kpt=0, spin=0) / 27.211
        f_n = c.get_occupation_numbers(kpt=0, spin=0)
        for e, f in zip(eps_n, f_n):
            if f < 0.99:
                break
            e_homo = e
        e_homo = int(e_homo * 10000 + .5) / 10.
        diff = e_ref + e_homo
        print('%2s %8g %6.1f %4.1f' % (atom, e_ref, -e_homo, diff))
        assert abs(diff) < 7

    # HOMO energy in mHa and magn. mom. for open shell atoms
    e_HOMO_os = {'He': [851, 0],  # added for cross check
                 'H': [440, 1],
                 'N': [534 - 23, 3],
                 'Na': [189 + 17, 1],
                 'P': [385 - 16, 3]}

    for atom in sorted(e_HOMO_os):
        e_ref = e_HOMO_os[atom][0]
        magmom = e_HOMO_os[atom][1]
        # generate setup for the atom
        if world.rank == 0 and atom not in setups:
            g = Generator(atom, 'LB94', nofiles=True, txt=txt)
            g.run(**parameters[atom])
            setups[atom] = 1
        world.barrier()

        SS = Atoms(atom, magmoms=[magmom], cell=(7, 7, 7), pbc=False)
        SS.center()
        # fine grid needed for convergence!
        c = GPAW(mode='fd',
                 h=0.2,
                 xc='LB94',
                 nbands=-2,
                 spinpol=True,
                 hund=True,
                 eigensolver=Davidson(3),
                 mixer=MixerSum(0.5, 7, 50.0),
                 occupations=FermiDirac(0.0, fixmagmom=True),
                 txt=txt)
        c.calculate(SS)
        # find HOMO energy
        eps_n = c.get_eigenvalues(kpt=0, spin=0) / 27.211
        f_n = c.get_occupation_numbers(kpt=0, spin=0)
        for e, f in zip(eps_n, f_n):
            if f < 0.99:
                break
            e_homo = e
        e_homo = int(e_homo * 10000 + .5) / 10.
        diff = e_ref + e_homo
        print('%2s %8g %6.1f %4.1f' % (atom, e_ref, -e_homo, diff))
        assert abs(diff) < 15
