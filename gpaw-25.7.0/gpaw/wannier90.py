import numpy as np
from gpaw.utilities.blas import gemmdot
from gpaw.ibz2bz import (get_overlap, get_overlap_coefficients,
                         get_phase_shifted_overlap_coefficients,
                         IBZ2BZMaps)
from gpaw.spinorbit import soc_eigenstates


class Wannier90:
    def __init__(self, calc, seed=None, bands=None, orbitals_ai=None,
                 spin=0, spinors=False):

        if seed is None:
            seed = calc.atoms.get_chemical_formula()
        self.seed = seed

        if bands is None:
            bands = range(calc.get_number_of_bands())
        self.bands = bands

        Na = len(calc.atoms)
        if orbitals_ai is None:
            orbitals_ai = []
            for ia in range(Na):
                ni = 0
                setup = calc.wfs.setups[ia]
                for l, n in zip(setup.l_j, setup.n_j):
                    if not n == -1:
                        ni += 2 * l + 1
                orbitals_ai.append(range(ni))

        self.calc = calc
        self.ibz2bz = IBZ2BZMaps.from_calculator(calc)
        self.bands = bands
        self.Nn = len(bands)
        self.Na = Na
        self.orbitals_ai = orbitals_ai
        self.Nw = np.sum([len(orbitals_ai[ai]) for ai in range(Na)])
        self.kpts_kc = calc.get_ibz_k_points()
        self.Nk = len(self.kpts_kc)
        self.spin = spin
        self.spinors = spinors

        if spinors:
            # spinorbit.WaveFunctions.transform currently do not suppport
            # transformation of wavefunctions, only projections.
            # XXX: should be updated in the future
            assert calc.wfs.kd.nbzkpts == calc.wfs.kd.nibzkpts
            self.soc = soc_eigenstates(calc)
        else:
            self.soc = None

    def write_input(self,
                    mp=None,
                    plot=False,
                    num_iter=100,
                    write_xyz=False,
                    write_rmn=False,
                    translate_home_cell=False,
                    dis_num_iter=200,
                    dis_froz_max=0.1,
                    dis_mix_ratio=0.5,
                    dis_win_min=None,
                    dis_win_max=None,
                    search_shells=None,
                    write_u_matrices=False):
        calc = self.calc
        seed = self.seed
        bands = self.bands
        orbitals_ai = self.orbitals_ai
        spinors = self.spinors

        if seed is None:
            seed = calc.atoms.get_chemical_formula()

        if bands is None:
            bands = range(calc.get_number_of_bands())

        Na = len(calc.atoms)
        if orbitals_ai is None:
            orbitals_ai = []
            for ia in range(Na):
                ni = 0
                setup = calc.wfs.setups[ia]
                for l, n in zip(setup.l_j, setup.n_j):
                    if not n == -1:
                        ni += 2 * l + 1
                orbitals_ai.append(range(ni))
        assert len(orbitals_ai) == Na

        Nw = np.sum([len(orbitals_ai[ai]) for ai in range(Na)])
        if spinors:
            Nw *= 2
            new_bands = []
            for n in bands:
                new_bands.append(2 * n)
                new_bands.append(2 * n + 1)
            bands = new_bands

        f = open(seed + '.win', 'w')

        pos_ac = calc.spos_ac
        # pos_av = calc.atoms.get_positions()
        # cell_cv = calc.atoms.get_cell()
        # pos_ac = np.dot(pos_av, np.linalg.inv(cell_cv))

        print('begin projections', file=f)
        for ia, orbitals_i in enumerate(orbitals_ai):
            setup = calc.wfs.setups[ia]
            l_i = []
            n_i = []
            for n, l in zip(setup.n_j, setup.l_j):
                if not n == -1:
                    l_i += (2 * l + 1) * [l]
                    n_i += (2 * l + 1) * [n]
            r_c = pos_ac[ia]
            for orb in orbitals_i:
                l = l_i[orb]
                n = n_i[orb]
                print(f'f={r_c[0]:1.2f}, {r_c[1]:1.2f}, {r_c[2]:1.2f} : s ',
                      end='', file=f)
                print(f'# n = {n}, l = {l}', file=f)

        print('end projections', file=f)
        print(file=f)

        if spinors:
            print('spinors = True', file=f)
        else:
            print('spinors = False', file=f)
        if write_u_matrices:
            print('write_u_matrices = True', file=f)
        print('write_hr = True', file=f)
        if write_xyz:
            print('write_xyz = True', file=f)
        if write_rmn:
            print('write_tb = True', file=f)
            print('write_rmn = True', file=f)
        if translate_home_cell:
            print('translate_home_cell = True', file=f)
        print(file=f)
        print('num_bands       = %d' % len(bands), file=f)

        if search_shells is not None:
            print(f"search_shells = {search_shells}", file=f)

        maxn = max(bands)
        if maxn + 1 != len(bands):
            diffn = maxn - len(bands)
            print('exclude_bands : ', end='', file=f)
            counter = 0
            for n in range(maxn):
                if n not in bands:
                    counter += 1
                    if counter != diffn + 1:
                        print('%d,' % (n + 1), sep='', end='', file=f)
                    else:
                        print('%d' % (n + 1), file=f)
        print(file=f)

        print('guiding_centres = True', file=f)
        print('num_wann        = %d' % Nw, file=f)
        print('num_iter        = %d' % num_iter, file=f)
        print(file=f)

        if len(bands) > Nw:
            ef = calc.get_fermi_level()
            print('fermi_energy  = %2.3f' % ef, file=f)
            if dis_froz_max is not None:
                print('dis_froz_max  = %2.3f' % (ef + dis_froz_max), file=f)
            if dis_win_min is not None:
                print('dis_win_min  = %2.3f' % (ef + dis_win_min), file=f)
            if dis_win_max is not None:
                print('dis_win_max  = %2.3f' % (ef + dis_win_max), file=f)
            print('dis_num_iter  = %d' % dis_num_iter, file=f)
            print('dis_mix_ratio = %1.1f' % dis_mix_ratio, file=f)
        print(file=f)

        print('begin unit_cell_cart', file=f)
        for cell_c in calc.atoms.cell:
            print(f'{cell_c[0]:14.10f} {cell_c[1]:14.10f} {cell_c[2]:14.10f}',
                  file=f)
        print('end unit_cell_cart', file=f)
        print(file=f)

        print('begin atoms_frac', file=f)
        for atom, pos_c in zip(calc.atoms, pos_ac):
            print(atom.symbol, end='', file=f)
            print(f'{pos_c[0]:14.10f} {pos_c[1]:14.10f} {pos_c[2]:14.10f}',
                  file=f)
        print('end atoms_frac', file=f)
        print(file=f)

        if plot:
            print('wannier_plot   = True', file=f)
            print('wvfn_formatted = True', file=f)
            print(file=f)

        if mp is not None:
            N_c = mp
        else:
            N_c = calc.wfs.kd.N_c
        print('mp_grid =', N_c[0], N_c[1], N_c[2], file=f)
        print(file=f)
        print('begin kpoints', file=f)

        for kpt in calc.get_bz_k_points():
            print(f'{kpt[0]:14.10f} {kpt[1]:14.10f} {kpt[2]:14.10f}', file=f)
        print('end kpoints', file=f)

        f.close()

    def write_projections(self):
        calc = self.calc
        seed = self.seed
        spin = self.spin
        orbitals_ai = self.orbitals_ai
        soc = self.soc

        if seed is None:
            seed = calc.atoms.get_chemical_formula()

        bands = get_bands(seed)
        Nn = len(bands)

        spinors = False

        win_file = open(seed + '.win')
        for line in win_file.readlines():
            l_e = line.split()
            if len(l_e) > 0:
                if l_e[0] == 'spinors':
                    spinors = l_e[2]
                    if spinors in ['T', 'true', '1', 'True']:
                        spinors = True
                    else:
                        spinors = False
                if l_e[0] == 'num_wann':
                    Nw = int(l_e[2])
                if l_e[0] == 'mp_grid':
                    Nk = int(l_e[2]) * int(l_e[3]) * int(l_e[4])
                    assert Nk == len(calc.get_bz_k_points())

        Na = len(calc.atoms)
        if orbitals_ai is None:
            orbitals_ai = []
            for ia in range(Na):
                ni = 0
                setup = calc.wfs.setups[ia]
                for l, n in zip(setup.l_j, setup.n_j):
                    if not n == -1:
                        ni += 2 * l + 1
                orbitals_ai.append(range(ni))
        assert len(orbitals_ai) == Na

        if spinors:
            new_orbitals_ai = []
            for orbitals_i in orbitals_ai:
                new_orbitals_i = []
                for i in orbitals_i:
                    new_orbitals_i.append(2 * i)
                    new_orbitals_i.append(2 * i + 1)
                new_orbitals_ai.append(new_orbitals_i)
            orbitals_ai = new_orbitals_ai

        Ni = 0
        for orbitals_i in orbitals_ai:
            Ni += len(orbitals_i)
        assert Nw == Ni

        f = open(seed + '.amn', 'w')

        print('Kohn-Sham input generated from GPAW calculation', file=f)
        print('%10d %6d %6d' % (Nn, Nk, Nw), file=f)

        P_kni = np.zeros((Nk, Nn, Nw), complex)
        for ik in range(Nk):
            if spinors:
                P_ani = soc[ik].P_amj
            else:
                P_ani = get_projections_in_bz(calc.wfs,
                                              ik,
                                              spin,
                                              self.ibz2bz,
                                              bcomm=None)
            for i in range(Nw):
                icount = 0
                for ai in range(Na):
                    ni = len(orbitals_ai[ai])
                    P_ni = P_ani[ai][bands]
                    P_ni = P_ni[:, orbitals_ai[ai]]
                    P_kni[ik, :, icount:ni + icount] = P_ni.conj()
                    icount += ni

        for ik in range(Nk):
            for i in range(Nw):
                for n in range(Nn):
                    P = P_kni[ik, n, i]
                    data = (n + 1, i + 1, ik + 1, P.real, P.imag)
                    print('%4d %4d %4d %18.12f %20.12f' % data, file=f)

        f.close()

    def write_eigenvalues(self):
        calc = self.calc
        seed = self.seed
        spin = self.spin
        soc = self.soc

        bands = get_bands(seed)

        f = open(seed + '.eig', 'w')

        for ik in range(len(calc.get_bz_k_points())):
            if soc is None:
                ibzk = calc.wfs.kd.bz2ibz_k[ik]  # IBZ k-point
                e_n = calc.get_eigenvalues(kpt=ibzk, spin=spin)
            else:
                e_n = soc[ik].eig_m
            for i, n in enumerate(bands):
                data = (i + 1, ik + 1, e_n[n])
                print('%5d %5d %14.6f' % data, file=f)

        f.close()

    def write_overlaps(self, less_memory=False):
        calc = self.calc
        seed = self.seed
        spin = self.spin
        soc = self.soc
        ibz2bz = self.ibz2bz

        if seed is None:
            seed = calc.atoms.get_chemical_formula()

        if soc is None:
            spinors = False
        else:
            spinors = True

        bands = get_bands(seed)
        Nn = len(bands)
        kpts_kc = calc.get_bz_k_points()
        Nk = len(kpts_kc)

        nnkp = open(seed + '.nnkp')
        lines = nnkp.readlines()
        for il, line in enumerate(lines):
            if len(line.split()) > 1:
                if line.split()[0] == 'begin' and line.split()[1] == 'nnkpts':
                    Nb = eval(lines[il + 1].split()[0])
                    i0 = il + 2
                    break

        f = open(seed + '.mmn', 'w')

        print('Kohn-Sham input generated from GPAW calculation', file=f)
        print('%10d %6d %6d' % (Nn, Nk, Nb), file=f)

        icell_cv = (2 * np.pi) * np.linalg.inv(calc.wfs.gd.cell_cv).T
        r_g = calc.wfs.gd.get_grid_point_coordinates()

        spos_ac = calc.spos_ac
        wfs = calc.wfs
        dO_aii = get_overlap_coefficients(wfs)

        if not less_memory:
            u_knG = []
            for ik in range(Nk):
                u_nG = self.wavefunctions(ik, bands)
                u_knG.append(u_nG)

        proj_k = []
        for ik in range(Nk):
            if spinors:
                proj_k.append(soc[ik].projections)
            else:
                proj_k.append(get_projections_in_bz(calc.wfs,
                                                    ik, spin,
                                                    ibz2bz,
                                                    bcomm=None))

        for ik1 in range(Nk):
            if less_memory:
                u1_nG = self.wavefunctions(ik1, bands)
            else:
                u1_nG = u_knG[ik1]
            for ib in range(Nb):
                # b denotes nearest neighbor k-points
                line = lines[i0 + ik1 * Nb + ib].split()
                ik2 = int(line[1]) - 1
                if less_memory:
                    u2_nG = self.wavefunctions(ik2, bands)
                else:
                    u2_nG = u_knG[ik2]

                G_c = np.array([int(line[i]) for i in range(2, 5)])
                bG_v = np.dot(G_c, icell_cv)
                u2_nG = u2_nG * np.exp(-1.0j * gemmdot(bG_v, r_g, beta=0.0))
                bG_c = kpts_kc[ik2] - kpts_kc[ik1] + G_c
                phase_shifted_dO_aii = get_phase_shifted_overlap_coefficients(
                    dO_aii, spos_ac, -bG_c)
                M_mm = get_overlap(bands,
                                   wfs.gd,
                                   u1_nG,
                                   u2_nG,
                                   proj_k[ik1],
                                   proj_k[ik2],
                                   phase_shifted_dO_aii)
                indices = (ik1 + 1, ik2 + 1, G_c[0], G_c[1], G_c[2])
                print('%3d %3d %4d %3d %3d' % indices, file=f)
                for m1 in range(len(M_mm)):
                    for m2 in range(len(M_mm)):
                        M = M_mm[m2, m1]
                        print(f'{M.real:20.12f} {M.imag:20.12f}', file=f)

        f.close()

    def write_wavefunctions(self):

        calc = self.calc
        soc = self.soc
        spin = self.spin
        seed = self.seed

        if soc is None:
            spinors = False
        else:
            spinors = True

        if seed is None:
            seed = calc.atoms.get_chemical_formula()

        bands = get_bands(seed)
        Nn = len(bands)
        Nk = len(calc.get_bz_k_points())

        for ik in range(Nk):
            if spinors:
                # For spinors, G denotes spin and grid: G = (s, gx, gy, gz)
                u_nG = soc[ik].wavefunctions(calc, periodic=True)
            else:
                # For non-spinors, G denotes grid: G = (gx, gy, gz)
                u_nG = self.wavefunctions(ik, bands)

            f = open('UNK%s.%d' % (str(ik + 1).zfill(5), spin + 1), 'w')
            grid_v = np.shape(u_nG)[1:]
            print(grid_v[0], grid_v[1], grid_v[2], ik + 1, Nn, file=f)
            for n in bands:
                for iz in range(grid_v[2]):
                    for iy in range(grid_v[1]):
                        for ix in range(grid_v[0]):
                            u = u_nG[n, ix, iy, iz]
                            print(u.real, u.imag, file=f)
            f.close()

    def wavefunctions(self, bz_index, bands):
        maxband = bands[-1] + 1
        if self.spinors:
            # For spinors, G denotes spin and grid: G = (s, gx, gy, gz)
            return self.soc[bz_index].wavefunctions(
                self.calc, periodic=True)[bands]
        # For non-spinors, G denotes grid: G = (gx, gy, gz)
        ibz_index = self.calc.wfs.kd.bz2ibz_k[bz_index]
        ut_nR = np.array([self.calc.wfs.get_wave_function_array(
            n, ibz_index, self.spin,
            periodic=True) for n in range(maxband)])
        ut_nR_sym = np.array([self.ibz2bz[bz_index].map_pseudo_wave_to_BZ(
            ut_nR[n]) for n in range(maxband)])

        return ut_nR_sym


def get_bands(seed):
    win_file = open(seed + '.win')
    exclude_bands = None
    for line in win_file.readlines():
        l_e = line.split()
        if len(l_e) > 0:
            if l_e[0] == 'num_bands':
                Nn = int(l_e[2])
            if l_e[0] == 'exclude_bands':
                exclude_bands = line.split()[2]
                exclude_bands = [int(n) - 1 for n in exclude_bands.split(',')]
    if exclude_bands is None:
        bands = range(Nn)
    else:
        bands = range(Nn + len(exclude_bands))
        bands = [n for n in bands if n not in exclude_bands]
    win_file.close()

    return bands


def get_projections_in_bz(wfs, K, s, ibz2bz, bcomm=None):
    """ Returns projections object in full BZ
    wfs: calc.wfs object
    K: BZ k-point index
    s: spin index
    ibz2bz: IBZ2BZMaps
    bcomm: band communicator
    """
    ik = wfs.kd.bz2ibz_k[K]  # IBZ k-point
    kpt = wfs.kpt_qs[ik][s]
    nbands = wfs.bd.nbands
    # Get projections in ibz
    proj = kpt.projections.new(nbands=nbands, bcomm=bcomm)
    proj.array[:] = kpt.projections.array[:nbands]

    # map projections to bz
    proj_sym = ibz2bz[K].map_projections(proj)
    return proj_sym


def read_umat(seed, kd, dis=False):
    """
    Reads wannier transformation matrix
    """
    if ".mat" not in seed:
        if dis:
            seed += "_u_dis.mat"
        else:
            seed += "_u.mat"
    f = open(seed, "r")
    f.readline()  # first line is a comment
    nk, nw1, nw2 = [int(i) for i in f.readline().split()]
    assert nk == kd.nbzkpts
    uwan = np.empty([nw1, nw2, nk], dtype=complex)
    iklist = []  # list to store found iks
    for ik1 in range(nk):
        f.readline()  # empty line
        K_c = [float(rdum) for rdum in f.readline().split()]
        ik = kd.where_is_q(K_c, kd.bzk_kc)
        assert np.allclose(np.array(K_c), kd.bzk_kc[ik])
        iklist.append(ik)
        for ib1 in range(nw1):
            for ib2 in range(nw2):
                rdum1, rdum2 = [float(rdum) for rdum in
                                f.readline().split()]
                uwan[ib1, ib2, ik] = complex(rdum1, rdum2)
    assert set(iklist) == set(range(nk))  # check that all k:s were found
    return uwan, nk, nw1, nw2


def read_uwan(seed, kd, dis=False):
    """
    Reads wannier transformation matrix
    Input parameters:
    -----------------
    seed: str
          seed in wannier calculation
    kd: kpt descriptor
    dis: logical
        should be set to true if nband > nwan
    """
    assert '.mat' not in seed
    # reads in wannier transformation matrix
    umat, nk, nw1, nw2 = read_umat(seed, kd, dis=False)

    if dis:
        # Reads in transformation to optimal subspace
        umat_dis, nk, nw1, nw2 = read_umat(seed, kd, dis=True)
        uwan = np.zeros_like(umat_dis)
        for ik in range(nk):
            uwan[:, :, ik] = umat[:, :, ik] @ umat_dis[:, :, ik]
    else:
        uwan = umat
    return uwan, nk, nw1, nw2
