import functools
from math import sqrt
from pathlib import Path
import numpy as np

from ase import Atom, Atoms
from ase.units import Bohr
from ase.build import bulk, molecule
from ase.lattice.compounds import L1_2
from ase.lattice.hexagonal import Graphene
from gpaw import Davidson, FermiDirac, GPAW, Mixer, PW, FD, LCAO
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.directmin.tools import excite
from gpaw.directmin.derivatives import Davidson as SICDavidson
from gpaw.mom import prepare_mom_calculation
from gpaw.mpi import world, serial_comm
from gpaw.new.ase_interface import GPAW as GPAWNew
from gpaw.poisson import FDPoissonSolver, PoissonSolver
from gpaw.test.cachedfilehandler import CachedFilesHandler

response_band_cutoff = {}
_all_gpw_methodnames = set()


def with_band_cutoff(*, gpw, band_cutoff):
    # Store the band cutoffs in a dictionary to aid response tests
    response_band_cutoff[gpw] = band_cutoff

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, band_cutoff=band_cutoff, **kwargs)
        return wrapper

    return decorator


def partial(fun, **kwargs):
    import copy
    kwargs = copy.deepcopy(kwargs)

    def f(self):
        return fun(self, **kwargs)
    return f


def _si_gw(self, *, a, symm, name):
    atoms = self.generate_si_systems()[a]
    return self._si_gw(atoms=atoms,
                       symm=symm,
                       name=f'{name}.txt')


def si_gpwfiles():
    gpw_file_dict = {}
    for a in [0, 1]:
        for symm, name1 in [({}, 'all'), ('off', 'no'),
                            ({'point_group': False}, 'tr'),
                            ({'time_reversal': False}, 'pg')]:
            name = f'si_gw_a{a}_{name1}'
            """
            In !2153, a bug related to late binding of local functions was
            fixed, and the partial wrapper utilized here is a temporary fix.
            Previously, the test would only test the last symmetry in the loop,
            but four times.
            """
            fun = partial(_si_gw, a=a, symm=symm, name=name)
            fun.__name__ = name
            gpw_file_dict[name] = gpwfile(fun)

    return gpw_file_dict


def gpwfile(meth):
    """Decorator to identify the methods that produce gpw files."""
    _all_gpw_methodnames.add(meth.__name__)
    return meth


class GPWFiles(CachedFilesHandler):
    """Create gpw-files."""

    def __init__(self, folder: Path):
        super().__init__(folder, '.gpw')

    def _calculate_and_write(self, name, work_path):
        calc = getattr(self, name)()
        calc.write(work_path, mode='all')

    @gpwfile
    def bcc_li_pw(self):
        return self.bcc_li({'name': 'pw', 'ecut': 200})

    @gpwfile
    def bcc_li_fd(self):
        return self.bcc_li({'name': 'fd'})

    @gpwfile
    def bcc_li_lcao(self):
        return self.bcc_li({'name': 'lcao'})

    def bcc_li(self, mode):
        li = bulk('Li', 'bcc', 3.49)
        li.calc = GPAW(mode=mode,
                       kpts=(3, 3, 3),
                       txt=self.folder / f'bcc_li_{mode["name"]}.txt')
        li.get_potential_energy()
        return li.calc

    @gpwfile
    def be_atom_fd(self):
        atoms = Atoms('Be', [(0, 0, 0)], pbc=False)
        atoms.center(vacuum=6)
        calc = GPAW(mode='fd', h=0.35, symmetry={'point_group': False},
                    txt=self.folder / 'be_atom_fd.txt')
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def fcc_Ni_col(self):
        return self._fcc_Ni('col')

    @gpwfile
    def fcc_Ni_ncol(self):
        return self._fcc_Ni('ncol')

    @gpwfile
    def fcc_Ni_ncolsoc(self):
        return self._fcc_Ni('ncolsoc')

    def _fcc_Ni(self, calc_type):
        Ni = bulk('Ni', 'fcc', 3.48)
        Ni.center()

        mm = 0.5
        easy_axis = 1 / np.sqrt(3) * np.ones(3)
        Ni.set_initial_magnetic_moments([mm])

        symmetry = {'point_group': False, 'time_reversal': True} if \
            calc_type == 'col' else 'off'
        magmoms = None if calc_type == 'col' else [mm * easy_axis]
        soc = True if calc_type == 'ncolsoc' else False

        Ni.calc = GPAWNew(mode={'name': 'pw', 'ecut': 380},
                          xc='LDA',
                          nbands='200%',
                          kpts={'size': (4, 4, 4), 'gamma': True},
                          parallel={'domain': 1, 'band': 1},
                          mixer={'beta': 0.5},
                          symmetry=symmetry,
                          occupations={'name': 'fermi-dirac', 'width': 0.05},
                          convergence={'density': 1e-8,
                                       'bands': 'CBM+10',
                                       'eigenstates': 1e-12},
                          magmoms=magmoms,
                          soc=soc,
                          txt=self.folder / f'fcc_Ni_{calc_type}.txt')
        Ni.get_potential_energy()
        return Ni.calc

    @gpwfile
    def h2_pw(self):
        return self.h2({'name': 'pw', 'ecut': 200})

    @gpwfile
    def h2_fd(self):
        return self.h2({'name': 'fd'})

    @gpwfile
    def h2_lcao(self):
        return self.h2({'name': 'lcao'})

    def h2(self, mode):
        h2 = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])
        h2.center(vacuum=2.5)
        h2.calc = GPAW(mode=mode,
                       txt=self.folder / f'h2_{mode["name"]}.txt')
        h2.get_potential_energy()
        return h2.calc

    @gpwfile
    def h2_pw_0(self):
        h2 = Atoms('H2',
                   positions=[[-0.37, 0, 0], [0.37, 0, 0]],
                   cell=[5.74, 5, 5],
                   pbc=True)
        h2.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                       txt=self.folder / 'h2_pw_0.txt')
        h2.get_potential_energy()
        return h2.calc

    @gpwfile
    def h2_bcc_afm(self):
        a = 2.75
        atoms = bulk(name='H', crystalstructure='bcc', a=a, cubic=True)
        atoms.set_initial_magnetic_moments([1., -1.])

        atoms.calc = GPAW(xc='LDA',
                          txt=self.folder / 'h2_bcc_afm.txt',
                          mode=PW(250),
                          nbands=4,
                          convergence={'bands': 4},
                          kpts={'density': 2.0, 'gamma': True})
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def c2h4_do_fd(self):
        atm = Atoms('CCHHHH',
                    positions=[
                        [-0.66874198, -0.00001714, -0.00001504],
                        [0.66874210, 0.00001699, 0.00001504],
                        [-1.24409879, 0.00000108, -0.93244784],
                        [-1.24406253, 0.00000112, 0.93242153],
                        [1.24406282, -0.93242148, 0.00000108],
                        [1.24409838, 0.93244792, 0.00000112]
                    ]
                    )
        atm.center(vacuum=4.0)
        atm.set_pbc(False)
        atm.calc = GPAW(mode=FD(), h=0.3,
                        xc='PBE',
                        occupations={'name': 'fixed-uniform'},
                        eigensolver={'name': 'etdm-fdpw',
                                     'converge_unocc': True},
                        mixer={'backend': 'no-mixing'},
                        spinpol=True,
                        symmetry='off',
                        nbands=-5,
                        convergence={'eigenstates': 4.0e-6},
                        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def c2h4_do_pw(self):
        atm = Atoms(
            "CCHHHH",
            positions=[
                [-0.66874198, -0.00001714, -0.00001504],
                [0.66874210, 0.00001699, 0.00001504],
                [-1.24409879, 0.00000108, -0.93244784],
                [-1.24406253, 0.00000112, 0.93242153],
                [1.24406282, -0.93242148, 0.00000108],
                [1.24409838, 0.93244792, 0.00000112],
            ],
        )
        atm.center(vacuum=4.0)
        atm.set_pbc(False)
        atm.calc = GPAW(
            mode=PW(300, force_complex_dtype=True),
            xc="PBE",
            occupations={"name": "fixed-uniform"},
            eigensolver={"name": "etdm-fdpw", "converge_unocc": True},
            mixer={"backend": "no-mixing"},
            spinpol=True,
            symmetry="off",
            nbands=-5,
            convergence={"eigenstates": 4.0e-6},
        )
        atm.get_potential_energy()
        return atm.calc

    def h3_maker(self, vacuum=2.0, pbc=True, **kwargs):
        atoms = Atoms('H3', positions=[(0, 0, 0),
                                       (0.59, 0, 0),
                                       (1.1, 0, 0)])
        atoms.center(vacuum=vacuum)
        atoms.set_pbc(pbc)
        return atoms

    @gpwfile
    def h3_do_num_pw_complex(self):
        atoms = self.h3_maker()
        calc = GPAW(mode=PW(300, force_complex_dtype=True),
                    basis='sz(dzp)',
                    h=0.3,
                    spinpol=False,
                    convergence={'energy': np.inf,
                                 'eigenstates': np.inf,
                                 'density': np.inf,
                                 'minimum iterations': 1},
                    eigensolver=FDPWETDM(converge_unocc=True),
                    occupations={'name': 'fixed-uniform'},
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    symmetry='off')
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h3_do_num_pw(self):
        atoms = self.h3_maker()
        calc = GPAW(mode=PW(300, force_complex_dtype=False),
                    basis='sz(dzp)',
                    h=0.3,
                    spinpol=False,
                    convergence={'energy': np.inf,
                                 'eigenstates': np.inf,
                                 'density': np.inf,
                                 'minimum iterations': 1},
                    eigensolver=FDPWETDM(converge_unocc=True),
                    occupations={'name': 'fixed-uniform'},
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    symmetry='off')
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h3_do_sd_lcao(self):
        atoms = self.h3_maker(pbc=False)
        atoms.set_initial_magnetic_moments([1, 0, 0])
        calc = GPAW(mode='lcao',
                    h=0.3,
                    spinpol=True,
                    convergence={'energy': 0.1,
                                 'eigenstates': 1e-4,
                                 'density': 1e-4},
                    eigensolver={'name': 'etdm-lcao'},
                    occupations={'name': 'fixed-uniform'},
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    symmetry='off')
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h3_do_num_lcao(self):
        atm = self.h3_maker()
        atm.calc = GPAW(mode=LCAO(force_complex_dtype=True),
                        basis='sz(dzp)',
                        h=0.3,
                        spinpol=False,
                        convergence={'eigenstates': 10.0,
                                     'density': 10.0,
                                     'energy': 10.0},
                        occupations={'name': 'fixed-uniform'},
                        eigensolver={'name': 'etdm-lcao',
                                     'matrix_exp': 'egdecomp'},
                        mixer={'backend': 'no-mixing'},
                        nbands='nao',
                        symmetry='off',
                        txt=None)
        atm.get_potential_energy()
        return atm.calc

    def h2o_maker(self, vacuum, t=np.pi / 180 * 104.51, eps=0, **kwargs):
        d = 0.9575
        H2O = Atoms('OH2',
                    positions=[(0, 0, 0),
                               (d + eps, 0, 0),
                               (d * np.cos(t), d * np.sin(t), 0)],
                    **kwargs)
        H2O.center(vacuum=vacuum)
        return H2O

    @gpwfile
    def h2o_do_gmf_lcao(self):
        atm = self.h2o_maker(vacuum=4.0)
        atm.calc = GPAW(
            mode=LCAO(),
            basis="dzp",
            h=0.22,
            occupations={"name": "fixed-uniform"},
            eigensolver="etdm-lcao",
            mixer={"backend": "no-mixing"},
            nbands="nao",
            symmetry="off",
            spinpol=True,
            convergence={"density": 1.0e-4, "eigenstates": 4.0e-8},
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_do_lcao(self):
        atm = self.h2o_maker(vacuum=5.0)
        atm.calc = GPAW(
            mode=LCAO(),
            basis="dzp",
            occupations={"name": "fixed-uniform"},
            eigensolver="etdm",
            mixer={"backend": "no-mixing"},
            nbands="nao",
            symmetry="off",
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_cdo_lcao(self):
        atm = self.h2o_maker(vacuum=4.0)
        atm.calc = GPAW(
            mode=LCAO(),
            basis="dzp",
            h=0.22,
            occupations={"name": "fixed-uniform"},
            eigensolver={"name": "etdm-lcao"},
            mixer={"backend": "no-mixing"},
            nbands="nao",
            symmetry="off",
            spinpol=True,
            convergence={"density": 1.0e-4, "eigenstates": 4.0e-8},
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_cdo_lcao_sic(self):
        atm = self.h2o_maker(vacuum=4.0)
        atm.calc = GPAW(
            mode=LCAO(force_complex_dtype=True),
            h=0.22,
            occupations={"name": "fixed-uniform"},
            eigensolver={
                "name": "etdm-lcao",
                "localizationtype": "PM_PZ",
                "localizationseed": 42,
                "subspace_convergence": 1e-3,
                "functional": {"name": "PZ-SIC", "scaling_factor": (0.5, 0.5)},
            },
            convergence={"eigenstates": 1e-4},
            mixer={"backend": "no-mixing"},
            nbands="nao",
            symmetry="off",
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_fdsic(self):
        atm = self.h2o_maker(vacuum=4.0,
                             t=np.pi / 180 * (104.51 + 2.0),
                             eps=0.02)
        atm.calc = GPAW(
            mode=FD(force_complex_dtype=True),
            h=0.25,
            occupations={"name": "fixed-uniform"},
            eigensolver=FDPWETDM(
                functional={"name": "PZ-SIC", "scaling_factor": (0.5, 0.5)},
                localizationseed=42,
                localizationtype="FB_ER",
                grad_tol_pz_localization=1.0e-3,
                maxiter_pz_localization=200,
                converge_unocc=True,
            ),
            convergence={"eigenstates": 1e-4},
            mixer={"backend": "no-mixing"},
            symmetry="off",
            spinpol=True,
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_lcaosic(self):
        atm = self.h2o_maker(vacuum=4.0)
        atm.calc = GPAW(
            mode=LCAO(force_complex_dtype=True),
            h=0.22,
            occupations={"name": "fixed-uniform"},
            eigensolver=LCAOETDM(
                localizationtype="PM_PZ",
                localizationseed=42,
                functional={"name": "PZ-SIC", "scaling_factor": (0.5, 0.5)},
            ),
            convergence={"eigenstates": 1e-4},
            mixer={"backend": "no-mixing"},
            nbands="nao",
            symmetry="off",
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_mom_lcaosic(self):
        atm = self.h2o_maker(vacuum=3.0)
        calc = GPAW(
            mode=LCAO(force_complex_dtype=True),
            h=0.24,
            basis="sz(dzp)",
            spinpol=True,
            symmetry="off",
            eigensolver="etdm-lcao",
            mixer={"backend": "no-mixing"},
            occupations={"name": "fixed-uniform"},
            convergence={"eigenstates": 1e-4},
        )
        atm.calc = calc
        atm.get_potential_energy()
        atm.calc.set(eigensolver=LCAOETDM(excited_state=True))
        f_sn = excite(atm.calc, 0, 0, (0, 0))
        prepare_mom_calculation(atm.calc, atm, f_sn)
        atm.get_potential_energy()
        atm.calc.set(
            eigensolver=LCAOETDM(
                searchdir_algo={"name": "l-sr1p"},
                linesearch_algo={"name": "max-step"},
                need_init_orbs=False,
                localizationtype="PM_PZ",
                localizationseed=42,
                functional={"name": "pz-sic", "scaling_factor": (0.5, 0.5)},
            ),
            convergence={"eigenstates": 1e-2},
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_gmf_lcaosic(self):
        atm = self.h2o_maker(vacuum=3.0)
        calc = GPAW(
            mode=LCAO(),
            basis="sz(dzp)",
            h=0.24,
            occupations={"name": "fixed-uniform"},
            eigensolver="etdm-lcao",
            convergence={"eigenstates": 1e-4},
            mixer={"backend": "no-mixing"},
            nbands="nao",
            spinpol=True,
            symmetry="off",
        )
        atm.calc = calc
        atm.get_potential_energy()
        atm.calc.set(eigensolver=LCAOETDM(excited_state=True))
        f_sn = excite(atm.calc, 0, 0, (0, 0))
        prepare_mom_calculation(atm.calc, atm, f_sn)
        atm.get_potential_energy()
        dave = SICDavidson(atm.calc.wfs.eigensolver, None)
        appr_sp_order = dave.estimate_sp_order(atm.calc)

        for kpt in atm.calc.wfs.kpt_u:
            f_sn[kpt.s] = kpt.f_n
        atm.calc.set(
            eigensolver=LCAOETDM(
                partial_diagonalizer={
                    "name": "Davidson",
                    "seed": 42,
                    "m": 20,
                    "eps": 5e-3,
                    "remember_sp_order": True,
                    "sp_order": appr_sp_order,
                },
                linesearch_algo={"name": "max-step"},
                searchdir_algo={"name": "LBFGS-P_GMF"},
                localizationtype="PM",
                functional={"name": "PZ-SIC", "scaling_factor": (0.5, 0.5)},
                need_init_orbs=False,
            ),
            occupations={"name": "mom",
                         "numbers": f_sn,
                         "use_fixed_occupations": True},
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_mom_pwsic(self):
        atm = self.h2o_maker(vacuum=3.0)
        calc = GPAW(
            mode=PW(300, force_complex_dtype=True),
            spinpol=True,
            symmetry="off",
            eigensolver=FDPWETDM(converge_unocc=True),
            mixer={"backend": "no-mixing"},
            occupations={"name": "fixed-uniform"},
            convergence={"eigenstates": 1e-4},
        )
        atm.calc = calc
        atm.get_potential_energy()
        atm.calc.set(eigensolver=FDPWETDM(excited_state=True))
        f_sn = excite(atm.calc, 0, 0, (0, 0))
        prepare_mom_calculation(atm.calc, atm, f_sn)
        atm.get_potential_energy()
        atm.calc.set(
            eigensolver=FDPWETDM(
                excited_state=True,
                need_init_orbs=False,
                functional={"name": "PZ-SIC",
                            "scaling_factor": (0.5, 0.5)},  # SIC/2
                localizationseed=42,
                localizationtype="PM",
                grad_tol_pz_localization=1.0e-2,
                printinnerloop=False,
                grad_tol_inner_loop=1.0e-2,
            ),
            convergence={"eigenstates": 1e-3, "density": 1e-3},
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_pwsic(self):
        atm = self.h2o_maker(vacuum=4.0,
                             t=np.pi / 180 * (104.51 + 2.0),
                             eps=0.02)
        atm.calc = GPAW(
            mode=PW(300, force_complex_dtype=True),
            occupations={"name": "fixed-uniform"},
            eigensolver=FDPWETDM(
                functional={"name": "pz-sic", "scaling_factor": (0.5, 0.5)},
                localizationseed=42,
                localizationtype="FB_ER",
                grad_tol_pz_localization=5.0e-3,
                maxiter_pz_localization=200,
                converge_unocc=True,
            ),
            convergence={"eigenstates": 1e-4},
            mixer={"backend": "no-mixing"},
            symmetry="off",
            spinpol=True,
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_mom_do_pw(self):
        atm = self.h2o_maker(vacuum=4.0)
        calc = GPAW(
            mode=PW(300),
            spinpol=True,
            symmetry="off",
            eigensolver={"name": "etdm-fdpw", "converge_unocc": True},
            mixer={"backend": "no-mixing"},
            occupations={"name": "fixed-uniform"},
            convergence={"eigenstates": 1e-4},
            txt=None,
        )
        atm.calc = calc
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def co_mom_do_lcao_forces(self):
        L = 4.0
        d = 1.13
        atoms = Atoms(
            "CO",
            [[0.5 * L, 0.5 * L, 0.5 * L - 0.5 * d],
             [0.5 * L, 0.5 * L, 0.5 * L + 0.5 * d]],
        )
        atoms.set_cell([L, L, L])
        atoms.set_pbc(True)
        calc = GPAW(
            mode="lcao",
            basis="dzp",
            h=0.22,
            xc="PBE",
            spinpol=True,
            symmetry="off",
            occupations={"name": "fixed-uniform"},
            eigensolver={"name": "etdm-lcao", "linesearch_algo": "max-step"},
            mixer={"backend": "no-mixing"},
            nbands="nao",
            convergence={"density": 1.0e-4, "eigenstates": 4.0e-8},
        )
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h2o_mom_do_lcao(self):
        atm = self.h2o_maker(vacuum=4.0)
        atm.calc = GPAW(
            mode=LCAO(),
            basis="dzp",
            h=0.22,
            occupations={"name": "fixed-uniform"},
            eigensolver="etdm-lcao",
            mixer={"backend": "no-mixing"},
            nbands="nao",
            symmetry="off",
            spinpol=True,
            convergence={"density": 1.0e-4, "eigenstates": 4.0e-8},
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h2o_pz_localization_pw(self):
        atm = self.h2o_maker(vacuum=3.0,
                             t=np.pi / 180 * (104.51 + 2.0),
                             eps=0.02)
        atm.calc = GPAW(
            mode=PW(300, force_complex_dtype=True),
            occupations={"name": "fixed-uniform"},
            convergence={
                "energy": np.inf,
                "eigenstates": np.inf,
                "density": np.inf,
                "minimum iterations": 0,
            },
            eigensolver=FDPWETDM(converge_unocc=False),
            mixer={"backend": "no-mixing"},
            symmetry="off",
            spinpol=True,
        )
        atm.get_potential_energy()
        atm.calc.set(
            eigensolver=FDPWETDM(
                functional={"name": "PZ-SIC", "scaling_factor": (0.5, 0.5)},
                localizationseed=42,
                localizationtype="KS_PZ",
                localization_tol=5.0e-2,
                converge_unocc=False,
            )
        )
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def c2h4_do_lcao(self):
        atoms = Atoms(
            "C2H4",
            [
                [6.68748500e-01, 2.00680000e-04, 5.55800000e-05],
                [-6.68748570e-01, -2.00860000e-04, -5.51500000e-05],
                [4.48890600e-01, -5.30146300e-01, 9.32670330e-01],
                [4.48878120e-01, -5.30176640e-01, -9.32674730e-01],
                [-1.24289513e00, 1.46164400e-02, 9.32559990e-01],
                [-1.24286000e00, -1.46832100e-02, -9.32554970e-01],
            ],
        )
        atoms.center(vacuum=4)
        eigensolver = LCAOETDM(
            searchdir_algo={"name": "l-sr1p"},
            linesearch_algo={"name": "max-step"}
        )
        calc = GPAW(
            mode="lcao",
            basis="dzp",
            h=0.24,
            xc="PBE",
            symmetry="off",
            occupations={"name": "fixed-uniform"},
            eigensolver=eigensolver,
            mixer={"backend": "no-mixing"},
            nbands="nao",
            convergence={"density": 1.0e-4, "eigenstates": 4.0e-8},
        )
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h3_orthonorm_lcao(self):
        atoms = Atoms("H3", positions=[(0, 0, 0), (0.59, 0, 0), (1.1, 0, 0)])
        atoms.set_initial_magnetic_moments([1, 0, 0])

        atoms.center(vacuum=2.0)
        atoms.set_pbc(False)
        calc = GPAW(
            mode="lcao",
            basis="sz(dzp)",
            h=0.3,
            spinpol=True,
            convergence={
                "energy": np.inf,
                "eigenstates": np.inf,
                "density": np.inf,
                "minimum iterations": 1,
            },
            eigensolver={"name": "etdm-lcao"},
            occupations={"name": "fixed-uniform"},
            mixer={"backend": "no-mixing"},
            nbands="nao",
            symmetry="off",
            txt=None,
        )
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h2_sic_scfsic(self):
        a = 6.0
        atm = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.737)], cell=(a, a, a))
        atm.center()
        calc = GPAW(mode="fd", xc="LDA-PZ-SIC",
                    eigensolver="rmm-diis", setups="hgh")
        atm.calc = calc
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h_magmom(self):
        a = 6.0
        atm = Atoms("H", magmoms=[1.0], cell=(a, a, a))
        atm.center()
        calc = GPAW(mode="fd", xc="LDA-PZ-SIC",
                    eigensolver="rmm-diis", setups="hgh")
        atm.calc = calc
        atm.get_potential_energy()
        return atm.calc

    @gpwfile
    def h_hess_num_pw(self):
        calc = GPAW(
            xc="PBE",
            mode=PW(300, force_complex_dtype=False),
            h=0.25,
            convergence={
                "energy": np.inf,
                "eigenstates": np.inf,
                "density": np.inf,
                "minimum iterations": 1,
            },
            spinpol=False,
            eigensolver=FDPWETDM(converge_unocc=True),
            occupations={"name": "fixed-uniform"},
            mixer={"backend": "no-mixing"},
            nbands=2,
            symmetry="off",
        )
        atoms = Atoms("H", positions=[[0, 0, 0]])
        atoms.center(vacuum=5.0)
        atoms.set_pbc(False)
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h2_break_ilcao(self):
        atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 2.0)])
        atoms.center(vacuum=2.0)
        atoms.set_pbc(False)
        calc = GPAW(xc='PBE',
                    mode='lcao',
                    h=0.24,
                    basis='sz(dzp)',
                    spinpol=True,
                    eigensolver='etdm-lcao',
                    convergence={'density': 1.0e-2,
                                 'eigenstates': 1.0e-2},
                    occupations={'name': 'fixed-uniform'},
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    symmetry='off')
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h_do_gdavid_lcao(self):
        calc = GPAW(xc='PBE',
                    mode=LCAO(force_complex_dtype=True),
                    h=0.25,
                    basis='dz(dzp)',
                    spinpol=False,
                    eigensolver={'name': 'etdm-lcao',
                                 'representation': 'u-invar'},
                    occupations={'name': 'fixed-uniform'},
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    symmetry='off',
                    )

        atoms = Atoms('H', positions=[[0, 0, 0]])
        atoms.center(vacuum=5.0)
        atoms.set_pbc(False)
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h2_mom_do_pwh(self):
        d = 1.4 * Bohr
        h2 = Atoms('H2',
                   positions=[[-d / 2, 0, 0],
                              [d / 2, 0, 0]])
        h2.center(vacuum=3)
        calc = GPAW(mode=PW(300),
                    # h=0.3,
                    xc={'name': 'HSE06', 'backend': 'pw'},
                    eigensolver={'name': 'etdm-fdpw',
                                 'converge_unocc': True},
                    mixer={'backend': 'no-mixing'},
                    occupations={'name': 'fixed-uniform'},
                    symmetry='off',
                    nbands=2,
                    convergence={'eigenstates': 4.0e-6})
        h2.calc = calc
        h2.get_potential_energy()
        return h2.calc

    @gpwfile
    def h_hess_num_lcao(self):
        calc = GPAW(xc='PBE',
                    mode=LCAO(force_complex_dtype=True),
                    h=0.25,
                    basis='dz(dzp)',
                    spinpol=False,
                    eigensolver={'name': 'etdm-lcao',
                                 'representation': 'u-invar'},
                    occupations={'name': 'fixed-uniform'},
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    symmetry='off',
                    )
        atoms = Atoms('H', positions=[[0, 0, 0]])
        atoms.center(vacuum=5.0)
        atoms.set_pbc(False)
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def silicon_pdens_tool(self):
        # used by response code's pdens tool test
        pw = 200
        kpts = 3
        nbands = 8

        a = 5.431
        atoms = bulk('Si', 'diamond', a=a)

        calc = GPAW(mode=PW(pw),
                    kpts=(kpts, kpts, kpts),
                    nbands=nbands,
                    convergence={'bands': -1},
                    xc='LDA',
                    occupations=FermiDirac(0.001))

        atoms.calc = calc
        atoms.get_potential_energy()
        return calc

    @gpwfile
    def h_pw(self):
        h = Atoms('H', magmoms=[1])
        h.center(vacuum=4.0)
        h.calc = GPAW(mode={'name': 'pw', 'ecut': 500},
                      txt=self.folder / 'h_pw.txt')
        h.get_potential_energy()
        return h.calc

    @gpwfile
    def h_chain(self):
        from gpaw.new.ase_interface import GPAW
        a = 2.5
        k = 4
        """Compare 2*H AFM cell with 1*H q=1/2 spin-spiral cell."""
        h = Atoms('H',
                  magmoms=[1],
                  cell=[a, 0, 0],
                  pbc=[1, 0, 0])
        h.center(vacuum=2.0, axis=(1, 2))
        h.calc = GPAW(mode={'name': 'pw',
                            'ecut': 400,
                            'qspiral': [0.5, 0, 0]},
                      magmoms=[[1, 0, 0]],
                      symmetry='off',
                      kpts=(2 * k, 1, 1),
                      txt=self.folder / 'h_chain.txt')
        h.get_potential_energy()
        return h.calc

    @gpwfile
    def h_shift(self):
        # Check for Hydrogen atom
        atoms = Atoms('H', cell=(3 * np.eye(3)), pbc=True)

        # Do a GS and save it
        calc = GPAW(
            mode=PW(600), symmetry={'point_group': False},
            kpts={'size': (2, 2, 2)}, nbands=5, txt=None)
        atoms.calc = calc
        atoms.get_potential_energy()

        return atoms.calc

    @gpwfile
    def h2_chain(self):
        a = 2.5
        k = 4
        h2 = Atoms('H2',
                   [(0, 0, 0), (a, 0, 0)],
                   magmoms=[1, -1],
                   cell=[2 * a, 0, 0],
                   pbc=[1, 0, 0])
        h2.center(vacuum=2.0, axis=(1, 2))
        h2.calc = GPAW(mode={'name': 'pw',
                             'ecut': 400},
                       kpts=(k, 1, 1),
                       txt=self.folder / 'h2_chain.txt')
        h2.get_potential_energy()
        return h2.calc

    @gpwfile
    def h2_lcao_pair(self):
        atoms = molecule('H2')
        atoms.set_cell([6.4, 6.4, 6.4])
        atoms.center()

        atoms.calc = GPAW(mode='lcao', occupations=FermiDirac(0.1),
                          poissonsolver={'name': 'fd'},
                          txt=self.folder / 'h2_lcao_pair.txt')
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def na_chain(self):
        a = 3
        atom = Atoms('Na',
                     cell=[a, 0, 0],
                     pbc=[1, 1, 1])
        atom.center(vacuum=1 * a, axis=(1, 2))
        atom.center()
        atoms = atom.repeat((2, 1, 1))

        calc = GPAW(mode=PW(200),
                    nbands=4,
                    setups={'Na': '1'},
                    txt=self.folder / 'na_chain.txt',
                    kpts=(16, 2, 2))

        atoms.calc = calc
        atoms.get_potential_energy()
        return calc

    @gpwfile
    def n2_pw(self):
        N2 = molecule('N2')
        N2.center(vacuum=2.0)

        N2.calc = GPAW(mode=PW(force_complex_dtype=True),
                       xc='PBE',
                       parallel={'domain': 1},
                       eigensolver='rmm-diis',
                       txt=self.folder / 'n2_pw.txt')

        N2.get_potential_energy()
        N2.calc.diagonalize_full_hamiltonian(nbands=104, scalapack=True)
        return N2.calc

    @gpwfile
    def n_pw(self):
        N2 = molecule('N2')
        N2.center(vacuum=2.0)

        N = molecule('N')
        N.set_cell(N2.cell)
        N.center()

        N.calc = GPAW(mode=PW(force_complex_dtype=True),
                      xc='PBE',
                      parallel={'domain': 1},
                      eigensolver='rmm-diis',
                      txt=self.folder / 'n_pw.txt')
        N.get_potential_energy()
        N.calc.diagonalize_full_hamiltonian(nbands=104, scalapack=True)
        return N.calc

    @gpwfile
    def o2_pw(self):
        d = 1.1
        a = Atoms('O2', positions=[[0, 0, 0], [d, 0, 0]], magmoms=[1, 1])
        a.center(vacuum=4.0)
        a.calc = GPAW(mode={'name': 'pw', 'ecut': 800},
                      txt=self.folder / 'o2_pw.txt')
        a.get_potential_energy()
        return a.calc

    @gpwfile
    def Cu3Au_qna(self):
        ecut = 300
        kpts = (1, 1, 1)

        QNA = {'alpha': 2.0,
               'name': 'QNA',
               'stencil': 1,
               'orbital_dependent': False,
               'parameters': {'Au': (0.125, 0.1), 'Cu': (0.0795, 0.005)},
               'setup_name': 'PBE',
               'type': 'qna-gga'}

        atoms = L1_2(['Au', 'Cu'], latticeconstant=3.7)
        atoms[0].position[0] += 0.01  # Break symmetry already here
        calc = GPAW(mode=PW(ecut),
                    eigensolver=Davidson(2),
                    nbands='120%',
                    mixer=Mixer(0.4, 7, 50.0),
                    parallel=dict(domain=1),
                    convergence={'density': 1e-4},
                    xc=QNA,
                    kpts=kpts,
                    txt=self.folder / 'Cu3Au_qna.txt')
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def co_mom(self):
        atoms = molecule('CO')
        atoms.center(vacuum=2)

        calc = GPAW(mode='lcao',
                    basis='dzp',
                    nbands=7,
                    h=0.24,
                    xc='PBE',
                    spinpol=True,
                    symmetry='off',
                    convergence={'energy': 100,
                                 'density': 1e-3},
                    txt=self.folder / 'co_mom.txt')

        atoms.calc = calc
        # Ground-state calculation
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def co_lcao(self):
        d = 1.1
        co = Atoms('CO', positions=[[0, 0, 0], [d, 0, 0]])
        co.center(vacuum=4.0)
        co.calc = GPAW(mode='lcao',
                       txt=self.folder / 'co_lcao.txt')
        co.get_potential_energy()
        return co.calc

    def _c2h4(self):
        d = 1.54
        h = 1.1
        x = d * (2 / 3)**0.5
        z = d / 3**0.5
        pe = Atoms('C2H4',
                   positions=[[0, 0, 0],
                              [x, 0, z],
                              [0, -h * (2 / 3)**0.5, -h / 3**0.5],
                              [0, h * (2 / 3)**0.5, -h / 3**0.5],
                              [x, -h * (2 / 3)**0.5, z + h / 3**0.5],
                              [x, h * (2 / 3)**0.5, z + h / 3**0.5]],
                   cell=[2 * x, 0, 0],
                   pbc=(1, 0, 0))
        pe.center(vacuum=2.0, axis=(1, 2))
        return pe

    def c2h4_pw_nosym(self):
        pe = self._c2h4()
        pe.calc = GPAW(mode='pw',
                       kpts=(3, 1, 1),
                       symmetry='off',
                       txt=self.folder / 'c2h4_pw_nosym.txt')
        pe.get_potential_energy()
        return pe.calc

    @gpwfile
    def c6h12_pw(self):
        pe = self._c2h4()
        pe = pe.repeat((3, 1, 1))
        pe.calc = GPAW(mode='pw', txt=self.folder / 'c6h12_pw.txt')
        pe.get_potential_energy()
        return pe.calc

    @gpwfile
    def h2o_lcao(self):
        atoms = molecule('H2O', cell=[8, 8, 8], pbc=1)
        atoms.center()
        atoms.calc = GPAW(mode='lcao', txt=self.folder / 'h2o_lcao.txt')
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h2o_xas(self):
        setupname = 'h2o_xas_hch1s'
        self.generator2_setup(
            'O', 8, '2s,s,2p,p,d', [1.2], 1.0, None, 2,
            core_hole='1s,0.5',
            name=setupname)

        a = 5.0
        H2O = self.h2o_maker(vacuum=None, cell=[a, a, a], pbc=False)
        calc = GPAW(
            txt=self.folder / 'h2o_xas.txt',
            mode='fd',
            nbands=10,
            h=0.2,
            setups={'O': 'h2o_xas_hch1s'},
            poissonsolver=FDPoissonSolver(use_charge_center=True),
        )
        H2O.calc = calc
        _ = H2O.get_potential_energy()
        return calc

    @gpwfile
    def h20_lr2_nbands8(self):
        return self.h2o_nbands(
            {'poissonsolver': {'name': 'fd'}, 'nbands': 8, 'basis': 'dzp'})

    @gpwfile
    def h20_lr2_nbands6(self):
        return self.h2o_nbands({'nbands': 6, 'basis': 'sz(dzp)'})

    def h2o_nbands(self, params):
        atoms = molecule('H2O')
        atoms.center(vacuum=4)

        calc = GPAW(h=0.4, **params, xc='LDA', mode='lcao',
                    txt=self.folder / f'h2o_lr2_nbands{params["nbands"]}.out')
        atoms.calc = calc
        atoms.get_potential_energy()

        return atoms.calc

    @gpwfile
    def si_fd_ibz(self):
        si = bulk('Si', 'diamond', a=5.43)
        k = 3
        si.calc = GPAW(mode='fd', kpts=(k, k, k),
                       txt=self.folder / 'si_fd_ibz.txt')
        si.get_potential_energy()
        return si.calc

    @gpwfile
    def si_fd_bz(self):
        si = bulk('Si', 'diamond', a=5.43)
        k = 3
        si.calc = GPAW(mode='fd', kpts=(k, k, k,),
                       symmetry={'point_group': False,
                                 'time_reversal': False},
                       txt=self.folder / 'si_fd_bz.txt')
        si.get_potential_energy()
        return si.calc

    @gpwfile
    def si_pw(self):
        si = bulk('Si')
        calc = GPAW(mode='pw',
                    xc='LDA',
                    occupations=FermiDirac(width=0.001),
                    kpts={'size': (2, 2, 2), 'gamma': True},
                    txt=self.folder / 'si_pw.txt')
        si.calc = calc
        si.get_potential_energy()
        return si.calc

    @gpwfile
    def si_noisy_kpoints(self):
        # Test system for guarding against inconsistent kpoints as in #1178.
        from ase.calculators.calculator import kpts2kpts
        atoms = bulk('Si')
        kpts = kpts2kpts(kpts={'size': (2, 2, 2), 'gamma': True}, atoms=atoms)

        # Error happened when qpoint was ~1e-17 yet was not considered gamma.
        # Add a bit of noise on purpose so we are sure to hit such a case,
        # even if the underlying implementation changes:
        kpts.kpts += np.linspace(1e-16, 1e-15, 24).reshape(8, 3)

        calc = GPAW(mode='pw',
                    xc='LDA',
                    occupations=FermiDirac(width=0.001),
                    kpts=kpts.kpts,
                    txt=self.folder / 'si_noisy_kpoints.txt')
        atoms.calc = calc
        atoms.get_potential_energy()
        return calc

    @gpwfile
    def si_pw_nbands10_converged(self):
        calc = GPAW(mode='pw',
                    kpts={'size': (2, 2, 2), 'gamma': True},
                    occupations=FermiDirac(0.01),
                    nbands=10,
                    symmetry='off',
                    convergence={'bands': -4, 'density': 1e-7,
                                 'eigenstates': 1e-10})

        atoms = bulk('Si', 'diamond', a=5.431)
        atoms.calc = calc
        atoms.get_potential_energy()
        return calc

    @property
    def testing_setup_path(self):
        # Some calculations in gpwfile fixture like to use funny setups.
        # This is not so robust since the setups will be all jumbled.
        # We could improve the mechanism by programmatic naming/subfolders.
        return self.folder / 'setups'

    def save_setup(self, setup):
        self.testing_setup_path.mkdir(parents=True, exist_ok=True)
        setup_file = self.testing_setup_path / setup.stdfilename
        if world.rank == 0:
            setup.write_xml(setup_file)
        world.barrier()
        return setup

    def generate_setup(self, *args, **kwargs):
        from gpaw.test import gen
        setup = gen(*args, **kwargs, write_xml=False)
        self.save_setup(setup)
        return setup

    def generator2_setup(self, *args, name, **kwargs):
        from gpaw.atom.generator2 import generate
        gen = generate(*args, **kwargs)
        setup = gen.make_paw_setup(name)
        self.save_setup(setup)
        return setup

    @gpwfile
    def si_corehole_pw(self):
        # Generate setup for oxygen with half a core-hole:
        setupname = 'si_corehole_pw_hch1s'
        self.generate_setup('Si', name=setupname,
                            corehole=(1, 0, 0.5), gpernode=30)

        a = 2.6
        si = Atoms('Si', cell=(a, a, a), pbc=True)

        calc = GPAW(mode='fd',
                    txt=self.folder / 'si_corehole_pw.txt',
                    h=0.25,
                    occupations=FermiDirac(width=0.05),
                    setups='si_corehole_pw_hch1s',
                    convergence={'maximum iterations': 1})
        si.calc = calc
        _ = si.get_potential_energy()
        return si.calc

    @gpwfile
    def si_corehole_sym_pw(self):
        setupname = 'si_corehole_sym_pw_hch1s'
        self.generate_setup('Si', name=setupname, corehole=(1, 0, 0.5),
                            gpernode=30)
        return self.si_corehole_sym(sym={}, setupname=setupname)

    @gpwfile
    def si_corehole_nosym_pw(self):
        setupname = 'si_corehole_sym_pw_hch1s'
        # XXX same setup as above, but we have it twice since caching
        # works per gpw file and not per setup
        self.generate_setup('Si', name=setupname, corehole=(1, 0, 0.5),
                            gpernode=30)
        return self.si_corehole_sym(sym='off', setupname=setupname)

    def si_corehole_sym(self, sym, setupname):
        tag = 'nosym' if sym == 'off' else 'sym'

        a = 5.43095
        si_nonortho = Atoms(
            [Atom('Si', (0, 0, 0)), Atom('Si', (a / 4, a / 4, a / 4))],
            cell=[(a / 2, a / 2, 0), (a / 2, 0, a / 2), (0, a / 2, a / 2)],
            pbc=True,
        )
        # calculation with full symmetry
        calc = GPAW(
            mode='fd',
            txt=self.folder / f'si_corehole_{tag}_hch1s.txt',
            nbands=-10,
            h=0.25,
            kpts=(2, 2, 2),
            occupations=FermiDirac(width=0.05),
            setups={0: setupname},
            symmetry=sym
        )
        si_nonortho.calc = calc
        _ = si_nonortho.get_potential_energy()
        return calc

    @gpwfile
    @with_band_cutoff(gpw='fancy_si_pw',
                      band_cutoff=8)  # 2 * (3s, 3p)
    def _fancy_si(self, *, band_cutoff, symmetry=None):
        if symmetry is None:
            symmetry = {}
        xc = 'LDA'
        kpts = 4
        pw = 300
        occw = 0.01
        conv = {'bands': band_cutoff + 1,
                'density': 1.e-8}
        atoms = bulk('Si')
        atoms.center()

        tag = '_nosym' if symmetry == 'off' else ''
        atoms.calc = GPAW(
            xc=xc,
            mode=PW(pw),
            kpts={'size': (kpts, kpts, kpts), 'gamma': True},
            nbands=band_cutoff + 12,  # + 2 * (3s, 3p),
            occupations=FermiDirac(occw),
            convergence=conv,
            txt=self.folder / f'fancy_si_pw{tag}.txt',
            symmetry=symmetry)

        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def fancy_si_pw(self):
        return self._fancy_si()

    @gpwfile
    def fancy_si_pw_nosym(self):
        return self._fancy_si(symmetry='off')

    @with_band_cutoff(gpw='sic_pw',
                      band_cutoff=8)  # (3s, 3p) + (2s, 2p)
    def _sic_pw(self, *, band_cutoff, spinpol=False):
        """Simple semi-conductor with broken inversion symmetry."""
        # Use the diamond crystal structure as blue print
        diamond = bulk('C', 'diamond')
        si = bulk('Si', 'diamond')
        # Break inversion symmetry by substituting one Si for C
        atoms = si.copy()
        atoms.symbols = 'CSi'
        # Scale the cell to the diamond/Si average
        cell_cv = (diamond.get_cell() + si.get_cell()) / 2.
        atoms.set_cell(cell_cv)

        # Set up calculator
        tag = '_spinpol' if spinpol else ''
        atoms.calc = GPAW(
            mode=PW(400),
            xc='LDA',
            kpts={'size': (4, 4, 4)},
            symmetry={'point_group': False,
                      'time_reversal': True},
            nbands=band_cutoff + 6,
            occupations=FermiDirac(0.001),
            convergence={'bands': band_cutoff + 1,
                         'density': 1e-8},
            spinpol=spinpol,
            txt=self.folder / f'sic_pw{tag}.txt'
        )

        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def sic_pw(self):
        return self._sic_pw()

    @gpwfile
    def sic_pw_spinpol(self):
        return self._sic_pw(spinpol=True)

    @staticmethod
    def generate_si_systems():
        a = 5.43
        si1 = bulk('Si', 'diamond', a=a)
        si2 = si1.copy()
        si2.positions -= a / 8
        return [si1, si2]

    def _si_gw(self, atoms, symm, name):
        atoms.calc = GPAW(mode=PW(250),
                          eigensolver='rmm-diis',
                          occupations=FermiDirac(0.01),
                          symmetry=symm,
                          kpts={'size': (2, 2, 2), 'gamma': True},
                          convergence={'density': 1e-7},
                          parallel={'domain': 1},
                          txt=self.folder / name)
        atoms.get_potential_energy()
        scalapack = atoms.calc.wfs.bd.comm.size
        atoms.calc.diagonalize_full_hamiltonian(nbands=8, scalapack=scalapack)
        return atoms.calc

    @gpwfile
    def c2_gw_more_bands(self):
        a = 3.567
        atoms = bulk('C', 'diamond', a=a)
        atoms.calc = GPAW(mode=PW(400),
                          parallel={'domain': 1},
                          kpts={'size': (2, 2, 2), 'gamma': True},
                          xc='LDA',
                          occupations=FermiDirac(0.001))
        atoms.get_potential_energy()
        atoms.calc.diagonalize_full_hamiltonian(nbands=128)
        return atoms.calc

    @gpwfile
    def na_pw(self):
        from ase.build import bulk

        blk = bulk('Na', 'bcc', a=4.23)

        ecut = 350
        blk.calc = GPAW(mode=PW(ecut),
                        basis='dzp',
                        kpts={'size': (4, 4, 4), 'gamma': True},
                        parallel={'domain': 1, 'band': 1},
                        txt=self.folder / 'na_pw.txt',
                        nbands=4,
                        occupations=FermiDirac(0.01),
                        setups={'Na': '1'})
        blk.get_potential_energy()
        blk.calc.diagonalize_full_hamiltonian(nbands=520)
        return blk.calc

    @gpwfile
    def na2_tddft_poisson_sym(self):
        return self._na2_tddft(name='poisson_sym', basis='dzp', xc='LDA')

    @gpwfile
    def na2_tddft_poisson(self):
        return self._na2_tddft(name='poisson', basis='dzp', xc='LDA')

    @gpwfile
    def na2_tddft_dzp(self):
        return self._na2_tddft(name='dzp', basis='dzp', xc='LDA')

    @gpwfile
    def na2_tddft_sz(self):
        return self._na2_tddft(name='sz', basis='sz(dzp)', xc='oldLDA')

    def _na2_tddft(self, name, basis, xc):
        atoms = molecule('Na2')
        atoms.center(vacuum=4.0)

        names = ['poisson_sym', 'poisson']
        poisson = PoissonSolver('fd', eps=1e-16) if name in names \
            else None
        symmetry = {'point_group': True if name == 'poisson_sym' else False}

        calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
                    basis=basis, mode='lcao', xc=xc,
                    convergence={'density': 1e-8},
                    poissonsolver=poisson,
                    communicator=serial_comm if xc == 'oldLDA' else world,
                    symmetry=symmetry,
                    txt=self.folder / f'na2_tddft_{name}.out')
        atoms.calc = calc
        atoms.get_potential_energy()

        return calc

    @gpwfile
    def na2_fd(self):
        """Sodium dimer, Na2."""
        d = 1.5
        atoms = Atoms(symbols='Na2',
                      positions=[(0, 0, d),
                                 (0, 0, -d)],
                      pbc=False)

        atoms.center(vacuum=6.0)
        # Larger grid spacing, LDA is ok
        gs_calc = GPAW(mode='fd',
                       txt=self.folder / 'na2_fd.txt',
                       nbands=1, h=0.35, xc='LDA',
                       setups={'Na': '1'},
                       symmetry={'point_group': False})
        atoms.calc = gs_calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def na2_fd_with_sym(self):
        """Sodium dimer, Na2."""
        d = 1.5
        atoms = Atoms(symbols='Na2',
                      positions=[(0, 0, d),
                                 (0, 0, -d)],
                      pbc=False)

        atoms.center(vacuum=6.0)
        # Larger grid spacing, LDA is ok
        gs_calc = GPAW(mode='fd', nbands=1, h=0.35, xc='LDA',
                       txt=self.folder / 'na2_fd_with_sym.txt',
                       setups={'Na': '1'})
        atoms.calc = gs_calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def na2_isolated(self):
        # Permittivity file
        if world.rank == 0:
            fo = open(self.folder / 'ed.txt', 'w')
            fo.writelines(['1.20 0.20 25.0'])
            fo.close()
        world.barrier()

        from gpaw.fdtd.poisson_fdtd import FDTDPoissonSolver
        from gpaw.fdtd.polarizable_material import (
            PermittivityPlus,
            PolarizableMaterial,
            PolarizableSphere,
        )

        # Whole simulation cell (Angstroms)
        large_cell = [20, 20, 30]

        # Quantum subsystem
        atom_center = np.array([10.0, 10.0, 20.0])
        atoms = Atoms('Na2', [atom_center + [0.0, 0.0, -1.50],
                              atom_center + [0.0, 0.0, +1.50]])

        # Classical subsystem
        classical_material = PolarizableMaterial()
        sphere_center = np.array([10.0, 10.0, 10.0])
        classical_material.add_component(
            PolarizableSphere(
                permittivity=PermittivityPlus(self.folder / 'ed.txt'),
                center=sphere_center,
                radius=5.0
            )
        )

        # Accuracy
        energy_eps = 0.0005
        density_eps = 1e-6
        poisson_eps = 1e-12

        # Combined Poisson solver
        poissonsolver = FDTDPoissonSolver(
            classical_material=classical_material,
            eps=poisson_eps,
            qm_spacing=0.40,
            cl_spacing=0.40 * 4,
            cell=large_cell,
            remove_moments=(1, 4),
            communicator=world,
            potential_coupler='Refiner',
        )
        poissonsolver.set_calculation_mode('iterate')

        # Combined system
        atoms.set_cell(large_cell)
        atoms, qm_spacing, gpts = poissonsolver.cut_cell(atoms, vacuum=2.50)

        # Initialize GPAW
        gs_calc = GPAW(mode='fd',
                       txt=self.folder / 'na2_isolated.txt',
                       gpts=gpts,
                       eigensolver='cg',
                       nbands=-1,
                       poissonsolver=poissonsolver,
                       symmetry={'point_group': False},
                       convergence={'energy': energy_eps,
                                    'density': density_eps})
        atoms.calc = gs_calc

        # Ground state
        atoms.get_potential_energy()

        return gs_calc

    @gpwfile
    def na3_pw_restart(self):
        params = dict(mode=PW(200), convergence={
            'eigenstates': 1.24, 'energy': 2e-1, 'density': 1e-1})
        return self._na3_restart(params=params)

    @gpwfile
    def na3_fd_restart(self):
        params = dict(mode='fd', h=0.30, convergence={
            'eigenstates': 1.24, 'energy': 2e-1, 'density': 1e-1})
        return self._na3_restart(params=params)

    @gpwfile
    def na3_fd_kp_restart(self):
        params = dict(mode='fd', h=0.30, kpts=(1, 1, 3), convergence={
            'eigenstates': 1.24, 'energy': 2e-1, 'density': 1e-1})
        return self._na3_restart(params=params)

    @gpwfile
    def na3_fd_density_restart(self):
        params = dict(mode='fd', h=0.30, convergence={
            'eigenstates': 1.e-3, 'energy': 2e-1, 'density': 1e-1})
        return self._na3_restart(params=params)

    def _na3_restart(self, params):
        d = 3.0
        atoms = Atoms('Na3',
                      positions=[(0, 0, 0),
                                 (0, 0, d),
                                 (0, d * sqrt(3 / 4), d / 2)],
                      magmoms=[1.0, 1.0, 1.0],
                      cell=(3.5, 3.5, 4 + 2 / 3),
                      pbc=True)

        atoms.calc = GPAW(nbands=3,
                          setups={'Na': '1'},
                          **params)
        atoms.get_potential_energy()

        return atoms.calc

    @gpwfile
    def sih4_xc_gllbsc_lcao(self):
        return self._sih4_gllbsc(mode='lcao', basis='dzp')

    @gpwfile
    def sih4_xc_gllbsc_fd(self):
        return self._sih4_gllbsc(mode='fd', basis={})

    def _sih4_gllbsc(self, mode, basis):
        atoms = molecule('SiH4')
        atoms.center(vacuum=4.0)

        # Ground-state calculation
        calc = GPAW(mode=mode, nbands=7, h=0.4,
                    convergence={'density': 1e-8},
                    xc='GLLBSC',
                    basis=basis,
                    symmetry={'point_group': False},
                    txt=self.folder / f'sih4_xc_gllbsc_{mode}.txt')
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def nacl_nospin(self):
        return self._nacl_mol(spinpol=False)

    @gpwfile
    def nacl_spin(self):
        return self._nacl_mol(spinpol=True)

    def _nacl_mol(self, spinpol):
        atoms = molecule('NaCl')
        atoms.center(vacuum=4.0)
        calc = GPAW(nbands=6,
                    h=0.4,
                    setups=dict(Na='1'),
                    basis='dzp',
                    mode='lcao',
                    convergence={'density': 1e-8},
                    spinpol=spinpol,
                    communicator=world,
                    symmetry={'point_group': False},
                    txt=self.folder / 'gs.out')
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def nacl_fd(self):
        d = 4.0
        atoms = Atoms('NaCl', [(0, 0, 0), (0, 0, d)])
        atoms.center(vacuum=4.5)

        gs_calc = GPAW(
            txt=self.folder / 'nacl_fd.txt',
            mode='fd', nbands=4,  # eigensolver='cg',
            gpts=(32, 32, 44), xc='LDA', symmetry={'point_group': False},
            setups={'Na': '1'})
        atoms.calc = gs_calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def bn_pw(self):
        atoms = bulk('BN', 'zincblende', a=3.615)
        atoms.calc = GPAW(mode=PW(400),
                          kpts={'size': (2, 2, 2), 'gamma': True},
                          nbands=12,
                          convergence={'bands': 9},
                          occupations=FermiDirac(0.001),
                          txt=self.folder / 'bn_pw.txt')
        atoms.get_potential_energy()
        return atoms.calc

    def _hbn_pw(self, symmetry={}):
        atoms = Graphene(symbol='B',
                         latticeconstant={'a': 2.5, 'c': 1.0},
                         size=(1, 1, 1))
        atoms[0].symbol = 'N'
        atoms.pbc = (1, 1, 0)
        atoms.center(axis=2, vacuum=3.0)
        atoms.calc = GPAW(txt=self.folder / 'hbn_pw.txt',
                          mode=PW(400),
                          xc='LDA',
                          nbands=50,
                          occupations=FermiDirac(0.001),
                          parallel={'domain': 1},
                          convergence={'bands': 26},
                          kpts={'size': (3, 3, 1), 'gamma': True},
                          symmetry=symmetry)
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def hbn_pw(self):
        return self._hbn_pw()

    @gpwfile
    def hbn_pw_nosym(self):
        return self._hbn_pw(symmetry='off')

    @gpwfile
    def graphene_pw(self):
        from ase.lattice.hexagonal import Graphene
        atoms = Graphene(symbol='C',
                         latticeconstant={'a': 2.45, 'c': 1.0},
                         size=(1, 1, 1))
        atoms.pbc = (1, 1, 0)
        atoms.center(axis=2, vacuum=4.0)
        ecut = 250
        nkpts = 6
        atoms.calc = GPAW(mode=PW(ecut),
                          kpts={'size': (nkpts, nkpts, 1), 'gamma': True},
                          nbands=len(atoms) * 6,
                          txt=self.folder / 'graphene_pw.txt')
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def i2sb2_pw_nosym(self):
        # Structure from c2db
        atoms = Atoms('I2Sb2',
                      positions=[[0.02437357, 0.05048655, 6.11612164],
                                 [0.02524896, 3.07135573, 11.64646853],
                                 [0.02717742, 0.01556495, 8.89278807],
                                 [0.02841809, 3.10675382, 8.86983839]],
                      cell=[[5.055642258802973, -9.89475498615942e-15, 0.0],
                            [-2.5278211265136266, 4.731999711338355, 0.0],
                            [3.38028806436979e-15, 0.0, 18.85580293064]],
                      pbc=(1, 1, 0))
        atoms.calc = GPAW(mode=PW(250),
                          xc='PBE',
                          kpts={'size': (6, 6, 1), 'gamma': True},
                          txt=self.folder / 'i2sb2_pw_nosym.txt',
                          symmetry='off')

        atoms.get_potential_energy()
        return atoms.calc

    @with_band_cutoff(gpw='bi2i6_pw',
                      band_cutoff=36)
    def _bi2i6(self, *, band_cutoff, symmetry=None):
        if symmetry is None:
            symmetry = {}
        positions = [[4.13843656, 2.38932746, 9.36037077],
                     [0.00000000, 4.77865492, 9.36034750],
                     [3.89827619, 0.00000000, 7.33713295],
                     [2.18929748, 3.79197674, 7.33713295],
                     [-1.94913711, 3.37600678, 7.33713295],
                     [3.89827619, 0.00000000, 11.3835853],
                     [2.18929961, 3.79197551, 11.3835853],
                     [-1.94913924, 3.37600555, 11.3835853]]
        cell = [[8.276873113486648, 0.0, 0.0],
                [-4.138436556743325, 7.167982380179831, 0.0],
                [0.0, 0.0, 0.0]]
        pbc = [True, True, False]
        atoms = Atoms('Bi2I6',
                      positions=positions,
                      cell=cell,
                      pbc=pbc)
        atoms.center(vacuum=3.0, axis=2)

        ecut = 120
        nkpts = 4
        conv = {'bands': band_cutoff + 1,
                'density': 1.e-4}

        tag = '_nosym' if symmetry == 'off' else ''
        atoms.calc = GPAW(mode=PW(ecut),
                          xc='LDA',
                          kpts={'size': (nkpts, nkpts, 1), 'gamma': True},
                          occupations=FermiDirac(0.01),
                          mixer={'beta': 0.5},
                          convergence=conv,
                          nbands=band_cutoff + 9,
                          txt=self.folder / f'bi2i6_pw{tag}.txt',
                          symmetry=symmetry)

        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def bi2i6_pw(self):
        return self._bi2i6()

    @gpwfile
    def bi2i6_pw_nosym(self):
        return self._bi2i6(symmetry='off')

    def _mos2(self, symmetry=None):
        if symmetry is None:
            symmetry = {}
        from ase.build import mx2
        atoms = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.127,
                    size=(1, 1, 1), vacuum=5)
        atoms.pbc = (1, 1, 0)
        ecut = 250
        nkpts = 6
        tag = '_nosym' if symmetry == 'off' else ''
        atoms.calc = GPAW(mode=PW(ecut),
                          xc='LDA',
                          kpts={'size': (nkpts, nkpts, 1), 'gamma': True},
                          occupations=FermiDirac(0.01),
                          txt=self.folder / f'mos2_pw{tag}.txt',
                          symmetry=symmetry)

        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def mos2_pw(self):
        return self._mos2()

    @gpwfile
    def mos2_pw_nosym(self):
        return self._mos2(symmetry='off')

    @gpwfile
    def mos2_5x5_pw(self):
        calc = GPAW(mode=PW(180),
                    xc='PBE',
                    nbands='nao',
                    setups={'Mo': '6'},
                    occupations=FermiDirac(0.001),
                    convergence={'bands': -5},
                    kpts=(5, 5, 1))

        from ase.build import mx2
        layer = mx2(formula='MoS2', kind='2H', a=3.1604, thickness=3.172,
                    size=(1, 1, 1), vacuum=3.414)
        layer.pbc = (1, 1, 0)
        layer.calc = calc
        layer.get_potential_energy()
        return layer.calc

    @with_band_cutoff(gpw='p4_pw',
                      band_cutoff=40)
    def _p4(self, band_cutoff, spinpol=False):
        atoms = Atoms('P4', positions=[[0.03948480, -0.00027057, 7.49990646],
                                       [0.86217564, -0.00026338, 9.60988536],
                                       [2.35547782, 1.65277230, 9.60988532],
                                       [3.17816857, 1.65277948, 7.49990643]],
                      cell=[4.63138807675, 3.306178252090, 17.10979291],
                      pbc=[True, True, False])
        atoms.center(vacuum=1.5, axis=2)
        tag = '_spinpol' if spinpol else ''
        nkpts = 2
        atoms.calc = GPAW(mode=PW(250),
                          xc='LDA', spinpol=spinpol,
                          kpts={'size': (nkpts, nkpts, 1), 'gamma': True},
                          occupations={'width': 0},
                          nbands=band_cutoff + 10,
                          convergence={'bands': band_cutoff + 1},
                          txt=self.folder / f'p4_pw{tag}.txt')
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def p4_pw(self):
        return self._p4()

    @gpwfile
    def p4_pw_spinpol(self):
        return self._p4(spinpol=True)

    def _ni_pw_kpts333(self, setups={'Ni': '10'}):
        from ase.dft.kpoints import monkhorst_pack
        # from gpaw.mpi import serial_comm
        Ni = bulk('Ni', 'fcc')
        Ni.set_initial_magnetic_moments([0.7])

        kpts = monkhorst_pack((3, 3, 3))

        calc = GPAW(mode='pw',
                    txt=self.folder / 'ni_pw_kpts333.txt',
                    kpts=kpts,
                    occupations=FermiDirac(0.001),
                    setups=setups,
                    parallel=dict(domain=1),  # >1 fails on 8 cores
                    # communicator=serial_comm
                    )

        Ni.calc = calc
        Ni.get_potential_energy()
        calc.diagonalize_full_hamiltonian()
        return calc

    @gpwfile
    def ni_pw(self):
        return self._ni_pw_kpts333(setups={})

    @gpwfile
    def ni_pw_kpts333(self):
        return self._ni_pw_kpts333()

    @gpwfile
    def c_pw(self):
        atoms = bulk('C')
        atoms.center()
        calc = GPAW(mode=PW(150),
                    txt=self.folder / 'c_pw.txt',
                    convergence={'bands': 6},
                    nbands=12,
                    kpts={'gamma': True, 'size': (2, 2, 2)},
                    xc='LDA')

        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    def _nicl2_pw(self, vacuum=3.0, identifier='', ecut=285, **kwargs):
        from ase.build import mx2

        # Define input parameters
        kpts = 6
        occw = 0.01
        conv = {'density': 1.e-3}

        a = 3.502
        thickness = 2.617
        mm = 2.0

        # Set up atoms
        atoms = mx2(formula='NiCl2', kind='1T', a=a,
                    thickness=thickness, vacuum=vacuum)
        atoms.set_initial_magnetic_moments([mm, 0.0, 0.0])
        # Use pbc to allow for real-space density interpolation
        atoms.pbc = True

        dct = dict(
            mixer={'beta': 0.75, 'nmaxold': 8, 'weight': 100.0},
            mode=PW(ecut,
                    # Interpolate the density in real-space
                    interpolation=3),
            kpts={'size': (kpts, kpts, 1), 'gamma': True},
            occupations=FermiDirac(occw),
            convergence=conv,
            txt=self.folder / f'nicl2_pw{identifier}.txt')
        dct.update(kwargs)
        atoms.calc = GPAW(**dct)

        atoms.get_potential_energy()

        return atoms.calc

    @gpwfile
    def nicl2_pw(self):
        return self._nicl2_pw(vacuum=3.0, ecut=300.0)

    @gpwfile
    def nicl2_pw_evac(self):
        return self._nicl2_pw(vacuum=10.0, identifier='_evac')

    @gpwfile
    def Tl_box_pw(self):
        Tl = Atoms('Tl',
                   cell=[5, 5, 5],
                   scaled_positions=[[0.5, 0.5, 0.5]],
                   pbc=False)

        Tl.calc = GPAWNew(
            mode={'name': 'pw', 'ecut': 300},
            xc='LDA',
            occupations={'name': 'fermi-dirac', 'width': 0.01},
            symmetry='off',
            convergence={'density': 1e-6},
            parallel={'domain': 1, 'band': 1},
            magmoms=[[0, 0, 0.5]],
            soc=True,
            txt=self.folder / 'Tl_box_pw.txt')
        Tl.get_potential_energy()
        return Tl.calc

    @with_band_cutoff(gpw='v2br4_pw',
                      band_cutoff=28)  # V(4s,3d) = 6, Br(4s,4p) = 4
    def _v2br4(self, *, band_cutoff, symmetry=None):
        from ase.build import mx2

        if symmetry is None:
            symmetry = {}

        # Define input parameters
        xc = 'LDA'
        kpts = 4
        pw = 200
        occw = 0.01
        conv = {'density': 1.e-4,
                'bands': band_cutoff + 1}

        a = 3.840
        thickness = 2.897
        vacuum = 3.0
        mm = 3.0

        # Set up atoms
        atoms = mx2(formula='VBr2', kind='1T', a=a,
                    thickness=thickness, vacuum=vacuum)
        atoms = atoms.repeat((1, 2, 1))
        atoms.set_initial_magnetic_moments([mm, 0.0, 0.0, -mm, 0.0, 0.0])
        # Use pbc to allow for real-space density interpolation
        atoms.pbc = True

        # Set up calculator
        tag = '_nosym' if symmetry == 'off' else ''
        atoms.calc = GPAW(
            xc=xc,
            mode=PW(pw,
                    # Interpolate the density in real-space
                    interpolation=3),
            kpts={'size': (kpts, kpts // 2, 1), 'gamma': True},
            mixer={'beta': 0.5},
            setups={'V': '5'},
            nbands=band_cutoff + 12,
            occupations=FermiDirac(occw),
            convergence=conv,
            symmetry=symmetry,
            txt=self.folder / f'v2br4_pw{tag}.txt')

        atoms.get_potential_energy()

        return atoms.calc

    @gpwfile
    def v2br4_pw(self):
        return self._v2br4()

    @gpwfile
    def v2br4_pw_nosym(self):
        return self._v2br4(symmetry='off')

    @with_band_cutoff(gpw='fe_pw',
                      band_cutoff=9)  # 4s, 4p, 3d = 9
    def _fe(self, *, band_cutoff, symmetry=None):
        if symmetry is None:
            symmetry = {}
        """See also the fe_fixture_test.py test."""
        xc = 'LDA'
        kpts = 4
        pw = 300
        occw = 0.01
        conv = {'bands': band_cutoff + 1,
                'density': 1.e-8}
        a = 2.867
        mm = 2.21
        atoms = bulk('Fe', 'bcc', a=a)
        # It is necessary to rattle the atoms to make sure that all tests pass
        # on all machines - see https://gitlab.com/gpaw/gpaw/-/issues/1397
        atoms.rattle(0.01, seed=42)
        atoms.set_initial_magnetic_moments([mm])
        atoms.center()
        tag = '_nosym' if symmetry == 'off' else ''

        atoms.calc = GPAW(
            xc=xc,
            mode=PW(pw),
            kpts={'size': (kpts, kpts, kpts)},
            nbands=band_cutoff + 9,
            occupations=FermiDirac(occw),
            convergence=conv,
            txt=self.folder / f'fe_pw{tag}.txt',
            symmetry=symmetry)

        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def fe_pw(self):
        return self._fe()

    @gpwfile
    def fe_pw_nosym(self):
        return self._fe(symmetry='off')

    @with_band_cutoff(gpw='co_pw',
                      band_cutoff=14)  # 2 * (4s + 3d)
    def _co(self, *, band_cutoff, symmetry=None):
        if symmetry is None:
            symmetry = {}
        # ---------- Inputs ---------- #

        # Atomic configuration
        a = 2.5071
        c = 4.0695
        mm = 1.6
        atoms = bulk('Co', 'hcp', a=a, c=c)
        atoms.set_initial_magnetic_moments([mm, mm])
        atoms.center()

        # Ground state parameters
        xc = 'LDA'
        occw = 0.01
        ebands = 2 * 2  # extra bands for ground state calculation
        pw = 200
        conv = {'density': 1e-8,
                'forces': 1e-8,
                'bands': band_cutoff + 1}

        # ---------- Calculation ---------- #

        tag = '_nosym' if symmetry == 'off' else ''
        atoms.calc = GPAW(xc=xc,
                          mode=PW(pw),
                          kpts={'size': (4, 4, 4), 'gamma': True},
                          occupations=FermiDirac(occw),
                          convergence=conv,
                          nbands=band_cutoff + ebands,
                          symmetry=symmetry,
                          txt=self.folder / f'co_pw{tag}.txt')

        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def co_pw(self):
        return self._co()

    @gpwfile
    def co_pw_nosym(self):
        return self._co(symmetry='off')

    @with_band_cutoff(gpw='srvo3_pw',
                      band_cutoff=20)
    def _srvo3(self, *, band_cutoff, symmetry=None):
        if symmetry is None:
            symmetry = {}

        nk = 3
        cell = bulk('V', 'sc', a=3.901).cell
        atoms = Atoms('SrVO3', cell=cell, pbc=True,
                      scaled_positions=((0.5, 0.5, 0.5),
                                        (0, 0, 0),
                                        (0, 0.5, 0),
                                        (0, 0, 0.5),
                                        (0.5, 0, 0)))
        # Ground state parameters
        xc = 'LDA'
        occw = 0.01
        ebands = 10  # extra bands for ground state calculation
        pw = 200
        conv = {'density': 1e-8,
                'bands': band_cutoff + 1}

        # ---------- Calculation ---------- #

        tag = '_nosym' if symmetry == 'off' else ''
        atoms.calc = GPAW(xc=xc,
                          mode=PW(pw),
                          kpts={'size': (nk, nk, nk), 'gamma': True},
                          occupations=FermiDirac(occw),
                          convergence=conv,
                          nbands=band_cutoff + ebands,
                          symmetry=symmetry,
                          txt=self.folder / f'srvo3_pw{tag}.txt')

        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def srvo3_pw(self):
        return self._srvo3()

    @gpwfile
    def srvo3_pw_nosym(self):
        return self._srvo3(symmetry='off')

    @with_band_cutoff(gpw='al_pw',
                      band_cutoff=10)  # 3s, 3p, 4s, 3d
    def _al(self, *, band_cutoff, symmetry=None):
        if symmetry is None:
            symmetry = {}
        xc = 'LDA'
        kpts = 4
        pw = 300
        occw = 0.01
        conv = {'bands': band_cutoff + 1,
                'density': 1.e-8}
        a = 4.043
        atoms = bulk('Al', 'fcc', a=a)
        atoms.center()
        tag = '_nosym' if symmetry == 'off' else ''

        atoms.calc = GPAW(
            xc=xc,
            mode=PW(pw),
            kpts={'size': (kpts, kpts, kpts), 'gamma': True},
            nbands=band_cutoff + 4,  # + 4p, 5s
            occupations=FermiDirac(occw),
            convergence=conv,
            txt=self.folder / f'al_pw{tag}.txt',
            symmetry=symmetry)

        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def al_pw(self):
        return self._al()

    @gpwfile
    def al_pw_nosym(self):
        return self._al(symmetry='off')

    @gpwfile
    def bse_al(self):
        a = 4.043
        atoms = bulk('Al', 'fcc', a=a)
        calc = GPAW(mode='pw',
                    txt=self.folder / 'bse_al.txt',
                    kpts={'size': (4, 4, 4), 'gamma': True},
                    xc='LDA',
                    nbands=4,
                    convergence={'bands': 'all'})

        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def ag_plusU_pw(self):
        xc = 'LDA'
        kpts = 2
        nbands = 6
        pw = 300
        occw = 0.01
        conv = {'bands': nbands,
                'density': 1e-12}
        a = 4.07
        atoms = bulk('Ag', 'fcc', a=a)
        atoms.center()

        atoms.calc = GPAW(
            xc=xc,
            mode=PW(pw),
            kpts={'size': (kpts, kpts, kpts), 'gamma': True},
            setups={'Ag': '11:d,2.0,0'},
            nbands=nbands,
            occupations=FermiDirac(occw),
            convergence=conv,
            parallel={'domain': 1},
            txt=self.folder / 'ag_plusU_pw.txt')

        atoms.get_potential_energy()

        atoms.calc.diagonalize_full_hamiltonian()

        return atoms.calc

    @gpwfile
    def gaas_pw_nosym(self):
        return self._gaas(symmetry='off')

    @gpwfile
    def gaas_pw(self):
        return self._gaas()

    @with_band_cutoff(gpw='gaas_pw',
                      band_cutoff=8)
    def _gaas(self, *, band_cutoff, symmetry=None):
        if symmetry is None:
            symmetry = {}
        nk = 4
        cell = bulk('Ga', 'fcc', a=5.68).cell
        atoms = Atoms('GaAs', cell=cell, pbc=True,
                      scaled_positions=((0, 0, 0), (0.25, 0.25, 0.25)))
        tag = '_nosym' if symmetry == 'off' else ''
        conv = {'bands': band_cutoff + 1,
                'density': 1.e-8}

        calc = GPAW(mode=PW(400),
                    xc='LDA',
                    occupations=FermiDirac(width=0.01),
                    convergence=conv,
                    nbands=band_cutoff + 1,
                    kpts={'size': (nk, nk, nk), 'gamma': True},
                    txt=self.folder / f'gs_GaAs{tag}.txt',
                    symmetry=symmetry)

        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms.calc

    @gpwfile
    def h_pw280_fulldiag(self):
        return self._pw_280_fulldiag(Atoms('H'), hund=True, nbands=4)

    @gpwfile
    def h2_pw280_fulldiag(self):
        return self._pw_280_fulldiag(
            Atoms('H2', [(0, 0, 0), (0, 0, 0.7413)]), nbands=8)

    def _pw_280_fulldiag(self, atoms, **kwargs):
        atoms.set_pbc(True)
        atoms.set_cell((2., 2., 3.))
        atoms.center()
        calc = GPAW(mode=PW(280, force_complex_dtype=True),
                    txt=self.folder / f'{atoms.symbols}_pw_280_fulldiag.txt',
                    xc='LDA',
                    basis='dzp',
                    parallel={'domain': 1},
                    convergence={'density': 1.e-6},
                    **kwargs)
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.diagonalize_full_hamiltonian(nbands=80)
        return calc

    @gpwfile
    def fe_pw_distorted(self):
        xc = 'revTPSS'
        m = [2.9]
        fe = bulk('Fe')
        fe.set_initial_magnetic_moments(m)
        k = 3
        fe.calc = GPAW(mode=PW(800),
                       h=0.15,
                       occupations=FermiDirac(width=0.03),
                       xc=xc,
                       kpts=(k, k, k),
                       convergence={'energy': 1e-8},
                       parallel={'domain': 1, 'augment_grids': True},
                       txt=self.folder / 'fe_pw_distorted.txt')
        fe.set_cell(np.dot(fe.cell,
                           [[1.02, 0, 0.03],
                            [0, 0.99, -0.02],
                            [0.2, -0.01, 1.03]]),
                    scale_atoms=True)
        fe.get_potential_energy()
        return fe.calc

    @gpwfile
    def si8_fd(self):
        a = 5.404
        bulk = Atoms(symbols='Si8',
                     scaled_positions=[(0, 0, 0),
                                       (0, 0.5, 0.5),
                                       (0.5, 0, 0.5),
                                       (0.5, 0.5, 0),
                                       (0.25, 0.25, 0.25),
                                       (0.25, 0.75, 0.75),
                                       (0.75, 0.25, 0.75),
                                       (0.75, 0.75, 0.25)],
                     pbc=True, cell=(a, a, a))
        n = 20
        calc = GPAW(mode='fd',
                    gpts=(n, n, n),
                    nbands=8 * 3,
                    occupations=FermiDirac(width=0.01),
                    kpts=(1, 1, 1))
        bulk.calc = calc
        bulk.get_potential_energy()

        return bulk.calc

    @gpwfile
    def si_pw_distorted(self):
        xc = 'TPSS'
        si = bulk('Si')
        k = 3
        si.calc = GPAW(mode=PW(250),
                       mixer=Mixer(0.7, 5, 50.0),
                       xc=xc,
                       occupations=FermiDirac(0.01),
                       kpts=(k, k, k),
                       convergence={'energy': 1e-8},
                       parallel={'domain': min(2, world.size)},
                       txt=self.folder / 'si_pw_distorted.txt')
        si.set_cell(np.dot(si.cell,
                           [[1.02, 0, 0.03],
                            [0, 0.99, -0.02],
                            [0.2, -0.01, 1.03]]),
                    scale_atoms=True)
        si.get_potential_energy()
        return si.calc

    @gpwfile
    def IBiTe_pw_monolayer(self):
        # janus material. material parameters obtained from c2db.
        from ase.atoms import Atoms
        IBiTe_positions = np.array([[0, 2.552, 7.802],
                                    [0, 0, 9.872],
                                    [2.210, 1.276, 11.575]])
        IBiTe = Atoms('IBiTe', positions=IBiTe_positions)
        IBiTe.pbc = [True, True, False]
        cell = np.array([[4.4219, 0, 0.0, ],
                         [-2.211, 3.829, 0.0],
                         [0.0, 0.0, 19.5]])
        IBiTe.cell = cell
        calc = GPAW(mode=PW(200),
                    xc='LDA',
                    occupations=FermiDirac(0.01),
                    kpts={'size': (6, 6, 1), 'gamma': True},
                    txt=None)
        IBiTe.calc = calc
        IBiTe.get_potential_energy()
        return IBiTe.calc

    def _intraband(self, spinpol: bool):
        atoms = bulk('Na')
        if spinpol:
            atoms.set_initial_magnetic_moments([[0.1]])
        atoms.calc = GPAW(mode=PW(300),
                          kpts={'size': (8, 8, 8), 'gamma': True},
                          parallel={'band': 1},
                          txt=None)
        atoms.get_potential_energy()
        atoms.calc.diagonalize_full_hamiltonian(nbands=20)
        return atoms.calc

    @gpwfile
    def intraband_spinpaired_fulldiag(self):
        return self._intraband(False)

    @gpwfile
    def intraband_spinpolarized_fulldiag(self):
        return self._intraband(True)


# We add Si fixtures with various symmetries to the GPWFiles namespace
for name, method in si_gpwfiles().items():
    setattr(GPWFiles, name, method)


if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    calc = getattr(GPWFiles(Path()), name)()
    calc.write(name + '.gpw', mode='all')
