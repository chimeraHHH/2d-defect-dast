import pytest
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpoint import KPoint
from gpaw.symmetry import Symmetry
from gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.lfc import PWLFC
from gpaw.projections import Projections
from gpaw.mpi import world
from gpaw.spline import Spline
from gpaw.hybrids.energy import calculate_energy
from gpaw.hybrids.forces import calculate_forces
from gpaw.hybrids.coulomb import coulomb_interaction
from gpaw.hybrids.paw import calculate_paw_stuff
from gpaw.hybrids.symmetry import Symmetry as Sym
from gpaw.hybrids.kpts import get_kpt


N = 20
L = 2.5
nb = 2
r2 = np.linspace(0, 2, 101)**2


@pytest.mark.ci
@pytest.mark.skipif(world.size > 1, reason='Not parallelized')
def test_force():
    x = 0.2
    y = 0.35
    z = 0.1
    calc = Calc([[x, y, z]])
    omega = 0.2
    wfs = calc.wfs
    kd = wfs.kd
    coulomb = coulomb_interaction(omega, wfs.gd, kd)
    sym = Sym(kd)
    paw_s = calculate_paw_stuff(wfs, calc.density)
    F_v = calculate_forces(wfs, coulomb, sym, paw_s)[0] * Ha / Bohr
    print(F_v)
    dx = 0.0001
    y -= dx / 2
    calc = Calc([[x, y, z]])
    em = energy(calc, sym, coulomb)
    y += dx
    calc = Calc([[x, y, z]])
    ep = energy(calc, sym, coulomb)
    error = (em - ep) / (dx * L * Bohr) - F_v[1]
    print(error)
    assert abs(error) < 6e-10


class Calc:
    def __init__(self, spos_ac):
        self.spos_ac = np.array(spos_ac)
        self.wfs = WFS(self.spos_ac)
        self.density = Dens(self.wfs)
        self.hamiltonian = Ham()


class WFS:
    nspins = 1
    world = world

    def __init__(self, spos_ac):
        self.spos_ac = spos_ac
        self.gd = GridDescriptor([N, N, N], np.eye(3) * L)
        sym = Symmetry([], self.gd.cell_cv)
        self.kd = KPointDescriptor(None)
        self.kd.set_symmetry(Atoms(pbc=True), sym)
        self.setups = [Setup()]
        pd = PWDescriptor(50, self.gd, complex, self.kd)
        data = pd.zeros(nb)
        R = self.gd.get_grid_point_coordinates()
        R -= L / 2
        R2 = (R**2).sum(0)
        data[0] = pd.fft(np.exp(-R2 * (10 / L**2)))
        data[1] = pd.fft(np.exp(-R2 * (10 / L**2))) + 0.5
        psit = PlaneWaveExpansionWaveFunctions(nb, pd, data=data)
        proj = Projections(nb, [1], AP(), world, dtype=complex)
        self.pt = PWLFC([[Spline.from_data(0, 2.0, np.exp(-r2 * 10))]], pd)
        self.pt.set_positions(self.spos_ac)
        psit.matrix_elements(self.pt, out=proj)
        f_n = np.array([1.0, 0.5])
        kpt = KPoint(1.0, 1.0, 0, 0, 0, None)
        kpt.psit = psit
        kpt.projections = proj
        kpt.f_n = f_n
        self.kpt_u = [kpt]
        self.kpt_qs = [[kpt]]


class Ham:
    e_total_free = -100.0
    e_xc = -20.0


class Dens:
    nspins = 1

    def __init__(self, wfs):
        kpt = wfs.kpt_u[0]
        self.D_asp = D(kpt)
        self.finegd = wfs.gd.refine()
        self.nt_sg = self.finegd.zeros(1)


class D(dict):
    def __init__(self, kpt):
        dict.__init__(self)
        P_ni = kpt.P_ani[0]
        self[0] = kpt.f_n.dot(P_ni * P_ni.conj()).real[np.newaxis]
        self.partition = AP()


class AP:
    my_indices = [0]
    comm = world
    rank_a = [0]


class Setup:
    Delta_iiL = np.zeros((1, 1, 1)) + 0.1
    X_p = np.zeros(1) + 0.3
    ExxC = -10.0
    ghat_l = [Spline.from_data(0, 2.0, np.exp(-r2 * 10))]
    xc_correction = None
    M_pp = np.zeros((1, 1)) + 0.3


def energy(calc, sym, coulomb):
    wfs = calc.wfs
    paw_s = calculate_paw_stuff(wfs, calc.density)
    kpts = [get_kpt(wfs, 0, 0, 0, nb)]
    e1, e2 = calculate_energy(kpts, paw_s[0], wfs, sym, coulomb, wfs.spos_ac)
    evc = e1 * 2 * Ha
    evv = e2 * 2 * Ha
    return evc + evv
