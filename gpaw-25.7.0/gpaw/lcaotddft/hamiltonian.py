from gpaw.mixer import DummyMixer
from gpaw.xc import XC

from gpaw.lcaotddft.laser import create_laser
from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import read_wuMM
from gpaw.lcaotddft.utilities import write_uMM
from gpaw.lcaotddft.utilities import write_wuMM


class TimeDependentPotential:
    def __init__(self):
        self.ext_i = []
        self.laser_i = []
        self.initialized = False

    def add(self, ext, laser):
        self.ext_i.append(ext)
        self.laser_i.append(laser)

    def initialize(self, paw):
        if self.initialized:
            return

        self.ksl = paw.wfs.ksl
        self.kd = paw.wfs.kd
        self.kpt_u = paw.wfs.kpt_u
        get_matrix = paw.wfs.eigensolver.calculate_hamiltonian_matrix
        self.V_iuMM = []
        for ext in self.ext_i:
            V_uMM = []
            hamiltonian = KickHamiltonian(paw.hamiltonian, paw.density, ext)
            for kpt in paw.wfs.kpt_u:
                V_MM = get_matrix(hamiltonian, paw.wfs, kpt,
                                  add_kinetic=False, root=-1)
                V_uMM.append(V_MM)
            self.V_iuMM.append(V_uMM)
        self.Ni = len(self.ext_i)
        self.initialized = True

    def get_MM(self, u, time):
        V_MM = self.laser_i[0].strength(time) * self.V_iuMM[0][u]
        for i in range(1, self.Ni):
            V_MM += self.laser_i[i].strength(time) * self.V_iuMM[i][u]
        return V_MM

    def write(self, writer):
        writer.write(Ni=self.Ni)
        writer.write(laser_i=[laser.todict() for laser in self.laser_i])
        write_wuMM(self.kd, self.ksl, writer, 'V_iuMM',
                   self.V_iuMM, range(self.Ni))

    def read(self, reader):
        self.Ni = reader.Ni
        self.laser_i = [create_laser(**laser) for laser in reader.laser_i]
        self.V_iuMM = read_wuMM(self.kpt_u, self.ksl, reader, 'V_iuMM',
                                range(self.Ni))
        self.initialized = True


class KickHamiltonian:
    def __init__(self, ham, dens, ext):
        nspins = ham.nspins
        vext_g = ext.get_potential(ham.finegd)
        vext_G = ham.restrict_and_collect(vext_g)
        self.vt_sG = ham.gd.empty(nspins)
        for s in range(nspins):
            self.vt_sG[s] = vext_G

        dH_asp = ham.setups.empty_atomic_matrix(nspins,
                                                ham.atom_partition)
        W_aL = dens.ghat.dict()
        dens.ghat.integrate(vext_g, W_aL)
        # XXX this is a quick hack to get the distribution right
        # It'd be better to have these distributions abstracted elsewhere.
        # The idea is (thanks Ask, see discussion in !910):
        # * gd has D cores and coarse grid
        # * aux_gd is the same grid as gd but with either D or
        #   K * B * D cores (with augment_grids = True)
        # * finegd is then aux_gd.refine().
        # In integrals like W_aL, the atomic quantities are distributed
        # according to the domains of those atoms (so finegd for W_aL,
        # but gd for dH_asp).
        # And, some things (atomic XC corrections) are calculated evenly
        # distributed among all cores.
        dHaux_asp = ham.atomdist.to_aux(dH_asp)
        for a, W_L in W_aL.items():
            setup = dens.setups[a]
            dH_p = setup.Delta_pL @ W_L
            for s in range(nspins):
                dHaux_asp[a][s] = dH_p
        self.dH_asp = ham.atomdist.from_aux(dHaux_asp)


class TimeDependentHamiltonian:
    def __init__(self, fxc=None, td_potential=None, scale=None,
                 rremission=None):
        assert fxc is None or isinstance(fxc, str)
        self.fxc_name = fxc
        if isinstance(td_potential, dict):
            td_potential = [td_potential]
        if isinstance(td_potential, list):
            pot_i = td_potential
            td_potential = TimeDependentPotential()
            for pot in pot_i:
                td_potential.add(**pot)
        self.td_potential = td_potential
        self.has_scale = scale is not None
        if self.has_scale:
            self.scale = scale
        self.rremission = rremission

    def write(self, writer):
        if self.has_fxc:
            self.write_fxc(writer.child('fxc'))
        if self.has_scale:
            self.write_scale(writer.child('scale'))
        if self.td_potential is not None:
            self.td_potential.write(writer.child('td_potential'))
        if self.rremission is not None:
            self.rremission.write(writer.child('rremission'))

    def write_fxc(self, writer):
        writer.write(name=self.fxc_name)
        write_uMM(self.wfs.kd, self.wfs.ksl, writer, 'deltaXC_H_uMM',
                  self.deltaXC_H_uMM)

    def write_scale(self, writer):
        writer.write(scale=self.scale)
        write_uMM(self.wfs.kd, self.wfs.ksl, writer, 'scale_H_uMM',
                  self.scale_H_uMM)

    def read(self, reader):
        if 'fxc' in reader:
            self.read_fxc(reader.fxc)
        if 'scale' in reader:
            self.read_scale(reader.scale)
        if 'td_potential' in reader:
            assert self.td_potential is None
            self.td_potential = TimeDependentPotential()
            self.td_potential.ksl = self.wfs.ksl
            self.td_potential.kd = self.wfs.kd
            self.td_potential.kpt_u = self.wfs.kpt_u
            self.td_potential.read(reader.td_potential)
        if 'rremission' in reader:
            assert self.rremission is None
            from gpaw.lcaotddft.qed import RRemission
            self.rremission = RRemission.from_reader(reader.rremission)

    def read_fxc(self, reader):
        assert self.fxc_name is None or self.fxc_name == reader.name
        self.fxc_name = reader.name
        self.deltaXC_H_uMM = read_uMM(self.wfs.kpt_u, self.wfs.ksl, reader,
                                      'deltaXC_H_uMM')

    def read_scale(self, reader):
        assert not self.has_scale or self.scale == reader.scale
        self.has_scale = True
        self.scale = reader.scale
        self.scale_H_uMM = read_uMM(self.wfs.kpt_u, self.wfs.ksl, reader,
                                    'scale_H_uMM')

    def initialize(self, paw):
        self.timer = paw.timer
        self.timer.start('Initialize TDDFT Hamiltonian')
        self.wfs = paw.wfs
        self.density = paw.density
        self.hamiltonian = paw.hamiltonian
        niter = paw.niter
        if self.rremission is not None:
            self.rremission.initialize(paw)

        # Reset the density mixer
        # XXX: density mixer is not written to the gpw file
        # XXX: so we need to set it always
        self.density.set_mixer(DummyMixer())
        self.update()

        # Initialize fxc
        self.initialize_fxc(niter)

        # Initialize scale
        self.initialize_scale(niter)

        # Initialize td_potential
        if self.td_potential is not None:
            self.td_potential.initialize(paw)

        self.timer.stop('Initialize TDDFT Hamiltonian')

    def initialize_fxc(self, niter):
        self.has_fxc = self.fxc_name is not None
        if not self.has_fxc:
            return
        self.timer.start('Initialize fxc')
        # XXX: Similar functionality is available in
        # paw.py: PAW.linearize_to_xc(self, newxc)
        # See test/lcaotddft/fxc_vs_linearize.py

        # Calculate deltaXC: 1. take current H_MM
        if niter == 0:
            def get_H_MM(kpt):
                return self.get_hamiltonian_matrix(kpt, time=0.0,
                                                   addfxc=False, addpot=False,
                                                   scale=False)
            self.deltaXC_H_uMM = []
            for kpt in self.wfs.kpt_u:
                self.deltaXC_H_uMM.append(get_H_MM(kpt))

        # Update hamiltonian.xc
        if self.fxc_name == 'RPA':
            xc_name = 'null'
        else:
            xc_name = self.fxc_name
        # XXX: xc is not written to the gpw file
        # XXX: so we need to set it always
        xc = XC(xc_name)
        xc.initialize(self.density, self.hamiltonian, self.wfs)
        xc.set_positions(self.hamiltonian.spos_ac)
        self.hamiltonian.xc = xc
        self.update()

        # Calculate deltaXC: 2. update with new H_MM
        if niter == 0:
            for u, kpt in enumerate(self.wfs.kpt_u):
                self.deltaXC_H_uMM[u] -= get_H_MM(kpt)
        self.timer.stop('Initialize fxc')

    def initialize_scale(self, niter):
        if not self.has_scale:
            return
        self.timer.start('Initialize scale')

        # Take current H_MM and multiply with scale
        if niter == 0:
            def get_H_MM(kpt):
                return self.get_hamiltonian_matrix(kpt, time=0.0,
                                                   addfxc=False, addpot=False,
                                                   scale=False)
            self.scale_H_uMM = []
            for kpt in self.wfs.kpt_u:
                self.scale_H_uMM.append((1 - self.scale) * get_H_MM(kpt))
        self.timer.stop('Initialize scale')

    def update_projectors(self):
        self.timer.start('Update projectors')
        for kpt in self.wfs.kpt_u:
            self.wfs.atomic_correction.calculate_projections(self.wfs, kpt)
        self.timer.stop('Update projectors')

    def get_hamiltonian_matrix(self, kpt, time, addfxc=True, addpot=True,
                               scale=True):
        self.timer.start('Calculate H_MM')
        kpt_rank, q = self.wfs.kd.get_rank_and_index(kpt.k)
        u = q * self.wfs.nspins + kpt.s
        assert kpt_rank == self.wfs.kd.comm.rank

        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        H_MM = get_matrix(self.hamiltonian, self.wfs, kpt, root=-1)
        if addfxc and self.has_fxc:
            H_MM += self.deltaXC_H_uMM[u]

        if self.rremission is not None:
            H_MM += self.rremission.vradiationreaction(kpt, time)

        if scale and self.has_scale:
            H_MM *= self.scale
            H_MM += self.scale_H_uMM[u]
        if addpot and self.td_potential is not None:
            H_MM += self.td_potential.get_MM(u, time)
        self.timer.stop('Calculate H_MM')
        if hasattr(kpt, 'A_MM'):
            return H_MM + kpt.A_MM
        else:
            return H_MM

    def update(self, mode='all'):
        self.timer.start('Update TDDFT Hamiltonian')
        if mode in ['all', 'density']:
            self.update_projectors()
            self.density.update(self.wfs)
        if mode in ['all']:
            self.hamiltonian.update(self.density)
        self.timer.stop('Update TDDFT Hamiltonian')
