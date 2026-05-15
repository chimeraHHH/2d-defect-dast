import numpy as np

from numpy.linalg import inv, solve

from ase.utils.timing import timer

from gpaw.lcaotddft.hamiltonian import KickHamiltonian
from gpaw import debug
from gpaw.tddft.units import au_to_as
from gpaw.utilities.scalapack import (pblas_simple_hemm, pblas_simple_gemm,
                                      scalapack_inverse, scalapack_solve,
                                      scalapack_tri2full)


def create_propagator(name, **kwargs):
    if name is None:
        return create_propagator('sicn')
    elif isinstance(name, Propagator):
        return name
    elif isinstance(name, dict):
        kwargs.update(name)
        return create_propagator(**kwargs)
    elif name == 'sicn':
        return SICNPropagator(**kwargs)
    elif name == 'scpc':
        return SelfConsistentPropagator(**kwargs)
    elif name == 'ecn':
        return ECNPropagator(**kwargs)
    elif name.endswith('.ulm'):
        return ReplayPropagator(name, **kwargs)
    else:
        raise ValueError('Unknown propagator: %s' % name)


def equal(a, b, eps=1e-8):
    return abs(a - b) < eps


class Propagator:

    def __init__(self):
        object.__init__(self)

    def initialize(self, paw):
        self.timer = paw.timer
        self.log = paw.log

    def kick(self, ext, time):
        raise NotImplementedError()

    def propagate(self, time, time_step):
        raise NotImplementedError()

    def control_paw(self, paw):
        raise NotImplementedError()

    def todict(self):
        raise NotImplementedError()

    def get_description(self):
        return '%s' % self.__class__.__name__


class LCAOPropagator(Propagator):

    def __init__(self):
        super().__init__()

    def initialize(self, paw):
        super().initialize(paw)
        self.wfs = paw.wfs
        self.density = paw.density
        self.hamiltonian = paw.td_hamiltonian


class ReplayPropagator(LCAOPropagator):

    def __init__(self, filename, update='all'):
        from gpaw.lcaotddft.wfwriter import WaveFunctionReader
        super().__init__()
        self.filename = filename
        self.update_mode = update
        self.reader = WaveFunctionReader(self.filename)
        self.read_index = 1
        self.read_count = len(self.reader)

    def _align_read_index(self, time):
        while self.read_index < self.read_count:
            r = self.reader[self.read_index]
            if equal(r.time, time):
                break
            self.read_index += 1
        if self.read_index == self.read_count:
            raise RuntimeError('Time not found: %f' % time)

    def _read(self):
        reader = self.reader[self.read_index]
        r = reader.wave_functions
        self.wfs.read_wave_functions(r)
        self.wfs.read_occupations(r)
        self.read_index += 1

    def kick(self, ext, time):
        self._align_read_index(time)
        # Check that this is the step after kick
        assert not equal(self.reader[self.read_index].time,
                         self.reader[self.read_index + 1].time)
        self._read()
        self.hamiltonian.update(self.update_mode)

    def propagate(self, time, time_step):
        next_time = time + time_step
        self._align_read_index(next_time)
        self._read()
        self.hamiltonian.update(self.update_mode)
        return next_time

    def control_paw(self, paw):
        # Read the initial state
        index = 1
        r = self.reader[index]
        assert r.action == 'init'
        assert equal(r.time, paw.time)
        self.read_index = index
        self._read()
        index += 1
        # Read the rest
        while index < self.read_count:
            r = self.reader[index]
            if r.action == 'init':
                index += 1
            elif r.action == 'kick':
                assert equal(r.time, paw.time)
                paw.absorption_kick(r.kick_strength)
                assert equal(r.time, paw.time)
                index += 1
            elif r.action == 'propagate':
                # Skip earlier times
                if r.time < paw.time or equal(r.time, paw.time):
                    index += 1
                    continue
                # Count the number of steps with the same time step
                time = paw.time
                time_step = r.time - time
                iterations = 0
                while index < self.read_count:
                    r = self.reader[index]
                    if (r.action != 'propagate' or
                        not equal(r.time - time, time_step)):
                        break
                    iterations += 1
                    time = r.time
                    index += 1
                # Propagate
                paw.propagate(time_step * au_to_as, iterations)
                assert equal(time, paw.time)
            else:
                raise RuntimeError('Unknown action: %s' % r.action)

    def __del__(self):
        self.reader.close()

    def todict(self):
        return {'name': self.filename,
                'update': self.update_mode}

    def get_description(self):
        lines = [self.__class__.__name__]
        lines += ['    File: %s' % (self.filename)]
        lines += ['    Update: %s' % (self.update_mode)]
        return '\n'.join(lines)


class ECNPropagator(LCAOPropagator):

    def __init__(self):
        super().__init__()
        self.have_velocity_operator_matrix = False

    def initialize(self, paw, hamiltonian=None):
        super().initialize(paw)
        if hamiltonian is not None:
            self.hamiltonian = hamiltonian

        ksl = self.wfs.ksl
        using_blacs = ksl.using_blacs
        if using_blacs:
            from gpaw.blacs import Redistributor
            self.log('BLACS Parallelization')

            # Propagator function
            self.propagate_wfs = self.propagate_wfs_blacs

            # Parallel grid descriptors
            grid = ksl.blockgrid
            assert grid.nprow * grid.npcol == ksl.block_comm.size
            self.mm_block_descriptor = ksl.mmdescriptor
            self.Cnm_block_descriptor = grid.new_descriptor(ksl.bd.nbands,
                                                            ksl.nao,
                                                            ksl.blocksize,
                                                            ksl.blocksize)
            self.CnM_unique_descriptor = ksl.nM_unique_descriptor

            # Redistributors
            self.Cnm2nM = Redistributor(ksl.block_comm,
                                        self.Cnm_block_descriptor,
                                        self.CnM_unique_descriptor)
            self.CnM2nm = Redistributor(ksl.block_comm,
                                        self.CnM_unique_descriptor,
                                        self.Cnm_block_descriptor)

            for kpt in self.wfs.kpt_u:
                scalapack_tri2full(self.mm_block_descriptor, kpt.S_MM)
                scalapack_tri2full(self.mm_block_descriptor, kpt.T_MM)

            if self.density.gd.comm.rank != 0:
                # This is a (0, 0) dummy array that is needed for
                # redistributing between nM and nm block descriptor.
                # See propagate_wfs() and also
                # BlacsOrbitalLayouts.calculate_blocked_density_matrix()
                self.dummy_C_nM = \
                    self.CnM_unique_descriptor.zeros(dtype=complex)

        else:
            # Propagator function
            self.propagate_wfs = self.propagate_wfs_numpy

        if debug and using_blacs:
            nao = ksl.nao
            self.MM_descriptor = grid.new_descriptor(nao, nao, nao, nao)
            self.mm2MM = Redistributor(ksl.block_comm,
                                       self.mm_block_descriptor,
                                       self.MM_descriptor)
            self.MM2mm = Redistributor(ksl.block_comm,
                                       self.MM_descriptor,
                                       self.mm_block_descriptor)

    def calculate_velocity_operator_matrix(self):
        if getattr(self, 'have_velocity_operator_matrix', False):
            return
        ksl = self.wfs.ksl

        gcomm = self.wfs.gd.comm
        manytci = self.wfs.manytci
        Vkick_qvmM = manytci.O_qMM_T_qMM(gcomm,
                                         ksl.Mstart,
                                         ksl.Mstop,
                                         ignore_upper=ksl.using_blacs,
                                         derivative=True)[0] * (-1j)

        my_atoms = self.wfs.atom_partition.my_indices
        dnabla_vaii = {v: {a: -self.wfs.setups[a].nabla_iiv[:, :, v] * (-1j)
                       for a in my_atoms} for v in range(3)}
        for kpt in self.wfs.kpt_u:
            assert kpt.k == 0

        for v in range(3):
            self.wfs.atomic_correction.calculate(0, dnabla_vaii[v],
                                                 Vkick_qvmM[kpt.q][v],
                                                 ksl.Mstart, ksl.Mstop)

        if ksl.using_blacs:
            for Vkick_vmM in Vkick_qvmM:
                for Vkick_mM in Vkick_vmM:
                    scalapack_tri2full(ksl.mMdescriptor, Vkick_mM)

        q = 0
        if ksl.using_blacs:
            Vkick_vmm = self.wfs.ksl.distribute_overlap_matrix(
                Vkick_qvmM[kpt.q]
            )
        else:
            gcomm.sum(Vkick_qvmM[q])
            Vkick_vmm = Vkick_qvmM[q]

        for kpt in self.wfs.kpt_u:
            assert kpt.q == 0
            kpt.Vkick_vmm = Vkick_vmm

        self.have_velocity_operator_matrix = True

    def velocity_gauge_kick(self, magnitude, direction, time):
        self.calculate_velocity_operator_matrix()
        for kpt in self.wfs.kpt_u:
            kpt.A_MM = (
                -magnitude * np.einsum('v,vMN->MN', direction, kpt.Vkick_vmm)
            )

        # Update Hamiltonian (and density)
        self.hamiltonian.update()

    def kick(self, ext, time):
        # Propagate
        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        kick_hamiltonian = KickHamiltonian(self.hamiltonian.hamiltonian,
                                           self.density, ext)
        for kpt in self.wfs.kpt_u:
            Vkick_MM = get_matrix(kick_hamiltonian, self.wfs, kpt,
                                  add_kinetic=False, root=-1)
            for i in range(10):
                self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, Vkick_MM, 0.1)

        # Update Hamiltonian (and density)
        self.hamiltonian.update()

    def propagate(self, time, time_step):
        get_H_MM = self.hamiltonian.get_hamiltonian_matrix
        for kpt in self.wfs.kpt_u:
            H_MM = get_H_MM(kpt, time)
            self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, H_MM, time_step)
        self.hamiltonian.update()
        return time + time_step

    @timer('Linear solve')
    def propagate_wfs_blacs(self, source_C_nM, target_C_nM, S_mm, H_mm, dt):
        # XXX, Preallocate?
        target_C_nm = self.Cnm_block_descriptor.empty(dtype=complex)
        source_C_nm = self.Cnm_block_descriptor.empty(dtype=complex)

        # C_nM is duplicated over all ranks in gd.comm.
        # Master rank will provide the actual data and other
        # ranks use a dummy array in redistribute().
        if self.density.gd.comm.rank != 0:
            source = self.dummy_C_nM
        else:
            source = source_C_nM
        self.CnM2nm.redistribute(source, source_C_nm)

        # Note: tri2full for S_mm and T_mm is done already in initialize().
        # H_mm seems to be a full matrix as we are working with complex
        # dtype, so no need to do tri2full here again XXX
        # scalapack_tri2full(self.mm_block_descriptor, H_mm)
        SjH_mm = S_mm + (0.5j * dt) * H_mm

        # 1. target = (S - 0.5j*H*dt) * source
        pblas_simple_gemm(self.Cnm_block_descriptor,
                          self.mm_block_descriptor,
                          self.Cnm_block_descriptor,
                          source_C_nm,
                          SjH_mm,
                          target_C_nm,
                          transb='C')

        # 2. target = (S + 0.5j*H*dt)^-1 * target
        scalapack_solve(self.mm_block_descriptor,
                        self.Cnm_block_descriptor,
                        SjH_mm,
                        target_C_nm)

        # C_nM is duplicated over all ranks in gd.comm.
        # Master rank will receive the data and other
        # ranks use a dummy array in redistribute()
        if self.density.gd.comm.rank != 0:
            target = self.dummy_C_nM
        else:
            target = target_C_nM
        self.Cnm2nM.redistribute(target_C_nm, target)

        # Broadcast the new C_nM to all ranks in gd.comm
        self.density.gd.comm.broadcast(target_C_nM, 0)

    @timer('Linear solve')
    def propagate_wfs_numpy(self, source_C_nM, target_C_nM, S_MM, H_MM, dt):
        SjH_MM = S_MM + (0.5j * dt) * H_MM
        target_C_nM[:] = np.dot(source_C_nM, SjH_MM.conj().T)
        target_C_nM[:] = solve(SjH_MM.T, target_C_nM.T).T

    def blacs_mm_to_global(self, H_mm):
        if not debug:
            raise RuntimeError('Use debug mode')
        # Someone could verify that this works and remove the error.
        raise NotImplementedError('Method untested and thus unreliable')
        target = self.MM_descriptor.empty(dtype=complex)
        self.mm2MM.redistribute(H_mm, target)
        self.wfs.world.barrier()
        return target

    def blacs_nm_to_global(self, C_nm):
        # Someone could verify that this works and remove the error.
        raise NotImplementedError('Method untested and thus unreliable')
        target = self.CnM_unique_descriptor.empty(dtype=complex)
        self.Cnm2nM.redistribute(C_nm, target)
        self.wfs.world.barrier()
        return target

    def todict(self):
        return {'name': 'ecn'}


class SICNPropagator(ECNPropagator):

    def __init__(self):
        super().__init__()

    def initialize(self, paw):
        super().initialize(paw)
        # Allocate kpt.C2_nM arrays
        for kpt in self.wfs.kpt_u:
            kpt.C2_nM = np.empty_like(kpt.C_nM)

    def propagate(self, time, time_step):
        get_H_MM = self.hamiltonian.get_hamiltonian_matrix
        # --------------
        # Predictor step
        # --------------
        # 1. Store current C_nM
        self.save_wfs()  # kpt.C2_nM = kpt.C_nM
        for kpt in self.wfs.kpt_u:
            # H_MM(t) = <M|H(t)|M>
            kpt.H0_MM = get_H_MM(kpt, time)
            # 2. Solve Psi(t+dt) from
            #    (S_MM - 0.5j*H_MM(t)*dt) Psi(t+dt)
            #       = (S_MM + 0.5j*H_MM(t)*dt) Psi(t)
            self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM,
                               time_step)
        # ---------------
        # Propagator step
        # ---------------
        # 1. Calculate H(t+dt)
        self.hamiltonian.update()
        for kpt in self.wfs.kpt_u:
            # 2. Estimate H(t+0.5*dt) ~ 0.5 * [ H(t) + H(t+dt) ]
            kpt.H0_MM += get_H_MM(kpt, time + time_step)
            kpt.H0_MM *= 0.5
            # 3. Solve Psi(t+dt) from
            #    (S_MM - 0.5j*H_MM(t+0.5*dt)*dt) Psi(t+dt)
            #       = (S_MM + 0.5j*H_MM(t+0.5*dt)*dt) Psi(t)
            self.propagate_wfs(kpt.C2_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM,
                               time_step)
            kpt.H0_MM = None
        # 4. Calculate new Hamiltonian (and density)
        self.hamiltonian.update()
        return time + time_step

    def save_wfs(self):
        for kpt in self.wfs.kpt_u:
            kpt.C2_nM[:] = kpt.C_nM

    def todict(self):
        return {'name': 'sicn'}


# ToDo: Should there be an abstract baseclass for SelfConsistentPropagator
# and SICNPropagator instead of inheriting from ECNPropagator?
class SelfConsistentPropagator(SICNPropagator):
    """
    This is an actual Predictor-Corrector propagator that uses the SICN
    and combines it with an actual Corrector step. This is identical to
    SICN for a very low tolerance of e.g. 1e-2. The higher the tolerance,
    the better energy etc will be preserved by the propagator. Notice,
    that the standard SICN accuracy is often sufficient, but some
    routines (like the 3rd time-derivative in the RRemission class)
    require higher accuracy for reliable predictions. The PC update
    become especially important for large time-steps. Try for instance
    dt=100 and propagte for a few thousand step, compare SICN vs SCPC.
    You will notice an artifical exponential decay of the SICN dipole
    after kick while SCPC will preserve the dipole oscillations.
    """
    def __init__(self, tolerance=1e-8, max_pc_iterations=20):
        super().__init__()
        self.tolerance = tolerance
        self.max_pc_iterations = max_pc_iterations
        self.last_pc_iterations = 0

    def propagate(self, time, time_step):
        """
        Since the propagate + update call will change the result
        for H0 at time t, we have to somehow safe the previous H0
        in order to estimate the intermediate H0 at t+dt/2.
        """
        prevH0 = []
        get_H_MM = self.hamiltonian.get_hamiltonian_matrix
        # --------------
        # Predictor step
        # --------------
        # 1. Store current C_nM
        self.save_wfs()  # kpt.C2_nM = kpt.C_nM
        for kpt in self.wfs.kpt_u:
            # H_MM(t) = <M|H(t)|M>
            kpt.H0_MM = get_H_MM(kpt, time)
            prevH0.append(kpt.H0_MM)
            # 2. Solve Psi(t+dt) from
            #    (S_MM - 0.5j*H_MM(t)*dt) Psi(t+dt)
            #       = (S_MM + 0.5j*H_MM(t)*dt) Psi(t)
            self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM,
                               time_step)
        self.hamiltonian.update()
        for last_pc_iterations in range(self.max_pc_iterations):
            self.last_pc_iterations = last_pc_iterations
            # ---------------
            # Propagator step
            # ---------------
            # 1. Calculate H(t+dt)
            itkpt = - 1
            for kpt in self.wfs.kpt_u:
                # 2. Estimate H(t+0.5*dt) ~ 0.5 * [ H(t) + H(t+dt) ]
                itkpt += 1
                kpt.H0_MM = prevH0[itkpt] + get_H_MM(kpt, time + time_step)
                kpt.H0_MM *= 0.5
                # 3. Solve Psi(t+dt) from
                #    (S_MM - 0.5j*H_MM(t+0.5*dt)*dt) Psi(t+dt)
                #       = (S_MM + 0.5j*H_MM(t+0.5*dt)*dt) Psi(t)
                self.propagate_wfs(kpt.C2_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM,
                                   time_step)
                kpt.H0_MM = None

            prev_dipole_v = self.density.calculate_dipole_moment()
            # 4. Calculate new Hamiltonian (and density)
            self.hamiltonian.update()
            dipole_v = self.density.calculate_dipole_moment()
            if np.sum(np.abs(dipole_v - prev_dipole_v)) < self.tolerance:
                break
        if last_pc_iterations == self.max_pc_iterations - 1:
            raise RuntimeError('The SCPC propagator required too ',
                               'many iterations to reach the ',
                               'demanded accuracy.')
        return time + time_step

    def todict(self):
        return {'name': 'scpc', 'tolerance': self.tolerance,
                'max_pc_iterations': self.max_pc_iterations}


class TaylorPropagator(Propagator):

    def __init__(self):
        super().__init__()
        raise NotImplementedError('TaylorPropagator not implemented')

    def initialize(self, paw):
        if 1:
            # XXX to propagator class
            if self.propagator == 'taylor' and self.blacs:
                # cholS_mm = self.mm_block_descriptor.empty(dtype=complex)
                for kpt in self.wfs.kpt_u:
                    kpt.invS_MM = kpt.S_MM.copy()
                    scalapack_inverse(self.mm_block_descriptor,
                                      kpt.invS_MM, 'L')
            if self.propagator == 'taylor' and not self.blacs:
                tmp = inv(self.wfs.kpt_u[0].S_MM)
                self.wfs.kpt_u[0].invS = tmp

    def taylor_propagator(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Taylor propagator')

        if self.blacs:
            # XXX, Preallocate
            target_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            if self.density.gd.comm.rank != 0:
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                sourceC_nM = self.CnM_unique_descriptor.zeros(dtype=complex)

            # Zeroth order taylor to target
            self.CnM2nm.redistribute(sourceC_nM, target_blockC_nm)

            # XXX, preallocate, optimize use of temporal arrays
            temp_blockC_nm = target_blockC_nm.copy()
            temp2_blockC_nm = target_blockC_nm.copy()

            order = 4
            assert self.wfs.kd.comm.size == 1
            for n in range(order):
                # Multiply with hamiltonian
                pblas_simple_hemm(self.mm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  H_MM,
                                  temp_blockC_nm,
                                  temp2_blockC_nm, side='R')
                # XXX: replace with not simple gemm
                temp2_blockC_nm *= -1j * dt / (n + 1)
                # Multiply with inverse overlap
                pblas_simple_hemm(self.mm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.wfs.kpt_u[0].invS_MM,  # XXX
                                  temp2_blockC_nm,
                                  temp_blockC_nm, side='R')
                target_blockC_nm += temp_blockC_nm
            if self.density.gd.comm.rank != 0:  # Todo: Change to gd.rank
                # XXX Fake blacks nbands, nao, nbands, nao grid because
                # some weird asserts
                # (these are 0,x or x,0 arrays)
                target = self.CnM_unique_descriptor.zeros(dtype=complex)
            else:
                target = targetC_nM
            self.Cnm2nM.redistribute(target_blockC_nm, target)

            self.density.gd.comm.broadcast(targetC_nM, 0)
        else:
            assert self.wfs.kd.comm.size == 1
            if self.density.gd.comm.rank == 0:
                targetC_nM[:] = sourceC_nM[:]
                tempC_nM = sourceC_nM.copy()
                order = 4
                for n in range(order):
                    tempC_nM[:] = \
                        np.dot(self.wfs.kpt_u[0].invS,
                               np.dot(H_MM, 1j * dt / (n + 1) *
                                      tempC_nM.T.conjugate())).T.conjugate()
                    targetC_nM += tempC_nM
            self.density.gd.comm.broadcast(targetC_nM, 0)

        self.timer.stop('Taylor propagator')
