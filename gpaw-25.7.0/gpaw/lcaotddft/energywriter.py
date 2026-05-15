import numpy as np

from ase.utils import IOContext

from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.utilities.scalapack import scalapack_zero


class EnergyWriter(TDDFTObserver):
    version = 1

    def __init__(self, paw, dmat, filename, interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.ioctx = IOContext()
        self.dmat = dmat
        if paw.niter == 0:
            # Initialize
            self.fd = self.ioctx.openfile(filename, comm=paw.world, mode='w')
        else:
            # Read and continue
            self.fd = self.ioctx.openfile(filename, comm=paw.world, mode='a')

    def _write(self, line):
        self.fd.write(line)
        self.fd.flush()

    def _write_header(self, paw):
        if paw.niter != 0:
            return
        line = f'# {self.__class__.__name__}[version={self.version}]\n'
        line += ('# %15s %22s %22s %22s %22s %22s %22s\n' %
                 ('time', 'kinetic0', 'coulomb', 'zero', 'external',
                  'xc', 'band'))
        self._write(line)

    def _write_kick(self, paw):
        time = paw.time
        kick = paw.kick_strength
        line = '# Kick = [%22.12le, %22.12le, %22.12le]; ' % tuple(kick)
        line += 'Time = %.8lf\n' % time
        self._write(line)

    def _get_energies(self, paw):
        e_band = 0.0
        rho_uMM = self.dmat.get_density_matrix((paw.niter, paw.action))
        get_H_MM = paw.td_hamiltonian.get_hamiltonian_matrix
        ksl = paw.wfs.ksl
        for u, kpt in enumerate(paw.wfs.kpt_u):
            rho_MM = rho_uMM[u]

            # H_MM = get_H_MM(kpt, paw.time)
            H_MM = get_H_MM(kpt, paw.time, addfxc=False, addpot=False)

            if ksl.using_blacs:
                # rhoH_MM = (rho_MM * H_MM).real  # General case
                rhoH_MM = rho_MM.real * H_MM.real  # Hamiltonian is real
                # Hamiltonian has correct values only in lower half, so
                # 1. Add lower half and diagonal twice
                scalapack_zero(ksl.mmdescriptor, rhoH_MM, 'U')
                e = 2 * np.sum(rhoH_MM)
                # 2. Reduce the extra diagonal
                scalapack_zero(ksl.mmdescriptor, rhoH_MM, 'L')
                e -= np.sum(rhoH_MM)
                # Sum over all ranks
                e = ksl.block_comm.sum(e)
            else:
                e = np.sum(rho_MM * H_MM).real

            e_band += e

        paw.wfs.occupations.e_band = e_band
        paw.wfs.occupations.e_entropy = 0.0
        e_kinetic0 = paw.hamiltonian.e_kinetic0
        e_coulomb = paw.hamiltonian.e_coulomb
        e_zero = paw.hamiltonian.e_zero
        e_external = paw.hamiltonian.e_external
        e_xc = paw.hamiltonian.e_xc

        return np.array((e_kinetic0, e_coulomb, e_zero,
                         e_external, e_xc, e_band))

    def _write_energy(self, paw):
        time = paw.time
        energy_i = self._get_energies(paw) - self.energy0_i
        line = (
            '%20.8lf %22.12le %22.12le %22.12le %22.12le %22.12le %22.12le\n' %
            ((time, ) + tuple(energy_i)))
        self._write(line)

    def _update(self, paw):
        if paw.action == 'init':
            self._write_header(paw)
            self.energy0_i = self._get_energies(paw)
        elif paw.action == 'kick':
            self._write_kick(paw)
        self._write_energy(paw)

    def __del__(self):
        self.ioctx.close()
