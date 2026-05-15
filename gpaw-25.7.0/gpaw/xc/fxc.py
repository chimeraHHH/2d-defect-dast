from abc import ABC, abstractmethod
from time import time
from pathlib import Path

import ase.io.ulm as ulm
import numpy as np
from ase.units import Ha
from gpaw.response import timer
from scipy.special import p_roots, sici

from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.fd_operators import Gradient
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.response.qpd import SingleQPWDescriptor
from gpaw.utilities.blas import axpy, gemmdot
from gpaw.xc.rpa import RPACorrelation
from gpaw.heg import HEG
from gpaw.xc.fxc_kernels import (
    get_fHxc_Gr, get_pbe_fxc, get_fspinHxc_Gr_rALDA, get_fspinHxc_Gr_rAPBE)


def get_chi0v_spinsum(chi0_sGG, G_G):
    nG = chi0_sGG.shape[-1]
    chi0v = np.zeros((nG, nG), dtype=complex)
    for chi0_GG in chi0_sGG:
        chi0v += chi0_GG / G_G / G_G[:, np.newaxis]
    chi0v *= 4 * np.pi
    return chi0v


def get_chi0v_foreach_spin(chi0_sGG, G_G):
    ns, nG = chi0_sGG.shape[:2]

    chi0v_sGsG = np.zeros((ns, nG, ns, nG), dtype=complex)
    for s in range(ns):
        chi0v_sGsG[s, :, s, :] = chi0_sGG[s] / G_G / G_G[:, np.newaxis]
    chi0v_sGsG *= 4 * np.pi
    return chi0v_sGsG.reshape(ns * nG, ns * nG)


class FXCCache:
    def __init__(self, comm, tag, xc, ecut):
        self.comm = comm
        self.tag = tag
        self.xc = xc
        self.ecut = ecut

    @property
    def prefix(self):
        return f'{self.tag}_{self.xc}_{self.ecut}'

    def handle(self, iq):
        return Handle(self, iq)


class Handle:
    def __init__(self, cache, iq):
        self.cache = cache
        self.iq = iq
        self.comm = cache.comm

    @property
    def _path(self):
        return Path(f'fhxc_{self.cache.prefix}_{self.iq}.ulm')

    def exists(self):
        if self.comm.rank == 0:
            exists = int(self._path.exists())
        else:
            exists = 0
        exists = self.comm.sum_scalar(exists)
        return bool(exists)

    def read(self):
        from gpaw.mpi import broadcast
        if self.comm.rank == 0:
            array = self._read_master()
            assert array is not None
        else:
            array = None

        # The shape of the array is only known on rank0,
        # so we cannot use the in-place broadcast.  Therefore
        # we use the standalone function.
        array = broadcast(array, root=0, comm=self.comm)
        assert array is not None
        return array

    def write(self, array):
        if self.comm.rank == 0:
            assert array is not None
            self._write_master(array)
        self.comm.barrier()

    def _read_master(self):
        with ulm.open(self._path) as reader:
            return reader.array

    def _write_master(self, array):
        assert array is not None
        with ulm.open(self._path, 'w') as writer:
            writer.write(array=array)


class FXCCorrelation:
    def __init__(self,
                 calc,
                 xc='RPA',
                 nlambda=8,
                 frequencies=None,
                 weights=None,
                 density_cut=1.e-6,
                 unit_cells=None,
                 tag=None,
                 avg_scheme=None,
                 *,
                 ecut,
                 **kwargs):

        self.ecut = ecut
        if isinstance(ecut, (float, int)):
            self.ecut_max = ecut
        else:
            self.ecut_max = max(ecut)

        self.rpa = RPACorrelation(
            calc,
            xc=xc,
            nlambda=nlambda,
            frequencies=frequencies,
            weights=weights,
            calculate_q=self.calculate_q_fxc,
            ecut=self.ecut,
            **kwargs)

        self.gs = self.rpa.gs
        self.context = self.rpa.context

        self.l_l, self.weight_l = p_roots(nlambda)
        self.l_l = (self.l_l + 1.0) * 0.5
        self.weight_l *= 0.5
        self.xc = xc
        self.density_cut = density_cut
        if unit_cells is None:
            unit_cells = self.gs.kd.N_c
        self.unit_cells = unit_cells

        self.xcflags = XCFlags(self.xc)
        self.avg_scheme = self.xcflags.choose_avg_scheme(avg_scheme)

        if tag is None:
            tag = self.gs.atoms.get_chemical_formula(mode='hill')
            if self.avg_scheme is not None:
                tag += '_' + self.avg_scheme

        self.cache = FXCCache(self.context.comm, tag, self.xc, self.ecut_max)

    @property
    def blockcomm(self):
        # Cannot be aliased as attribute
        # because rpa gets blockcomm during calculate
        return self.rpa.wblocks.blockcomm

    def _calculate_kernel(self):
        # Find the first q vector to calculate kernel for
        # (density averaging scheme always calculates all q points anyway)

        q_empty = None

        for iq in reversed(range(len(self.rpa.integral.ibzq_qc))):
            handle = self.cache.handle(iq)

            if not handle.exists():
                q_empty = iq

        if q_empty is None:
            self.context.print('%s kernel already calculated\n' %
                               self.xc)
            return

        kernelkwargs = dict(
            gs=self.gs,
            xc=self.xc,
            ibzq_qc=self.rpa.integral.ibzq_qc,
            ecut=self.ecut_max,
            context=self.context)

        if self.avg_scheme == 'wavevector':
            self.context.print('Calculating %s kernel starting from '
                               'q point %s \n' % (self.xc, q_empty))
            kernelkwargs.update(q_empty=q_empty)
            kernel = KernelWave(**kernelkwargs)
        else:
            kernel = KernelDens(**kernelkwargs,
                                unit_cells=self.unit_cells,
                                density_cut=self.density_cut)

        for iq, fhxc_GG in kernel.calculate_fhxc():
            if self.context.comm.rank == 0:
                assert isinstance(fhxc_GG, np.ndarray), str(fhxc_GG)
            self.cache.handle(iq).write(fhxc_GG)

    @timer('FXC')
    def calculate(self, *, nbands=None):
        # kernel not required for RPA
        if self.xc != 'RPA':
            self._calculate_kernel()

        data = self.rpa.calculate_all_contributions(
            spin=self.gs.nspins > 1, nbands=nbands)
        return data.energy_i * Ha  # energies in eV

    @timer('Chi0(q)')
    def calculate_q_fxc(self, chi0_s, m1, m2, gcut):
        for s, chi0 in enumerate(chi0_s):
            self.rpa.chi0calc.update_chi0(chi0, m1=m1, m2=m2, spins=[s])

        qpd = chi0_s[0].qpd
        chi0_swGG = np.array([
            chi0.body.get_distributed_frequencies_array() for chi0 in chi0_s
        ])
        wblocks = chi0_s[0].body.get_distributed_frequencies_blocks1d()
        if wblocks.blockcomm.size > 1:  # why???
            chi0_swGG = np.swapaxes(chi0_swGG, 2, 3)

        # XXX Gamma-point code is NOT well tested!
        # Changed from qpd.kd.gamma to qpd.optical_limit cf. #1178.
        # This if/else was pasted from RPA where bug was also fixed.
        # We have not added regression test for fxc and the change
        # causes no test failures.
        if not chi0.qpd.optical_limit:
            energy_w = self.calculate_fxc_energies(qpd, chi0_swGG, gcut)
        else:
            chi0_swvv = [chi0.chi0_Wvv[wblocks.myslice] for chi0 in chi0_s]
            chi0_swxvG = [chi0.chi0_WxvG[wblocks.myslice] for chi0 in chi0_s]
            energy_w = self.calculate_optical_limit_fxc_energies(
                qpd, chi0_swGG, chi0_swvv, chi0_swxvG, gcut
            )
        return wblocks.all_gather(energy_w)

    def calculate_optical_limit_fxc_energies(
            self, qpd, chi0_swGG, chi0_swvv, chi0_swxvG, gcut):
        # For some reason, we "only" average out cartesian directions, instead
        # of performing an actual integral over the q-point volume as in rpa...
        energy_w = np.zeros(chi0_swGG.shape[1])
        for v in range(3):
            for chi0_wGG, chi0_wvv, chi0_wxvG in zip(
                    chi0_swGG, chi0_swvv, chi0_swxvG):
                chi0_wGG[:, 0] = chi0_wxvG[:, 0, v]
                chi0_wGG[:, :, 0] = chi0_wxvG[:, 1, v]
                chi0_wGG[:, 0, 0] = chi0_wvv[:, v, v]
            energy_w += self.calculate_fxc_energies(qpd, chi0_swGG, gcut) / 3
        return energy_w

    def calculate_energy_contribution(self, chi0v_sGsG, fv_sGsG, nG):
        """Calculate contribution to energy from a single frequency point.

        The RPA correlation energy is the integral over all frequencies
        from 0 to infinity of this expression."""

        e = 0.0
        assert len(chi0v_sGsG) % nG == 0
        ns = len(chi0v_sGsG) // nG

        for l, weight in zip(self.l_l, self.weight_l):
            chiv = np.linalg.solve(
                np.eye(nG * ns) - l * np.dot(chi0v_sGsG, fv_sGsG),
                chi0v_sGsG).real  # this is SO slow

            chiv = chiv.reshape(ns, nG, ns, nG)
            for s1 in range(ns):
                for s2 in range(ns):
                    e -= np.trace(chiv[s1, :, s2, :]) * weight

        e += np.trace(chi0v_sGsG.real)
        return e

    @timer('Energy')
    def calculate_fxc_energies(self, qpd, chi0_swGG, gcut):
        """Evaluate correlation energy from chi0 and the kernel fhxc"""
        ibzq_qc = self.rpa.integral.ibzq_qc
        ibzq2_q = [
            np.dot(ibzq_qc[i] - qpd.q_c,
                   ibzq_qc[i] - qpd.q_c)
            for i in range(len(ibzq_qc))
        ]

        qi = np.argsort(ibzq2_q)[0]

        G_G = gcut.cut(qpd.G2_qG[0]**0.5)  # |G+q|

        nG = len(G_G)
        ns = len(chi0_swGG)

        # There are three options to calculate the
        # energy depending on kernel and/or averaging scheme.

        # Option (1) - Spin-polarized form of kernel exists
        #              e.g. rALDA, rAPBE.
        #              Then, solve block diagonal form of Dyson
        #              equation (dimensions (ns*nG) * (ns*nG))
        #              (note this does not necessarily mean that
        #              the calculation is spin-polarized!)

        if self.xcflags.spin_kernel:
            fv_GG = gcut.spin_cut(self.cache.handle(qi).read(), ns)

            # the spin-polarized kernel constructed from wavevector average
            # is already multiplied by |q+G| |q+G'|/4pi, and doesn't require
            # special treatment of the head and wings.  However not true for
            # density average:

            if self.avg_scheme == 'density':
                # Create and modify a view:
                fv_sGsG = fv_GG.reshape(ns, nG, ns, nG)

                for s1 in range(ns):
                    for s2 in range(ns):
                        fv_sGsG[s1, :, s2, :] *= (
                            G_G * G_G[:, np.newaxis] / (4 * np.pi))

                        # XXX Gamma check changed cf. #1178 without
                        # further testing.
                        if np.prod(self.unit_cells) > 1 and qpd.optical_limit:
                            fv_sGsG[s1, 0, s2, :] = 0.0
                            fv_sGsG[s1, :, s2, 0] = 0.0
                            fv_sGsG[s1, 0, s2, 0] = 1.0

        else:
            fv_GG = np.eye(nG)

        if qpd.optical_limit:
            G_G[0] = 1.0

        energy_w = []
        for chi0_sGG in np.swapaxes(chi0_swGG, 0, 1):
            chi0_sGG = gcut.cut(chi0_sGG, [1, 2])

            if self.xcflags.spin_kernel:
                chi0v_sGsG = get_chi0v_foreach_spin(chi0_sGG, G_G)
            else:
                chi0v_sGsG = get_chi0v_spinsum(chi0_sGG, G_G)

            energy_w.append(self.calculate_energy_contribution(
                chi0v_sGsG, fv_GG, nG
            ))
        return np.array(energy_w)


class KernelIntegrator(ABC):
    def __init__(self, gs, xc, context, ibzq_qc, ecut):
        self.gs = gs
        self.xc = xc
        self.context = context

        self.xcflags = XCFlags(xc)
        self.gd = gs.density.gd
        self.ibzq_qc = ibzq_qc
        self.ecut = ecut

    def calculate_fhxc(self):
        self.context.print(
            f'Calculating {self.xc} kernel at {self.ecut} eV cutoff')
        fhxc_iterator = self._calculate_fhxc()

        while True:
            with self.context.timer('FHXC'):
                try:
                    yield next(fhxc_iterator)
                except StopIteration:
                    return

    @abstractmethod
    def _calculate_fhxc(self):
        """Perform computation and yield (iq, array) tuples."""


class KernelWave(KernelIntegrator):
    def __init__(self, *, q_empty, **kwargs):
        super().__init__(**kwargs)

        self.ns = self.gs.nspins
        self.q_empty = q_empty

        # Density grid
        n_sg, finegd = self.gs.get_all_electron_density(gridrefinement=2)
        self.n_g = n_sg.sum(axis=0).flatten()

        #  For atoms with large vacuum regions
        #  this apparently can take negative values!
        mindens = np.amin(self.n_g)

        if mindens < 0:
            self.context.print('Negative densities found! (magnitude %s)' %
                               np.abs(mindens), flush=False)
            self.context.print('These will be reset to 1E-12 elec/bohr^3)')
            self.n_g[np.where(self.n_g < 0.0)] = 1.0E-12

        r_g = finegd.get_grid_point_coordinates()
        self.x_g = 1.0 * r_g[0].flatten()
        self.y_g = 1.0 * r_g[1].flatten()
        self.z_g = 1.0 * r_g[2].flatten()
        self.gridsize = len(self.x_g)
        assert len(self.n_g) == self.gridsize

        # Enhancement factor for GGA
        if self.xcflags.is_apbe:
            nf_g = self.gs.hacky_all_electron_density(gridrefinement=4)
            gdf = self.gd.refine().refine()
            grad_v = [Gradient(gdf, v, n=1).apply for v in range(3)]
            gradnf_vg = gdf.empty(3)

            for v in range(3):
                grad_v[v](nf_g, gradnf_vg[v])

            self.s2_g = np.sqrt(np.sum(gradnf_vg[:, ::2, ::2, ::2]**2.0,
                                       0)).flatten()  # |\nabla\rho|
            self.s2_g *= 1.0 / (2.0 * (3.0 * np.pi**2.0)**(1.0 / 3.0) *
                                self.n_g**(4.0 / 3.0))
            # |\nabla\rho|/(2kF\rho) = s
            self.s2_g = self.s2_g**2  # s^2
            assert len(self.n_g) == len(self.s2_g)

            # Now we find all the regions where the
            # APBE kernel wants to be positive, and hack s to = 0,
            # so that we are really using the ALDA kernel
            # at these points
            apbe_g = get_pbe_fxc(self.n_g, self.s2_g)
            poskern_ind = np.where(apbe_g >= 0.0)
            if len(poskern_ind[0]) > 0:
                self.context.print(
                    'The APBE kernel takes positive values at '
                    + '%s grid points out of a total of %s (%3.2f%%).'
                    % (len(poskern_ind[0]), self.gridsize, 100.0 * len(
                        poskern_ind[0]) / self.gridsize), flush=False)
                self.context.print('The ALDA kernel will be used at these '
                                   'points')
                self.s2_g[poskern_ind] = 0.0

    def _calculate_fhxc(self):
        for iq, q_c in enumerate(self.ibzq_qc):
            if iq < self.q_empty:  # don't recalculate q vectors
                continue

            yield iq, self.calculate_one_qpoint(iq, q_c)

    def calculate_one_qpoint(self, iq, q_c):
        qpd = SingleQPWDescriptor.from_q(q_c, self.ecut / Ha, self.gd)

        nG = qpd.ngmax
        G_G = qpd.G2_qG[0]**0.5  # |G+q|
        Gv_G = qpd.get_reciprocal_vectors(q=0, add_q=False)
        # G as a vector (note we are at a specific q point here so set q=0)

        # Distribute G vectors among processors
        # Later we calculate for iG' > iG,
        # so stagger allocation in order to balance load
        local_Gvec_grid_size = nG // self.context.comm.size
        my_Gints = (self.context.comm.rank + np.arange(0,
                    local_Gvec_grid_size * self.context.comm.size,
                    self.context.comm.size))

        if (self.context.comm.rank +
                (local_Gvec_grid_size) * self.context.comm.size) < nG:
            my_Gints = np.append(my_Gints,
                                 [self.context.comm.rank +
                                  local_Gvec_grid_size *
                                  self.context.comm.size])

        my_Gv_G = Gv_G[my_Gints]

        # XXX Should this be if self.ns == 2 and self.xcflags.spin_kernel?
        calc_spincorr = (self.ns == 2) and (self.xc == 'rALDA'
                                            or self.xc == 'rAPBE')

        if calc_spincorr:
            # Form spin-dependent kernel according to
            # PRB 88, 115131 (2013) equation 20
            # (note typo, should be \tilde{f^rALDA})
            # spincorr is just the ALDA exchange kernel
            # with a step function (\equiv \tilde{f^rALDA})
            # fHxc^{up up}     = fHxc^{down down} = fv_nospin + fv_spincorr
            # fHxc^{up down}   = fHxc^{down up}   = fv_nospin - fv_spincorr
            fv_spincorr_GG = np.zeros((nG, nG), dtype=complex)

        fv_nospin_GG = np.zeros((nG, nG), dtype=complex)

        for iG, Gv in zip(my_Gints, my_Gv_G):  # loop over G vecs

            # For all kernels we
            # treat head and wings analytically
            if G_G[iG] > 1.0E-5:
                # Symmetrised |q+G||q+G'|, where iG' >= iG
                mod_Gpq = np.sqrt(G_G[iG] * G_G[iG:])

                # Phase factor \vec{G}-\vec{G'}
                deltaGv = Gv - Gv_G[iG:]

                if self.xc == 'rALDA':
                    # rALDA trick: the Hartree-XC kernel is exactly
                    # zero for densities below rho_min =
                    # min_Gpq^3/(24*pi^2),
                    # so we don't need to include these contributions
                    # in the Fourier transform

                    min_Gpq = np.amin(mod_Gpq)
                    rho_min = min_Gpq**3.0 / (24.0 * np.pi**2.0)
                    small_ind = np.where(self.n_g >= rho_min)
                elif self.xcflags.is_apbe:
                    # rAPBE trick: the Hartree-XC kernel
                    # is exactly zero at grid points where
                    # min_Gpq > cutoff wavevector

                    min_Gpq = np.amin(mod_Gpq)
                    small_ind = np.where(min_Gpq <= np.sqrt(
                        -4.0 * np.pi /
                        get_pbe_fxc(self.n_g, self.s2_g)))
                else:
                    small_ind = np.arange(self.gridsize)

                phase_Gpq = np.exp(
                    -1.0j *
                    (deltaGv[:, 0, np.newaxis] * self.x_g[small_ind] +
                     deltaGv[:, 1, np.newaxis] * self.y_g[small_ind] +
                     deltaGv[:, 2, np.newaxis] * self.z_g[small_ind]))

                def scaled_fHxc(spincorr):
                    return self.get_scaled_fHxc_q(
                        q=mod_Gpq,
                        sel_points=small_ind,
                        Gphase=phase_Gpq,
                        spincorr=spincorr)

                fv_nospin_GG[iG, iG:] = scaled_fHxc(spincorr=False)

                if calc_spincorr:
                    fv_spincorr_GG[iG, iG:] = scaled_fHxc(spincorr=True)
            else:
                # head and wings of q=0 are dominated by
                # 1/q^2 divergence of scaled Coulomb interaction

                assert iG == 0

                # The [0, 0] element would ordinarily be set to
                # 'l' if we have nonlinear kernel (which we are
                # removing).  Now l=1.0 always:
                fv_nospin_GG[0, 0] = 1.0
                fv_nospin_GG[0, 1:] = 0.0

                if calc_spincorr:
                    fv_spincorr_GG[0, :] = 0.0

            # End loop over G vectors

        self.context.comm.sum(fv_nospin_GG)

        # We've only got half the matrix here,
        # so add the hermitian conjugate:
        fv_nospin_GG += np.conj(fv_nospin_GG.T)
        # but now the diagonal's been doubled,
        # so we multiply these elements by 0.5
        fv_nospin_GG[np.diag_indices(nG)] *= 0.5

        # End of loop over coupling constant

        if calc_spincorr:
            self.context.comm.sum(fv_spincorr_GG)
            fv_spincorr_GG += np.conj(fv_spincorr_GG.T)
            fv_spincorr_GG[np.diag_indices(nG)] *= 0.5

        self.context.print('q point %s complete' % iq)

        if self.context.comm.rank == 0:
            if calc_spincorr:
                # Form the block matrix kernel
                fv_full_2G2G = np.empty((2 * nG, 2 * nG), dtype=complex)
                fv_full_2G2G[:nG, :nG] = fv_nospin_GG + fv_spincorr_GG
                fv_full_2G2G[:nG, nG:] = fv_nospin_GG - fv_spincorr_GG
                fv_full_2G2G[nG:, :nG] = fv_nospin_GG - fv_spincorr_GG
                fv_full_2G2G[nG:, nG:] = fv_nospin_GG + fv_spincorr_GG
                fhxc_sGsG = fv_full_2G2G

            else:
                fhxc_sGsG = fv_nospin_GG

            return fhxc_sGsG
        else:
            return None

    def get_scaled_fHxc_q(self, q, sel_points, Gphase, spincorr):
        # Given a coupling constant l, construct the Hartree-XC
        # kernel in q space a la Lein, Gross and Perdew,
        # Phys. Rev. B 61, 13431 (2000):
        #
        # f_{Hxc}^\lambda(q,\omega,r_s) = \frac{4\pi \lambda }{q^2}  +
        # \frac{1}{\lambda} f_{xc}(q/\lambda,\omega/\lambda^2,\lambda r_s)
        #
        # divided by the unscaled Coulomb interaction!!
        #
        # i.e. this subroutine returns f_{Hxc}^\lambda(q,\omega,r_s)
        #                              *  \frac{q^2}{4\pi}
        # = \lambda * [\frac{(q/lambda)^2}{4\pi}
        #              f_{Hxc}(q/\lambda,\omega/\lambda^2,\lambda r_s)]
        # = \lambda * [1/scaled_coulomb * fHxc computed with scaled quantities]

        # Apply scaling
        rho = self.n_g[sel_points]

        # GGA enhancement factor s is lambda independent,
        # but we might want to truncate it
        if self.xcflags.is_apbe:
            s2_g = self.s2_g[sel_points]
        else:
            s2_g = None

        l = 1.0  # Leftover from the age of non-linear kernels.
        # This would be an integration weight or something.
        scaled_q = q / l
        scaled_rho = rho / l**3.0
        scaled_rs = (3.0 / (4.0 * np.pi * scaled_rho))**(1.0 / 3.0
                                                         )  # Wigner radius

        if not spincorr:
            scaled_kernel = l * self.get_fHxc_q(scaled_rs, scaled_q, Gphase,
                                                s2_g)
        else:
            scaled_kernel = l * self.get_spinfHxc_q(scaled_rs, scaled_q,
                                                    Gphase, s2_g)

        return scaled_kernel

    def get_fHxc_q(self, rs, q, Gphase, s2_g):
        # Construct fHxc(q,G,:), divided by scaled Coulomb interaction

        heg = HEG(rs)
        qF = heg.qF

        fHxc_Gr = get_fHxc_Gr(self.xcflags, rs, q, qF, s2_g)

        # Integrate over r with phase
        fHxc_Gr *= Gphase
        fHxc_GG = np.sum(fHxc_Gr, 1) / self.gridsize
        return fHxc_GG

    def get_spinfHxc_q(self, rs, q, Gphase, s2_g):
        qF = HEG(rs).qF

        if self.xc == 'rALDA':
            fspinHxc_Gr = get_fspinHxc_Gr_rALDA(qF, q)

        elif self.xc == 'rAPBE':
            fspinHxc_Gr = get_fspinHxc_Gr_rAPBE(rs, q, s2_g)

        fspinHxc_Gr *= Gphase
        fspinHxc_GG = np.sum(fspinHxc_Gr, 1) / self.gridsize
        return fspinHxc_GG


class KernelDens(KernelIntegrator):
    def __init__(self, *, unit_cells, density_cut, **kwargs):
        super().__init__(**kwargs)

        self.unit_cells = unit_cells
        self.density_cut = density_cut

        self.A_x = -(3 / 4.) * (3 / np.pi)**(1 / 3.)

        self.n_g = self.gs.hacky_all_electron_density(gridrefinement=1)

        if self.xc[-3:] == 'PBE':
            nf_g = self.gs.hacky_all_electron_density(gridrefinement=2)
            gdf = self.gd.refine()
            grad_v = [Gradient(gdf, v, n=1).apply for v in range(3)]
            gradnf_vg = gdf.empty(3)
            for v in range(3):
                grad_v[v](nf_g, gradnf_vg[v])
            self.gradn_vg = gradnf_vg[:, ::2, ::2, ::2]

        qd = KPointDescriptor(self.ibzq_qc)
        self.pd = PWDescriptor(self.ecut / Ha, self.gd, complex, qd)

    def _calculate_fhxc(self):
        if self.xc[0] == 'r':  # wth?
            assert self.xcflags.spin_kernel
            yield from self.calculate_rkernel()
        else:
            assert self.xc[0] == 'A'  # wth?
            assert self.xc == 'ALDA'
            yield from self.calculate_local_kernel()

    def calculate_rkernel(self):
        gd = self.gd
        ng_c = gd.N_c
        cell_cv = gd.cell_cv
        icell_cv = 2 * np.pi * np.linalg.inv(cell_cv)
        vol = gd.volume

        ns = self.gs.nspins
        n_g = self.n_g  # density on rough grid

        fx_g = ns * self.get_fxc_g(n_g)  # local exchange kernel
        try:
            qc_g = (-4 * np.pi * ns / fx_g)**0.5  # cutoff functional
        except FloatingPointError as err:
            msg = (
                'Kernel is negative yet we want its square root.  '
                'You probably should not rely on this feature at all.  ',
                'See discussion https://gitlab.com/gpaw/gpaw/-/issues/723')
            raise RuntimeError(msg) from err
        flocal_g = qc_g**3 * fx_g / (6 * np.pi**2)  # ren. x-kernel for r=r'
        Vlocal_g = 2 * qc_g / np.pi  # ren. Hartree kernel for r=r'

        ng = np.prod(ng_c)  # number of grid points
        r_vg = gd.get_grid_point_coordinates()
        rx_g = r_vg[0].flatten()
        ry_g = r_vg[1].flatten()
        rz_g = r_vg[2].flatten()

        self.context.print('    %d grid points and %d plane waves at the '
                           'Gamma point' % (ng, self.pd.ngmax), flush=False)

        # Unit cells
        R_Rv = []
        weight_R = []
        nR_v = self.unit_cells
        nR = np.prod(nR_v)
        for i in range(-nR_v[0] + 1, nR_v[0]):
            for j in range(-nR_v[1] + 1, nR_v[1]):
                for h in range(-nR_v[2] + 1, nR_v[2]):
                    R_Rv.append(i * cell_cv[0] + j * cell_cv[1] +
                                h * cell_cv[2])
                    weight_R.append((nR_v[0] - abs(i)) * (nR_v[1] - abs(j)) *
                                    (nR_v[2] - abs(h)) / float(nR))
        if nR > 1:
            # with more than one unit cell only the exchange kernel is
            # calculated on the grid. The bare Coulomb kernel is added
            # in PW basis and Vlocal_g only the exchange part
            dv = self.gs.density.gd.dv
            gc = (3 * dv / 4 / np.pi)**(1 / 3.)
            Vlocal_g -= 2 * np.pi * gc**2 / dv
            self.context.print(
                '    Lattice point sampling: (%s x %s x %s)^2 '
                % (nR_v[0], nR_v[1], nR_v[2]) + ' Reduced to %s lattice points'
                % len(R_Rv), flush=False)

        l_g_size = -(-ng // self.context.comm.size)
        l_g_range = range(self.context.comm.rank * l_g_size,
                          min((self.context.comm.rank + 1) * l_g_size, ng))

        fhxc_qsGr = {}
        for iq in range(len(self.ibzq_qc)):
            fhxc_qsGr[iq] = np.zeros(
                (ns, len(self.pd.G2_qG[iq]), len(l_g_range)), dtype=complex)

        inv_error = np.seterr()
        np.seterr(invalid='ignore')
        np.seterr(divide='ignore')

        t0 = time()
        # Loop over Lattice points
        for i, R_v in enumerate(R_Rv):
            # Loop over r'. f_rr and V_rr are functions of r (dim. as r_vg[0])
            if i == 1:
                self.context.print(
                    '      Finished 1 cell in %s seconds' % int(time() - t0) +
                    ' - estimated %s seconds left' % int((len(R_Rv) - 1) *
                                                         (time() - t0)))
            if len(R_Rv) > 5:
                if (i + 1) % (len(R_Rv) / 5 + 1) == 0:
                    self.context.print(
                        '      Finished %s cells in %s seconds'
                        % (i, int(time() - t0)) + ' - estimated '
                        '%s seconds left' % int((len(R_Rv) - i) * (time() -
                                                                   t0) / i))
            for g in l_g_range:
                rx = rx_g[g] + R_v[0]
                ry = ry_g[g] + R_v[1]
                rz = rz_g[g] + R_v[2]

                # |r-r'-R_i|
                rr = ((r_vg[0] - rx)**2 + (r_vg[1] - ry)**2 +
                      (r_vg[2] - rz)**2)**0.5

                n_av = (n_g + n_g.flatten()[g]) / 2.
                fx_g = ns * self.get_fxc_g(n_av, index=g)
                qc_g = (-4 * np.pi * ns / fx_g)**0.5
                x = qc_g * rr
                osc_x = np.sin(x) - x * np.cos(x)
                f_rr = fx_g * osc_x / (2 * np.pi**2 * rr**3)
                if nR > 1:  # include only exchange part of the kernel here
                    V_rr = (sici(x)[0] * 2 / np.pi - 1) / rr
                else:  # include the full kernel (also hartree part)
                    V_rr = (sici(x)[0] * 2 / np.pi) / rr

                # Terms with r = r'
                if (np.abs(R_v) < 0.001).all():
                    tmp_flat = f_rr.flatten()
                    tmp_flat[g] = flocal_g.flatten()[g]
                    f_rr = tmp_flat.reshape(ng_c)
                    tmp_flat = V_rr.flatten()
                    tmp_flat[g] = Vlocal_g.flatten()[g]
                    V_rr = tmp_flat.reshape(ng_c)
                    del tmp_flat

                f_rr[np.where(n_av < self.density_cut)] = 0.0
                V_rr[np.where(n_av < self.density_cut)] = 0.0

                f_rr *= weight_R[i]
                V_rr *= weight_R[i]

                # r-r'-R_i
                r_r = np.array([r_vg[0] - rx, r_vg[1] - ry, r_vg[2] - rz])

                # Fourier transform of r
                for iq, q in enumerate(self.ibzq_qc):
                    q_v = np.dot(q, icell_cv)
                    e_q = np.exp(-1j * gemmdot(q_v, r_r, beta=0.0))
                    f_q = self.pd.fft((f_rr + V_rr) * e_q, iq) * vol / ng
                    fhxc_qsGr[iq][0, :, g - l_g_range[0]] += f_q
                    if ns == 2:
                        f_q = self.pd.fft(V_rr * e_q, iq) * vol / ng
                        fhxc_qsGr[iq][1, :, g - l_g_range[0]] += f_q

        self.context.comm.barrier()

        np.seterr(**inv_error)

        for iq, q in enumerate(self.ibzq_qc):
            npw = len(self.pd.G2_qG[iq])
            fhxc_sGsG = np.zeros((ns * npw, ns * npw), complex)
            # parallelize over PW below
            l_pw_size = -(-npw // self.context.comm.size)
            l_pw_range = range(self.context.comm.rank * l_pw_size,
                               min((self.context.comm.rank + 1) * l_pw_size,
                                   npw))

            if self.context.comm.size > 1:
                # redistribute grid and plane waves in fhxc_qsGr[iq]
                bg1 = BlacsGrid(self.context.comm, 1, self.context.comm.size)
                bg2 = BlacsGrid(self.context.comm, self.context.comm.size, 1)
                bd1 = bg1.new_descriptor(npw, ng, npw,
                                         -(-ng // self.context.comm.size))
                bd2 = bg2.new_descriptor(npw, ng,
                                         -(-npw // self.context.comm.size),
                                         ng)

                fhxc_Glr = np.zeros((len(l_pw_range), ng), dtype=complex)
                if ns == 2:
                    Koff_Glr = np.zeros((len(l_pw_range), ng), dtype=complex)

                r = Redistributor(bg1.comm, bd1, bd2)
                r.redistribute(fhxc_qsGr[iq][0], fhxc_Glr, npw, ng)
                if ns == 2:
                    r.redistribute(fhxc_qsGr[iq][1], Koff_Glr, npw, ng)
            else:
                fhxc_Glr = fhxc_qsGr[iq][0]
                if ns == 2:
                    Koff_Glr = fhxc_qsGr[iq][1]

            # Fourier transform of r'
            for iG in range(len(l_pw_range)):
                f_g = fhxc_Glr[iG].reshape(ng_c)
                f_G = self.pd.fft(f_g.conj(), iq) * vol / ng
                fhxc_sGsG[l_pw_range[0] + iG, :npw] = f_G.conj()
                if ns == 2:
                    v_g = Koff_Glr[iG].reshape(ng_c)
                    v_G = self.pd.fft(v_g.conj(), iq) * vol / ng
                    fhxc_sGsG[npw + l_pw_range[0] + iG, :npw] = v_G.conj()

            if ns == 2:  # f_00 = f_11 and f_01 = f_10
                fhxc_sGsG[:npw, npw:] = fhxc_sGsG[npw:, :npw]
                fhxc_sGsG[npw:, npw:] = fhxc_sGsG[:npw, :npw]

            self.context.comm.sum(fhxc_sGsG)
            fhxc_sGsG /= vol

            if self.context.comm.rank == 0:
                if nR > 1:  # add Hartree kernel evaluated in PW basis
                    Gq2_G = self.pd.G2_qG[iq]
                    if (q == 0).all():
                        Gq2_G = Gq2_G.copy()
                        Gq2_G[0] = 1.
                    vq_G = 4 * np.pi / Gq2_G
                    fhxc_sGsG += np.tile(np.eye(npw) * vq_G, (ns, ns))
                yield iq, fhxc_sGsG
            else:
                yield iq, None

    def calculate_local_kernel(self):
        # Standard ALDA exchange kernel
        # Use with care. Results are very difficult to converge
        # Sensitive to density_cut
        ns = self.gs.nspins
        gd = self.gd
        pd = self.pd
        cell_cv = gd.cell_cv
        icell_cv = 2 * np.pi * np.linalg.inv(cell_cv)
        vol = gd.volume

        fxc_sg = ns * self.get_fxc_g(ns * self.n_g)
        fxc_sg[np.where(self.n_g < self.density_cut)] = 0.0

        r_vg = gd.get_grid_point_coordinates()

        for iq in range(len(self.ibzq_qc)):
            Gvec_Gc = np.dot(pd.get_reciprocal_vectors(q=iq, add_q=False),
                             cell_cv / (2 * np.pi))
            npw = len(Gvec_Gc)
            l_pw_size = -(-npw // self.context.comm.size)
            l_pw_range = range(self.context.comm.rank * l_pw_size,
                               min((self.context.comm.rank + 1) * l_pw_size,
                                   npw))
            fhxc_sGsG = np.zeros((ns * npw, ns * npw), dtype=complex)
            for s in range(ns):
                for iG in l_pw_range:
                    for jG in range(npw):
                        fxc = fxc_sg[s].copy()
                        dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
                        dG_v = np.dot(dG_c, icell_cv)
                        dGr_g = gemmdot(dG_v, r_vg, beta=0.0)
                        ft_fxc = gd.integrate(np.exp(-1j * dGr_g) * fxc)
                        fhxc_sGsG[s * npw + iG, s * npw + jG] = ft_fxc

            self.context.comm.sum(fhxc_sGsG)
            fhxc_sGsG /= vol

            Gq2_G = self.pd.G2_qG[iq]
            if (self.ibzq_qc[iq] == 0).all():
                Gq2_G[0] = 1.
            vq_G = 4 * np.pi / Gq2_G
            fhxc_sGsG += np.tile(np.eye(npw) * vq_G, (ns, ns))

            yield iq, fhxc_sGsG

    def get_fxc_g(self, n_g, index=None):
        if self.xc[-3:] == 'LDA':
            return self.get_lda_g(n_g)
        elif self.xc[-3:] == 'PBE':
            return self.get_pbe_g(n_g, index=index)
        else:
            raise '%s kernel not recognized' % self.xc

    def get_lda_g(self, n_g):
        return (4. / 9.) * self.A_x * n_g**(-2. / 3.)

    def get_pbe_g(self, n_g, index=None):
        if index is None:
            gradn_vg = self.gradn_vg
        else:
            gradn_vg = self.gs.density.gd.empty(3)
            for v in range(3):
                gradn_vg[v] = (self.gradn_vg[v] +
                               self.gradn_vg[v].flatten()[index]) / 2

        kf_g = (3. * np.pi**2 * n_g)**(1 / 3.)
        s2_g = np.zeros_like(n_g)
        for v in range(3):
            axpy(1.0, gradn_vg[v]**2, s2_g)
        s2_g /= 4 * kf_g**2 * n_g**2

        from gpaw.xc.fxc_kernels import get_pbe_fxc
        return get_pbe_fxc(n_g, s2_g)


class XCFlags:
    _accepted_flags = {
        'RPA',
        'rALDA',
        'rAPBE',
        'ALDA'}

    _spin_kernels = {'rALDA', 'rAPBE', 'ALDA'}

    def __init__(self, xc):
        if xc not in self._accepted_flags:
            raise RuntimeError('%s kernel not recognized' % self.xc)

        self.xc = xc

    @property
    def spin_kernel(self):
        # rALDA/rAPBE are the only kernels which have spin-dependent forms
        return self.xc in self._spin_kernels

    @property
    def is_apbe(self):
        # If new GGA kernels are added, maybe there should be an
        # is_gga property.
        return self.xc in {'rAPBE'}

    def choose_avg_scheme(self, avg_scheme=None):
        xc = self.xc

        if self.spin_kernel:
            if avg_scheme is None:
                avg_scheme = 'density'
                # Two-point scheme default for rALDA and rAPBE

        if avg_scheme == 'density':
            assert self.spin_kernel, ('Two-point density average '
                                      'only implemented for rALDA and rAPBE')

        elif xc != 'RPA':
            avg_scheme = 'wavevector'
        else:
            avg_scheme = None

        return avg_scheme
