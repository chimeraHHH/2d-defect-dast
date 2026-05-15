"""
Calculates Raman matrices.

i -> j -> m -> n
i, n are valence; j, m are conduction, also i=n in the end
see https://doi.org/10.1038/s41467-020-16529-6
"""
import numpy as np

from ase.units import invcm, Hartree
from ase.utils.filecache import MultiFileJSONCache
from gpaw.calculator import GPAW
from gpaw.typing import ArrayND

# TODO: only take kd, don't need whole calculator


class ResonantRamanCalculator:
    def __init__(
        self,
        calc: GPAW,
        wph_w: ArrayND,
        momname: str = "mom_skvnm.npy",
        elphname: str = "gsqklnn.npy",
        raman_name: str = "Rlab",
    ):
        """Resonant Raman Matrix calculator

        Parameters
        ----------
        calc: GPAW
            GPAW calculator object.
        w_ph: str, np.ndarray
            Zone centre phonon frequencies in eV
        momname: str
            Name of momentum file
        elphname: str
            Name of electron-phonon file
        raman_name: str
            Name of Rlab file cache. Default 'Rlab'
        """
        # those won't make sense here
        assert calc.wfs.gd.comm.size == 1
        assert calc.wfs.bd.comm.size == 1
        self.kd = calc.wfs.kd
        self.calc = calc

        # Phonon frequencies
        if isinstance(wph_w, str):
            wph_w = np.load(wph_w)
        assert max(wph_w) < 1.0  # else not eV units
        self.w_ph = wph_w

        # Load files
        mom_skvnm = np.load(momname, mmap_mode="c")
        g_sqklnn = np.load(elphname, mmap_mode="c")  # [s,q=0,k,l,n,m]
        self.mom_skvnm = mom_skvnm
        self.g_sqklnn = g_sqklnn

        # Define a few more variables
        nspins = g_sqklnn.shape[0]
        nk = g_sqklnn.shape[2]
        assert wph_w.shape[0] == g_sqklnn.shape[3]
        assert mom_skvnm.shape[0] == nspins
        assert mom_skvnm.shape[1] == nk
        assert mom_skvnm.shape[-1] == g_sqklnn.shape[-1]

        # JSON cache
        self.raman_cache = MultiFileJSONCache(raman_name)
        with self.raman_cache.lock("phonon_frequencies") as handle:
            if handle is not None:
                handle.save(wph_w)
        # Resonant part of Raman tensor does not have a frequency grid
        with self.raman_cache.lock("frequency_grid") as handle:
            if handle is not None:
                handle.save(None)

    @classmethod
    def resonant_term(
        cls,
        f_vc: ArrayND,
        E_vc: ArrayND,
        mom_dnn: ArrayND,
        elph_lnn: ArrayND,
        nc: int,
        nv: int,
        w_in: float,
        wph_w: ArrayND,
    ) -> ArrayND:
        """Resonant term of the Raman tensor"""
        nmodes = elph_lnn.shape[0]
        term_l = np.zeros((nmodes), dtype=complex)
        t_ij = f_vc * mom_dnn[0, :nv, nc:] / (w_in - E_vc)
        for l in range(nmodes):
            t_xx = elph_lnn[l]
            t_mn = f_vc.T * mom_dnn[1, nc:, :nv] / (w_in - wph_w[l] - E_vc.T)
            term_l[l] += np.einsum("sj,jm,ms", t_ij, t_xx[nc:, nc:], t_mn)
            term_l[l] -= np.einsum("is,ni,sn", t_ij, t_xx[:nv, :nv], t_mn)
        return term_l

    def calculate(self, w_in, d_i, d_o, gamma_l=0.1, limit_sum=False):
        """Calculate resonant Raman matrix

        Parameters
        ----------
        w_in: float
            Laser frequency in eV
        d_i: int
            Incoming polarization
        d_o: int
            Outgoing polarization
        gamma_l: float
            Line broadening in eV, (0.1eV=806rcm)
        limit_sum: bool
            Limit sum to occupied valence/unoccupied conduction states
            Use for debugging and testing. Don't use in production
        """

        raman_l = np.zeros((self.w_ph.shape[0]), dtype=complex)

        print(f"Calculating Raman spectrum: Laser frequency = {w_in} eV")

        # Loop over kpoints - this is parallelised
        for kpt in self.calc.wfs.kpt_u:
            # print(f"Rank {self.kd.comm.rank}: s={kpt.s}, k={kpt.k}")

            f_n = kpt.f_n / kpt.weight
            assert np.isclose(max(f_n), 1.0, atol=0.1)

            vs = np.arange(0, len(f_n))
            cs = np.arange(0, len(f_n))
            nc = 0
            nv = len(f_n)
            if limit_sum:  # good to test that we got arrays right
                vs = np.where(f_n >= 0.1)[0]
                cs = np.where(f_n < 0.9)[0]
                nv = max(vs) + 1  # VBM+1 index
                nc = min(cs)  # CBM index

            # Precalculate f * (1-f) term
            f_vc = np.outer(f_n[vs], 1.0 - f_n[cs])

            # Precalculate E-E term
            E_vc = np.zeros((len(vs), len(cs)), dtype=complex) + 1j * gamma_l
            for n in range(len(vs)):
                E_vc[n] += (kpt.eps_n[cs] - kpt.eps_n[n]) * Hartree

            # Obtain appropriate part of mom and g arrays
            pols = [d_i, d_o]
            mom_dnn = np.ascontiguousarray(self.mom_skvnm[kpt.s, kpt.k, pols])
            assert mom_dnn.shape[0] == 2
            g_lnn = np.ascontiguousarray(self.g_sqklnn[kpt.s, 0, kpt.k])

            # Raman contribution of this k-point
            raman_l += self.resonant_term(
                f_vc, E_vc, mom_dnn, g_lnn, nc, nv, w_in, self.w_ph
            ) * kpt.weight
        # Collect parallel contributions
        self.kd.comm.sum(raman_l)

        if self.kd.comm.rank == 0:
            print(f"Raman intensities per mode in {'xyz'[d_i]}{'xyz'[d_o]}")
            print("--------------------------")
            ff = "  Phonon {} with energy = {:4.2f} rcm: {:.4f}"
            for l in range(self.w_ph.shape[0]):
                print(ff.format(l, self.w_ph[l] / invcm, raman_l[l]))

        return raman_l

    def calculate_raman_tensor(self, w_in, gamma_l=0.1):
        """Calculate whole Raman tensor

        Parameters
        ----------
        w_in: float
            Laser frequency in eV
        gamma_l: float
            Line broadening in eV, (0.1eV=806rcm)
        """
        # If exist already, don't recompute
        for i in range(3):
            for j in range(3):
                with self.raman_cache.lock(f"{'xyz'[i]}{'xyz'[j]}") as handle:
                    if handle is None:
                        continue
                    R_l = self.calculate(w_in, i, j, gamma_l)
                    self.kd.comm.barrier()
                    handle.save(R_l)

    def nm_to_eV(self, laser_wave_length):
        return 1.239841e3 / laser_wave_length
