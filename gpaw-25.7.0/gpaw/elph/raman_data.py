"""
Calculates Raman matrices from Raman tensor
"""
import numpy as np
from typing import Tuple

from ase.units import invcm
from ase.utils.filecache import MultiFileJSONCache
from gpaw.typing import ArrayND


def gaussian(w, sigma):
    g = 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-(w**2) / (2 * sigma**2))
    return g


class RamanData:
    def __init__(self, raman_name="Rlab", gridspacing=1.0) -> None:
        """Raman Spectroscopy data.

        Parameters
        ----------
        raman_name: str
            Name of Rlab file cache. Default 'Rlab'
        gridspacing: float
            grid spacing in cm^-1, default 1 rcm
        """
        # JSON cache
        self.raman_cache = MultiFileJSONCache(raman_name)
        self.wph_w = self.raman_cache["phonon_frequencies"]  # eV
        self.wrl_w = self.raman_cache["frequency_grid"]  # eV

        gridmax = self.wph_w[-1] + 6.3e-3  # highest freq + 50rcm
        self.gridev_w = np.linspace(
            0.0,
            gridmax,
            num=int(gridmax / (gridspacing * invcm) + 1),
        )  # eV

    def calculate_raman_intensity(self, d_i, d_o, T=300, sigma=5.0):
        """
        Calculates Raman intensities from Raman tensor.

        Returns bare `|R|^2` and and Bose occupation weighted values

        Parameters
        ----------
        d_i: int
            Incoming polarization
        d_o: int
            Outgoing polarization
        T: float
            Temperature in Kelvin
        sigma: float
            Gaussian line width in rcm, default 5rcm
        """

        KtoeV = 8.617278e-5
        sigmaev = sigma * invcm  # 1eV = 8.065544E3 rcm

        # grid for spectrum with extra space and sigma/5 spacing
        if self.gridev_w is None:
            self.gridev_w = np.linspace(
                0.0,
                self.wph_w[-1] * 1.1,
                num=int(self.wph_w[-1] / (sigmaev / 5) + 100),
            )

        int_bare = np.zeros_like(self.gridev_w, dtype=float)
        int_occ = np.zeros_like(self.gridev_w, dtype=float)

        # Note: Resonant Raman does not have a frequency grid
        R_lw = self.raman_cache[f"{'xyz'[d_i]}{'xyz'[d_o]}"]

        for l in range(len(self.wph_w)):
            occ = 1.0 / (np.exp(self.wph_w[l] / (KtoeV * T)) - 1.0) + 1.0
            delta_w = gaussian((self.gridev_w - self.wph_w[l]), sigmaev)
            R2_w = np.abs(R_lw[l])**2
            R2d_w = R2_w * delta_w
            int_bare += R2d_w
            int_occ += occ / self.wph_w[l] * R2d_w

        return int_bare, int_occ

    def calculate_raman_spectrum(self,
                                 entries, T=300, sigma=5.0
                                 ) -> Tuple[ArrayND, ArrayND]:
        """
        Calculates Raman intensities from Raman tensor.

        Returns Raman shift in eV, bare `|R|^2`
        and Bose occupation weighted `|R|^2` values

        Parameters
        ----------
        entries: str, list
            Sting or list of strings with desired polarisaitons
            For example: ["xx", "yy", "xy", "yx"]
        T: float
            Temperature in Kelvin
        sigma: float
            Gaussian line width in rcm, default 5rcm
        """
        if isinstance(entries, str):
            entries = [
                entries,
            ]

        spectrum_w = np.zeros_like(self.gridev_w)

        polnum = {"x": 0, "y": 1, "z": 2}
        for entry in entries:
            d_i = polnum[entry[0]]
            d_o = polnum[entry[1]]
            (int_bare, _) = self.calculate_raman_intensity(d_i, d_o, T, sigma)
            spectrum_w += int_bare

        return self.gridev_w, spectrum_w

    @classmethod
    def plot_raman(
        cls,
        figname,
        grid_w,
        spectra_nw,
        labels_n=None,
        relative=False,
    ):
        """Plots a given Raman spectrum.

        Parameters
        ----------
        figname: str
            Filename for figure.
        grid_w:
            Frequency grid of spectrum in eV
        spectra_nw: np.ndarray
            Raman spectra to plot
        labels_n: list
            Labels for the legend
        relative: bool
            If true, normalize each spectrum to 1
            Default is False
        """

        from scipy import signal
        import matplotlib.pyplot as plt

        if not isinstance(spectra_nw, np.ndarray):
            spectra_nw = np.asarray(spectra_nw)

        ylabel = "Intensity (arb. units)"
        if relative:
            ylabel = "I/I_max"
            spectra_nw /= np.max(spectra_nw, axis=1)[:, np.newaxis]
        else:
            spectra_nw /= np.max(spectra_nw)

        nspec = spectra_nw.shape[0]
        if labels_n is None:
            labels_n = nspec * [None]

        grid_w = cls.eVtorcm(grid_w)

        for i in range(nspec):
            peaks = signal.find_peaks(spectra_nw[i])[0]
            locations = np.take(grid_w, peaks)
            intensities = np.take(spectra_nw[i], peaks)

            plt.plot(grid_w, spectra_nw[i], label=labels_n[i])

            for j, loc in enumerate(locations):
                if intensities[j] / np.max(intensities) > 0.05:
                    plt.axvline(x=loc, color="grey", linestyle="--")

        plt.minorticks_on()
        if labels_n[0] is not None:
            plt.legend()
        plt.title("Raman intensity")
        plt.xlabel("Raman shift (cm$^{-1}$)")
        plt.ylabel(ylabel)
        if relative:
            plt.yticks([])

        plt.savefig(figname, dpi=300)
        plt.clf()

    @classmethod
    def eVtorcm(cls, energy):
        return energy / invcm
