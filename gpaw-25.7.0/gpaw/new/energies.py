"""PAW-DFT energy-contributions."""

from ase.units import Ha

# Contributions to free energy:
NAMES = ['kinetic', 'coulomb', 'zero', 'external', 'xc', 'entropy',
         'spinorbit', 'hybrid_xc']

# Other allowed names:
OTHERS = {'band', 'kinetic_correction', 'extrapolation',
          'hybrid_kinetic_correction'}


class DFTEnergies:
    def __init__(self, **energies: float):
        self._energies: dict[str, float] = {}
        self._total_free: float | None
        self.set(**energies)

    def set(self, **energies: float) -> None:
        # assert energies.keys() <= set(NAMES) | OTHERS, energies
        self._energies.update(energies)
        self._total_free = None

    @property
    def kinetic(self):
        e = self._energies.get('kinetic')
        if e is not None:
            return e
        # Use Kohn-Sham eq. to get kinetic energy as sum over
        # occupied eigenvalues + correction:
        return (self._energies['band'] +
                self._energies['kinetic_correction'] +
                self._energies.get('hybrid_kinetic_correction', 0.0))

    @property
    def total_free(self) -> float:
        if self._total_free is None:
            energies = self._energies.copy()
            energies['kinetic'] = self.kinetic
            self._total_free = sum(energies.get(name, 0.0) for name in energies
                                   if name not in OTHERS)
        return self._total_free

    @property
    def total_extrapolated(self) -> float:
        return self.total_free + self._energies['extrapolation']

    def __repr__(self) -> str:
        s = ', '.join(f'{k}={v}' for k, v in self._energies.items())
        return f'DFTEnergies({s})'

    @property
    def extensions_energies(self) -> list[tuple[str, float]]:
        return [(name, self._energies.get(name, 0.0))
                for name in self._energies
                if name not in OTHERS and name not in NAMES]

    def summary(self, log) -> None:
        for name in NAMES:
            if name in OTHERS:
                continue
            e = self._energies.get(name)
            if e is None:
                if name != 'kinetic':
                    continue
                e = self.kinetic
            log(f'{name + ":":10}   {e * Ha:14.6f}')
        extensions = self.extensions_energies
        if extensions:
            log('--------extensions:---------')
            for name, e in extensions:
                log(f'{name + ":":12} {e * Ha:14.6f}')
        log('----------------------------')
        log(f'Free energy: {self.total_free * Ha:14.6f}')
        log(f'Extrapolated:{self.total_extrapolated * Ha:14.6f}\n')

    def write_to_gpw(self, writer):
        writer.write(**{name: e * Ha for name, e in self._energies.items()})
