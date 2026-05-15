import sys
from ase.units import Hartree


class KSSRestrictor:
    """Object to handle KSSingles restrictions"""
    defaults = {'eps': 0.01,
                'istart': 0,
                'jend': sys.maxsize,
                'energy_range': None,
                'from': None,
                'to': None}

    def __init__(self, dictionary=None):
        if dictionary is None:
            dictionary = {}
        self._vals = {}
        self.update(dictionary)

    def __getitem__(self, index):
        assert index in self.defaults
        return self._vals.get(index, self.defaults[index])

    def __setitem__(self, index, value):
        assert index in self.defaults
        self._vals[index] = value

    def update(self, dictionary):
        for key, value in dictionary.items():
            self[key] = value

    def emin_emax(self):
        emin = -sys.float_info.max
        emax = sys.float_info.max
        if self['energy_range'] is not None:
            try:
                emin, emax = self['energy_range']
                emin /= Hartree
                emax /= Hartree
            except TypeError:
                emax = self['energy_range'] / Hartree
        return emin, emax

    @property
    def values(self):
        dct = {}
        dct.update(self._vals)
        return dct

    def __str__(self):
        return str(self.values)

    def is_good(self, ks) -> bool:
        """Check if Kohn-Sham single fulfills the criterion"""
        emin, emax = self.emin_emax()

        ok = (ks.fij / ks.weight) > self['eps']
        ok &= ks.i >= self['istart'] and ks.j <= self['jend']
        ok &= ks.energy >= emin and ks.energy <= emax
        if self['from'] is not None:
            ok &= ks.i in self['from']
        if self['to'] is not None:
            ok &= ks.j in self['to']
        return ok
