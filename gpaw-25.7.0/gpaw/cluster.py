"""Extensions to the ase Atoms class

"""
from ase import Atoms
from ase.build.connected import connected_indices
from ase.utils import deprecated

from .utilities.adjust_cell import adjust_cell


class Cluster(Atoms):
    """A class for cluster structures
    to enable simplified manipulation"""

    def __init__(self, *args, **kwargs):

        self.data = {}

        if len(args) > 0:
            filename = args[0]
            if isinstance(filename, str):
                self.read(filename, kwargs.get('filetype'))
                return
        else:
            Atoms.__init__(self, [])

        if kwargs.get('filename') is not None:
            filename = kwargs.pop('filename')
            Atoms.__init__(self, *args, **kwargs)
            self.read(filename, kwargs.get('filetype'))
        else:
            Atoms.__init__(self, *args, **kwargs)

    @deprecated(
        'Please use connected_indices from ase.build.connected instead.')
    def find_connected(self, index, dmax=None, scale=1.5):
        """Find atoms connected to self[index] and return them."""
        return self[connected_indices(self, index, dmax, scale)]

    @deprecated(
        'Please use adjust_cell from gpaw.utilities.adjust_cell instead.')
    def minimal_box(self, border=4, h=None, multiple=4) -> None:
        adjust_cell(self, border, h, multiple)
