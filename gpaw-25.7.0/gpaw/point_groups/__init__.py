"""Point Group Symmetry.

The code was originally written for the paper:

    S. Kaappa, S. Malola, H. Hakkinen

    J. Phys. Chem. A; vol. 122, 43, pp. 8576-8584 (2018)

"""

from .group import PointGroup
from .check import SymmetryChecker

__all__ = ['PointGroup', 'SymmetryChecker', 'point_group_names']

point_group_names = ['C2', 'C2v', 'C3v', 'D2d', 'D3h', 'D5', 'D5h',
                     'Ico', 'Ih', 'Oh', 'Td', 'Th']
