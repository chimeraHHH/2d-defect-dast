"""Point-group object."""
from typing import Dict
import numpy as np
from gpaw.typing import Array2D


class PointGroup:
    def __init__(self, name: str):
        """Point-group object.

        Name must be one of: C2, C2v, C3v, D2d, D3h, D5, D5h,
        Ico, Ih, Oh, Td or Th.
        """
        import gpaw.point_groups.groups as groups
        self.name = name
        group = getattr(groups, name)()
        self.character_table = np.array(group.character_table)
        self.operations: Dict[str, Array2D] = {}
        for opname, op in group.operations:
            assert opname not in self.operations, opname
            if not isinstance(op, np.ndarray):
                op = op(np.eye(3))
            self.operations[opname] = op
        self.symmetries = group.symmetries
        self.nops = group.nof_operations
        self.complex = getattr(group, 'complex', False)
        self.translations = [self.symmetries[t]
                             for t in (group.Tx_i, group.Ty_i, group.Tz_i)]

    def __str__(self) -> str:
        lines = [[self.name] + list(self.operations)]
        for sym, coefs in zip(self.symmetries, self.character_table):
            lines.append([sym] + list(coefs))
        return '\n'.join(f'{line[0]:5}' +
                         ''.join(f'{word:>10}' for word in line[1:])
                         for line in lines) + '\n'

    def get_normalized_table(self) -> Array2D:
        """Normalized character table."""
        # Divide by degeneracies:
        return self.character_table / self.character_table[:, :1]
