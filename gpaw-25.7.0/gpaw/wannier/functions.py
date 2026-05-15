from ase import Atoms

from gpaw.calculator import GPAW
from gpaw.typing import Array3D


class WannierFunctions:
    def __init__(self,
                 atoms: Atoms,
                 centers,
                 value,
                 U_knn,
                 n1=0,
                 spin=0):
        self.atoms = atoms
        self.centers = centers
        self.U_knn = U_knn
        self.value = value
        self.n1 = 0
        self.spin = spin

    def centers_as_atoms(self):
        return self.atoms + Atoms(f'X{len(self.centers)}', self.centers)

    def get_function(self, calc: GPAW, n: int) -> Array3D:
        wf = calc.wfs.gd.zeros()
        for m, u in enumerate(self.U_knn[0][:, n]):
            wf += u * calc.wfs.get_wave_function_array(n=self.n1 + m,
                                                       s=self.spin,
                                                       k=0)
        return wf
