import os
os.environ['GPAW_NEW'] = '1'
os.environ['GPAW_USE_GPUS'] = '1'
from gpaw import GPAW
from ase.build import molecule
mol = molecule('H2')
mol.set_cell([10, 10, 10])
mol.center()
mol.pbc = False
calc = GPAW(mode='fd', xc='PBE', parallel={'gpu': False}, txt=None)
mol.calc = calc
mol.get_potential_energy()
print("Success")
