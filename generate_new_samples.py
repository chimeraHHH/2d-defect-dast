from ase.build import graphene
from ase.db import connect
import numpy as np
import os

if os.path.exists('/root/autodl-tmp/2d-defect-dast/new_samples.db'):
    os.remove('/root/autodl-tmp/2d-defect-dast/new_samples.db')

prim = graphene()
prim.cell[2, 2] = 20.0
prim.pbc = [True, True, False]
prim.center() # This centers the atoms along the non-periodic z-axis!

supercell = prim.repeat((5, 5, 1))

dopants = ['Li', 'Na', 'K', 'Mg', 'Ca', 'F', 'Cl', 'Br', 'I', 'O']

db = connect('/root/autodl-tmp/2d-defect-dast/new_samples.db')

formulas = []
for dopant in dopants:
    atoms = supercell.copy()
    center_x = np.mean(atoms.positions[:, 0])
    center_y = np.mean(atoms.positions[:, 1])
    z_max = np.max(atoms.positions[:, 2])
    
    from ase import Atom
    atoms.append(Atom(dopant, (center_x, center_y, z_max + 1.5))) # 1.5A above graphene plane
    
    kvp = {
        'host': 'C2',
        'dopant': dopant,
        'en2': 0.0,
        'hostenergy': 0.0,
        'dopant_chemical_potential': 0.0,
        'eform': 0.0
    }
    db.write(atoms, key_value_pairs=kvp)
    formulas.append(atoms.get_chemical_formula())

print("Created 10 new samples in new_samples.db")
print("Formulas:", formulas)
