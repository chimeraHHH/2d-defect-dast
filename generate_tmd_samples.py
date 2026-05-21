import os
import numpy as np
from ase.build import mx2
from ase.db import connect
from ase import Atom

db_path = '/root/autodl-tmp/2d-defect-dast/new_tmd_samples.db'
if os.path.exists(db_path):
    os.remove(db_path)
db = connect(db_path)

# Parameters for VS2 and CrS2 (approximate experimental/DFT lattice constants)
# VS2: a ~ 3.22, CrS2: a ~ 3.00
hosts_info = {
    'VS2': {'a': 3.22, 'thickness': 3.2, 'dopants': ['Li', 'O', 'F', 'Co', 'Au']},
    'CrS2': {'a': 3.00, 'thickness': 3.1, 'dopants': ['Na', 'Cl', 'N', 'Ni', 'Ag']}
}

formulas = []
for host, info in hosts_info.items():
    # Generate pristine 4x4 supercell
    pristine = mx2(formula=host, kind='2H', a=info['a'], thickness=info['thickness'], size=(4, 4, 1), vacuum=10.0)
    
    # We will add an adatom above a transition metal atom (index 0 is a TM atom)
    tm_idx = 0
    tm_pos = pristine.positions[tm_idx]
    
    for dopant in info['dopants']:
        defect = pristine.copy()
        
        # Place adatom 2.0 Angstroms above the top S layer directly above the TM atom
        # TM is in the middle, S are above and below.
        # Let's find the max Z of the cell atoms to place it safely on the surface
        z_max = np.max(defect.positions[:, 2])
        
        # We place it at the xy coordinate of the TM atom, but at z_max + 1.8
        adatom = Atom(dopant, (tm_pos[0], tm_pos[1], z_max + 1.8))
        defect.append(adatom)
        
        kvp = {
            'host': host,
            'dopant': dopant,
            'defecttype': 'adatom',
            'en2': 0.0,
            'hostenergy': 0.0,
            'dopant_chemical_potential': 0.0,
            'eform': 0.0
        }
        db.write(defect, key_value_pairs=kvp)
        formulas.append(defect.get_chemical_formula())

print(f"Created {len(formulas)} new TMD samples in {db_path}")
print("Formulas:", formulas)
