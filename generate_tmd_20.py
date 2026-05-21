import os
import numpy as np
from ase.build import mx2
from ase.db import connect
from ase import Atom

db_path = '/root/autodl-tmp/2d-defect-dast/new_tmd_samples_20.db'
if os.path.exists(db_path):
    os.remove(db_path)
db = connect(db_path)

hosts_info = {
    'VSe2': {'a': 3.35, 'thickness': 3.3, 'dopants': ['Li', 'Na', 'Mg', 'O', 'F', 'Cl', 'N', 'P', 'Co', 'Cu']},
    'CrSe2': {'a': 3.12, 'thickness': 3.2, 'dopants': ['K', 'Ca', 'Br', 'I', 'As', 'C', 'Si', 'Fe', 'Ni', 'Ag']}
}

formulas = []
for host, info in hosts_info.items():
    pristine = mx2(formula=host, kind='2H', a=info['a'], thickness=info['thickness'], size=(4, 4, 1), vacuum=10.0)
    tm_idx = 0
    tm_pos = pristine.positions[tm_idx]
    
    for dopant in info['dopants']:
        defect = pristine.copy()
        z_max = np.max(defect.positions[:, 2])
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
