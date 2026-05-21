import os
from ase.build import mx2, graphene
from ase.db import connect
from ase import Atom
import numpy as np

db_filtered = connect('/root/autodl-tmp/2d-defect-dast/imp2d_filtered.db')

new_samples = []

# 1. MoS2 + K adatom (4x4 supercell)
mos2_k = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(4, 4, 1), vacuum=10.0)
pos_k = mos2_k.positions[0].copy()
pos_k[2] = max(mos2_k.positions[:, 2]) + 2.2
mos2_k.append(Atom('K', position=pos_k))
mos2_k.pbc = [True, True, False]
f1 = mos2_k.get_chemical_formula()
new_samples.append((mos2_k, f1, 'K', 'MoS2'))

# 2. Graphene + Na adatom (5x5 supercell)
gr_na = graphene(formula='C2', a=2.46, size=(5, 5, 1), vacuum=10.0)
pos_na = gr_na.positions[0].copy()
pos_na[2] = max(gr_na.positions[:, 2]) + 2.0
gr_na.append(Atom('Na', position=pos_na))
gr_na.pbc = [True, True, False]
f2 = gr_na.get_chemical_formula()
new_samples.append((gr_na, f2, 'Na', 'C'))

# 3. WS2 + Rb adatom (4x4 supercell)
ws2_rb = mx2(formula='WS2', kind='2H', a=3.18, thickness=3.19, size=(4, 4, 1), vacuum=10.0)
pos_rb = ws2_rb.positions[0].copy()
pos_rb[2] = max(ws2_rb.positions[:, 2]) + 2.4
ws2_rb.append(Atom('Rb', position=pos_rb))
ws2_rb.pbc = [True, True, False]
f3 = ws2_rb.get_chemical_formula()
new_samples.append((ws2_rb, f3, 'Rb', 'WS2'))

# 4. Graphene + F adatom (5x5 supercell)
gr_f = graphene(formula='C2', a=2.46, size=(5, 5, 1), vacuum=10.0)
pos_f = gr_f.positions[0].copy()
pos_f[2] = max(gr_f.positions[:, 2]) + 1.5
gr_f.append(Atom('F', position=pos_f))
gr_f.pbc = [True, True, False]
f4 = gr_f.get_chemical_formula()
new_samples.append((gr_f, f4, 'F', 'C'))

# 5. MoSe2 + Cs adatom (4x4 supercell)
mose2_cs = mx2(formula='MoSe2', kind='2H', a=3.32, thickness=3.34, size=(4, 4, 1), vacuum=10.0)
pos_cs = mose2_cs.positions[0].copy()
pos_cs[2] = max(mose2_cs.positions[:, 2]) + 2.6
mose2_cs.append(Atom('Cs', position=pos_cs))
mose2_cs.pbc = [True, True, False]
f5 = mose2_cs.get_chemical_formula()
new_samples.append((mose2_cs, f5, 'Cs', 'MoSe2'))

formulas = []
for atoms, formula, dopant, host in new_samples:
    name = f"{host}_{dopant}_ads_custom_v2"
    print(f"Generated {formula}")
    formulas.append(formula)
    db_filtered.write(atoms, dopant=dopant, host=host, name=name, 
                      defecttype='adsorbate', eform=0.0, en2=0.0, hostenergy=0.0, 
                      dopant_chemical_potential=0.0, supercell=100)

print("Insertion complete.")
print("Target formulas:", formulas)
