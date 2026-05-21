from ase.build import mx2, graphene
from ase.db import connect
from ase import Atom
import numpy as np

db = connect('/root/autodl-tmp/2d-defect-dast/imp2d_filtered.db')

# 1. MoS2 + Na adatom (4x4 supercell)
mos2 = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(4, 4, 1), vacuum=10.0)
na_pos = mos2.positions[0].copy() # top of Mo
na_pos[2] = max(mos2.positions[:, 2]) + 2.0
mos2.append(Atom('Na', position=na_pos))
mos2.pbc = [True, True, False]
print("Generated Mo16NaS32")

db.write(mos2, dopant='Na', host='MoS2', name='MoS2_Na_ads_custom', 
         defecttype='adsorbate', eform=0.0, en2=0.0, hostenergy=0.0, 
         dopant_chemical_potential=0.0, supercell=441)

# 2. Graphene + Li adatom (5x5 supercell)
gr = graphene(formula='C2', a=2.46, size=(5, 5, 1), vacuum=10.0)
li_pos = gr.positions[0].copy()
li_pos[2] = max(gr.positions[:, 2]) + 1.8
gr.append(Atom('Li', position=li_pos))
gr.pbc = [True, True, False]
print("Generated C50Li")

db.write(gr, dopant='Li', host='C', name='Graphene_Li_ads_custom', 
         defecttype='adsorbate', eform=0.0, en2=0.0, hostenergy=0.0, 
         dopant_chemical_potential=0.0, supercell=551)

print("Insertion complete.")
