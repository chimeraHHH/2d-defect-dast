from ase.build import mx2, graphene
from ase.db import connect
from ase import Atom
import numpy as np

db = connect('/root/autodl-tmp/2d-defect-dast/imp2d_filtered.db')

def add_adsorbate(atoms, dopant, z_offset=2.0):
    pos = atoms.positions[0].copy()
    pos[2] = max(atoms.positions[:, 2]) + z_offset
    atoms.append(Atom(dopant, position=pos))
    atoms.pbc = [True, True, False]
    return atoms

def write_db(db_conn, atoms, dopant, host, name, supercell):
    f = atoms.get_chemical_formula()
    print(f"Generated {f} ({name})")
    db_conn.write(atoms, dopant=dopant, host=host, name=name, 
             defecttype='adsorbate', eform=0.0, en2=0.0, hostenergy=0.0, 
             dopant_chemical_potential=0.0, supercell=supercell)

# 1. WS2 + K adatom (4x4 supercell)
ws2 = mx2(formula='WS2', kind='2H', a=3.15, thickness=3.15, size=(4, 4, 1), vacuum=10.0)
ws2_k = add_adsorbate(ws2, 'K', 2.2)
write_db(db, ws2_k, 'K', 'WS2', 'WS2_K_ads_custom', 441)

# 2. Graphene + Na adatom (5x5 supercell)
gr_na = graphene(formula='C2', a=2.46, size=(5, 5, 1), vacuum=10.0)
gr_na = add_adsorbate(gr_na, 'Na', 1.8)
write_db(db, gr_na, 'Na', 'C', 'Graphene_Na_ads_custom', 551)

# 3. MoS2 + Mg adatom (4x4 supercell)
mos2_mg = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(4, 4, 1), vacuum=10.0)
mos2_mg = add_adsorbate(mos2_mg, 'Mg', 1.5)
write_db(db, mos2_mg, 'Mg', 'MoS2', 'MoS2_Mg_ads_custom', 441)

# 4. WSe2 + Li adatom (4x4 supercell)
wse2_li = mx2(formula='WSe2', kind='2H', a=3.28, thickness=3.34, size=(4, 4, 1), vacuum=10.0)
wse2_li = add_adsorbate(wse2_li, 'Li', 1.7)
write_db(db, wse2_li, 'Li', 'WSe2', 'WSe2_Li_ads_custom', 441)

# 5. Graphene + Ca adatom (5x5 supercell)
gr_ca = graphene(formula='C2', a=2.46, size=(5, 5, 1), vacuum=10.0)
gr_ca = add_adsorbate(gr_ca, 'Ca', 2.0)
write_db(db, gr_ca, 'Ca', 'C', 'Graphene_Ca_ads_custom', 551)

print("Insertion of 5 new samples complete.")
