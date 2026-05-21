from ase.db import connect
import numpy as np
db = connect('/root/autodl-tmp/2d-defect-dast/imp2d_filtered.db')
row = next(db.select(formula='Mo9Te19'))
atoms = row.toatoms()

# find a hetero-doped structure
hetero_row = None
for r in db.select(host=row.host, supercell=row.supercell):
    if r.dopant not in r.host:
        hetero_row = r
        break

h_atoms = hetero_row.toatoms()
d_idx = [a.index for a in h_atoms if a.symbol == hetero_row.dopant][0]
del h_atoms[d_idx]

# Map atoms to h_atoms
dopant_indices = [a.index for a in atoms if a.symbol == row.dopant]
min_dists = []
for idx in dopant_indices:
    pos = atoms[idx].position
    # find closest Te in h_atoms
    dists = [np.linalg.norm(pos - a.position) for a in h_atoms if a.symbol == row.dopant]
    min_dists.append(min(dists))

print('Distances to nearest host atom for each candidate:')
for idx, d in zip(dopant_indices, min_dists):
    print(f'Atom {idx}: {d:.3f}')
