from ase.db import connect
import numpy as np
db = connect('/root/autodl-tmp/2d-defect-dast/imp2d_filtered.db')

def get_pristine_host(row, db):
    defect_atoms = row.toatoms()
    if row.dopant not in row.host:
        dopant_idx = [a.index for a in defect_atoms if a.symbol == row.dopant][0]
        del defect_atoms[dopant_idx]
        return defect_atoms
    
    # 同质掺杂，寻找一个异质掺杂的骨架作为参考
    for r in db.select(host=row.host, supercell=row.supercell):
        if r.dopant not in r.host:
            h_atoms = r.toatoms()
            d_idx = [a.index for a in h_atoms if a.symbol == r.dopant][0]
            del h_atoms[d_idx]
            
            # 使用 h_atoms 匹配 defect_atoms，找到 defect_atoms 中多余的原子
            dopant_indices = [a.index for a in defect_atoms if a.symbol == row.dopant]
            max_dist = -1
            target_idx = -1
            for idx in dopant_indices:
                pos = defect_atoms[idx].position
                dists = [np.linalg.norm(pos - a.position) for a in h_atoms if a.symbol == row.dopant]
                min_d = min(dists)
                if min_d > max_dist:
                    max_dist = min_d
                    target_idx = idx
            
            del defect_atoms[target_idx]
            return defect_atoms

row = next(db.select(formula='Mo9Te19'))
host = get_pristine_host(row, db)
print(len(host))
