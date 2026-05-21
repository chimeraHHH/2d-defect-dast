import os
from ase.db import connect
from gpaw import GPAW, PW
from ase.optimize import BFGS  # 导入结构优化器

def main():
    # ⚠️ 请务必去终端核对文件名，确认 imp2d 和 .db 之间到底有没有空格！
    db_path = "/root/autodl-tmp/2d-defect-dast/imp2d.db"  
    
    print(f"Connecting to database: {db_path}")
    if not os.path.exists(db_path):
        print(f"Error: Database file does not exist at {db_path}.")
        return
        
    try:
        with connect(db_path) as db:
            # 按 natoms (原子数) 排序，选出最简单的 5 个结构
            rows = list(db.select(limit=5, sort='natoms'))
    except Exception as e:
        print(f"Error reading database: {e}")
        return
        
    print(f"Successfully selected {len(rows)} simple structures.\n")

    # 2. 定义参数 gpaw_kwargs
    gpaw_kwargs = {
        'mode': PW(600),
        'xc': 'PBE',
        'kpts': (1, 1, 1),
        'spinpol': True,
        'txt': 'gpaw_run.txt'
    }

    # 3. 利用 GPAW 对这些结构进行 DFT 计算
    for i, row in enumerate(rows):
        atoms = row.toatoms()
        formula = atoms.get_chemical_formula()
        
        # 尝试提取磁矩
        magmoms = None
        if hasattr(row, 'magmoms') and row.magmoms is not None:
            magmoms = row.magmoms
        elif row.data and 'magmoms' in row.data:
            magmoms = row.data['magmoms']
        elif 'initial_magmoms' in atoms.arrays:
            magmoms = atoms.get_initial_magnetic_moments()
            
        # ⚠️ 【核心改动】只有当磁矩数组的长度与当前原子的个数完美相等时，才进行设置
        if magmoms is not None and len(magmoms) == len(atoms):
            print(f"Sample {i+1} (ID: {row.id}, {formula}): Found valid magnetic moments, setting as initial guess.")
            atoms.set_initial_magnetic_moments(magmoms)
        else:
            print(f"Sample {i+1} (ID: {row.id}, {formula}): No magnetic moments found (or length mismatched). Using default spin-polarization.")
            
        # 为避免覆盖日志，将当前计算的 txt 文件名动态修改为包含 row.id
        current_kwargs = gpaw_kwargs.copy()
        current_kwargs['txt'] = f'gpaw_run_sample_{row.id}.txt'
        
        calc = GPAW(**current_kwargs)
        atoms.calc = calc
        
        print(f"-> Running GPAW calculation for Sample {i+1} ...")
        try:
            # 🌟【建议改动】文献里是进行结构弛豫的，这里建议用 BFGS 优化一下
            # 如果你只想测试单点能速度，可以把下面两行 relax 注释掉，直接执行 get_potential_energy
            relax = BFGS(atoms, trajectory=f'relax_sample_{row.id}.traj')
            relax.run(fmax=0.05, steps=40) # 限制最大 40 步，力收敛标准 0.05 eV/A
            
            # 提取 GPAW 算出的最终总能量
            energy = atoms.get_potential_energy()
            
            # 从数据库中提取文献当年用 VASP 算出来的总能量做对比
            vasp_energy = row.energy
            energy_diff = abs(energy - vasp_energy)
            
            print(f"-> Sample {i+1} GPAW Energy: {energy:.4f} eV")
            print(f"-> Sample {i+1} VASP Energy (DB): {vasp_energy:.4f} eV")
            print(f"-> Absolute Difference: {energy_diff:.4f} eV\n")
            
        except Exception as e:
            print(f"-> Error calculating Sample {i+1}: {e}\n")

if __name__ == "__main__":
    main()