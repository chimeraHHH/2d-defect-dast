import os
os.environ['GPAW_NEW'] = '1'
os.environ['GPAW_USE_GPUS'] = '1'

from ase.db import connect
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac
import time

E_CUTOFF = 600
F_CONV_1 = 5e-3
F_CONV_2 = 1e-4
MAX_STEPS = 20
XC_FUNC = 'PBE'
GPU_CONFIG = {'gpu': True}
DB_PATH = '/root/autodl-tmp/2d-defect-dast/imp2d_filtered.db'

def get_calculator(txt, elec_tol=1e-6):
    return GPAW(
        mode=PW(E_CUTOFF),
        xc=XC_FUNC,
        kpts=(1, 1, 1),
        spinpol=True,
        occupations=FermiDirac(0.05),
        convergence={'energy': elec_tol},
        parallel=GPU_CONFIG,
        txt=txt
    )

def two_stage_relax(atoms, name):
    print(f"\n--- 开始计算: {name} ---")
    
    # Stage 1
    atoms.calc = get_calculator(f'gpaw_{name}_stage1.txt', elec_tol=1e-4)
    dyn1 = BFGS(atoms, trajectory=f"{name}_s1.traj", logfile=f"{name}_s1.log")
    dyn1.run(fmax=F_CONV_1, steps=MAX_STEPS)
    
    # Stage 2
    atoms.calc = get_calculator(f'gpaw_{name}_stage2.txt', elec_tol=1e-6)
    dyn2 = BFGS(atoms, trajectory=f"{name}_s2.traj", logfile=f"{name}_s2.log")
    dyn2.run(fmax=F_CONV_2, steps=MAX_STEPS)
    
    return atoms.get_potential_energy()

def main():
    db = connect(DB_PATH)
    
    # 抽取第一个符合条件的样本
    row = None
    for r in db.select(limit=100):
        atoms = r.toatoms()
        lengths = atoms.cell.lengths()
        if all(l >= 10.0 for l in lengths[:2]):
            row = r
            break
            
    if row is None:
        print("未找到符合条件的样本")
        return

    print(f"================ 样本: {row.formula} (Dopant: {row.dopant}) ================")
    
    defect_atoms = row.toatoms()
    
    host_atoms = defect_atoms.copy()
    dopant_indices = [atom.index for atom in host_atoms if atom.symbol == row.dopant]
    if dopant_indices:
        del host_atoms[dopant_indices[0]]

    # 运行 DFT 计算
    start_time = time.time()
    
    en2 = two_stage_relax(defect_atoms, "sample_0_defect_test")
    hostenergy = two_stage_relax(host_atoms, "sample_0_host_test")
    
    print("\n" + "="*50)
    print("对比结果 (缺陷超胞能量 - 本征超胞能量):")
    print(f"GPAW: en2 - hostenergy = {en2 - hostenergy:.4f} eV")
    print(f"VASP: en2 - hostenergy = {row.en2 - row.hostenergy:.4f} eV")
    print(f"总耗时: {(time.time() - start_time) / 60:.1f} 分钟")
    print("="*50)

if __name__ == "__main__":
    main()
