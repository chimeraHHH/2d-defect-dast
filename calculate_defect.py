import os
# 强制在 Python 代码的最开头注入环境变量，确保 GPAW 初始化时能读取到
os.environ['GPAW_NEW'] = '1'
os.environ['GPAW_USE_GPUS'] = '1'
# 禁用 OpenMP 多线程，因为我们在用 GPU 并行，CPU 多线程反而会产生冲突和警告
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from ase.build import mx2, bulk, molecule, graphene
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac

# ==============================================================================
# 核心配置参数
# ==============================================================================
E_CUTOFF = 500                # 平面波截断能 (eV)
F_CONV = 0.02                 # 几何优化收敛标准 (eV/A)
XC_FUNC = 'PBE'               # 交换关联泛函
GPU_CONFIG = {'gpu': True}    # GPU 加速配置

def get_calculator(txt, kpts=(1, 1, 1), spinpol=False):
    """创建优化的GPAW计算器"""
    return GPAW(
        mode=PW(E_CUTOFF),
        xc=XC_FUNC,
        kpts=kpts,
        spinpol=spinpol,           # 开启/关闭自旋极化
        occupations=FermiDirac(0.05),
        parallel=GPU_CONFIG,
        txt=txt
    )

def relax(atoms, name):
    """几何优化通用函数"""
    traj = f"{name}.traj"
    log = f"{name}.log"
    dyn = BFGS(atoms, trajectory=traj, logfile=log)
    dyn.run(fmax=F_CONV)
    return atoms.get_potential_energy()

def main():
    # ==============================================================================
    # 1. 适配模型结构：构建完美的石墨烯 (Graphene) 5x5 超胞
    # ==============================================================================
    print("构建完美的 Graphene 5x5 超胞以适配模型 dmax_global (12.0 Å)...")
    # 石墨烯的晶格常数 a 约为 2.46 Å
    pristine = graphene(formula='C2', a=2.46, size=(5, 5, 1), vacuum=15.0)
    
    print("正在弛豫完美石墨烯晶体...")
    pristine.calc = get_calculator('gpaw_pristine.txt', kpts=(5, 5, 1), spinpol=False)
    try:
        E_pristine = relax(pristine, 'pristine')
        print(f"完美石墨烯晶体弛豫后能量: {E_pristine:.3f} eV")
    except Exception as e:
        print(f"计算失败，请检查环境或GPU设置: {e}")
        return

    # ==============================================================================
    # 2. 构建缺陷结构并进行结构弛豫：
    # 制造单氮取代掺杂 (N_C)
    # ==============================================================================
    print("构建缺陷：N 单原子取代掺杂 (N_C)...")
    defect = pristine.copy()
    
    # 找到靠近超胞中心的 C 原子
    center = defect.cell @ np.array([0.5, 0.5, 0.5])
    distances = np.linalg.norm(defect.positions - center, axis=1)
    
    c_indices = [atom.index for atom in defect if atom.symbol == 'C']
    c_distances = [(idx, distances[idx]) for idx in c_indices]
    c_distances.sort(key=lambda x: x[1])
    target_c_idx = c_distances[0][0]
    
    # 将该 C 原子替换为 N 原子
    defect[target_c_idx].symbol = 'N'
    
    print("开始对缺陷结构进行全自由度几何优化 (弛豫)... 这个过程可能需要几个小时。")
    # 氮掺杂可能引入不成对电子（自旋极化），因此开启 spinpol=True
    defect.calc = get_calculator('gpaw_defect.txt', kpts=(5, 5, 1), spinpol=True)
    E_defect = relax(defect, 'defect')
    print(f"缺陷石墨烯晶体弛豫后能量: {E_defect:.3f} eV")

    # ==============================================================================
    # 3. 计算参考化学势
    # 石墨烯的化学势: 我们可以直接用完美石墨烯的平均单原子能量来近似 mu_C
    # 氮原子的化学势: 通常使用 N2 气体分子的一半作为 mu_N
    # ==============================================================================
    print("计算参考化学势 (孤立 N2 分子单点能及平均单 C 原子能量)...")
    
    # C 的化学势：直接用之前计算的 5x5 石墨烯超胞总能量除以原子数
    mu_C = E_pristine / len(pristine)
    print(f"  -> C 化学势 (从完美石墨烯提取): {mu_C:.3f} eV")
    
    # N 的化学势：计算一个 N2 分子放在大盒子里的能量
    n2_mol = molecule('N2')
    n2_mol.set_cell([15.0, 15.0, 15.0])
    n2_mol.center()
    n2_mol.calc = get_calculator('gpaw_N2_mol.txt', kpts=(1, 1, 1), spinpol=False)
    
    print("  -> 正在弛豫 N2 分子...")
    E_N2 = relax(n2_mol, 'ref_N2_mol')
    mu_N = E_N2 / 2.0
    print(f"  -> N 化学势 (N2分子的一半): {mu_N:.3f} eV")

    # ==============================================================================
    # 4. 计算形成能并输出
    # 形成能公式: E_form = E_defect - E_pristine + mu_C - mu_N
    # 系统失去了一个 C 原子（加上 mu_C），得到一个 N 原子（减去 mu_N）
    # ==============================================================================
    E_form = E_defect - E_pristine + mu_C - mu_N
    print("-" * 50)
    print(f"弛豫后 N 掺杂石墨烯缺陷 (N_C) 形成能: {E_form:.3f} eV")
    print("-" * 50)

if __name__ == "__main__":
    main()
