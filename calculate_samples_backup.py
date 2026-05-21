import os
os.environ['GPAW_NEW'] = '1'
os.environ['GPAW_USE_GPUS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from ase.db import connect
from ase.build import bulk, molecule
from ase import Atoms
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from gpaw import GPAW, PW, FermiDirac
from gpaw import restart

# ==============================================================================
# 核心配置参数
# ==============================================================================
E_CUTOFF = 600                # 平面波截断能 (eV)
F_CONV_1 = 5e-3               # 几何优化第一阶段收敛标准 (eV/A)
F_CONV_2 = 1e-4               # 几何优化第二阶段收敛标准 (eV/A)
MAX_STEPS = 20                # 每阶段最大步数
XC_FUNC = 'PBE'               # 交换关联泛函
GPU_CONFIG = {'gpu': True}    # GPU 加速配置
DB_PATH = '/root/autodl-tmp/2d-defect-dast/imp2d_filtered.db'

def get_calculator(txt, elec_tol=1e-6, pbc=True, kpts=(1, 1, 1)):
    """创建优化的GPAW计算器"""
    if pbc:
        return GPAW(
            mode=PW(E_CUTOFF),
            xc=XC_FUNC,
            kpts=kpts,
            spinpol=True,             # 开启自旋极化
            occupations=FermiDirac(0.05),
            convergence={'energy': elec_tol},  # 电子收敛
            parallel=GPU_CONFIG,
            txt=txt
        )
    else:
        # 分子用实空间模式 (Finite Difference)
        # 注意：由于 GPAW GPU 模式下 fd 的 Poisson solver 与 scipy.fft 存在 cupy 数组兼容性问题，这里强制使用 CPU 计算
        return GPAW(
            mode='fd',
            xc=XC_FUNC,
            spinpol=True,
            convergence={'energy': elec_tol},
            parallel={'gpu': False},
            txt=txt
        )

def two_stage_relax(atoms, name, kpts=(1, 1, 1)):
    """两阶段几何优化"""
    pbc_flag = any(atoms.pbc)
    # 第一阶段
    atoms.calc = get_calculator(f'gpaw_{name}_stage1.txt', elec_tol=1e-4, pbc=pbc_flag, kpts=kpts)
    traj1 = f"{name}_s1.traj"
    log1 = f"{name}_s1.log"
    dyn1 = BFGS(atoms, trajectory=traj1, logfile=log1)
    
    print(f"[{name}] 开始第一阶段弛豫 (fmax={F_CONV_1}, max_steps={MAX_STEPS})...")
    dyn1.run(fmax=F_CONV_1, steps=MAX_STEPS)
    
    # 第二阶段：重新创建 calculator 和 optimizer
    atoms.calc = get_calculator(f'gpaw_{name}_stage2.txt', elec_tol=1e-6, pbc=pbc_flag, kpts=kpts)
    traj2 = f"{name}_s2.traj"
    log2 = f"{name}_s2.log"
    dyn2 = BFGS(atoms, trajectory=traj2, logfile=log2)
    
    print(f"[{name}] 开始第二阶段弛豫 (fmax={F_CONV_2}, max_steps={MAX_STEPS})...")
    dyn2.run(fmax=F_CONV_2, steps=MAX_STEPS)
    
    return atoms.get_potential_energy()

def get_dopant_structure(dopant_symbol):
    """获取掺杂元素的参考结构及原子数"""
    # 标准态为气体分子的元素
    gases = {
        'H': 'H2', 'N': 'N2', 'O': 'O2',
        'F': 'F2', 'Cl': 'Cl2', 'Br': 'Br2', 'I': 'I2'
    }
    
    # 特殊固体元素的准确晶体结构参数
    # 这些是形成能计算时使用的标准态参考相
    # 注意：ASE bulk() 不直接支持 'hexagonal' 这个名称，通常用 'hcp' 或需要特定的空间群参数
    # 我们这里对部分元素做修正，用 ASE 内置支持的常见结构名称（如 'hcp' 代替 'hexagonal'，或不填让其自动处理只指定参数）
    SPECIAL_BULK = {
        'Te': {'crystalstructure': 'hcp', 'a': 4.457, 'c': 5.927},
        'Se': {'crystalstructure': 'hcp', 'a': 4.366, 'c': 4.954},
        'As': {'crystalstructure': 'rhombohedral', 'a': 3.760, 'c': 10.548},
        'Hg': {'crystalstructure': 'rhombohedral', 'a': 2.993, 'alpha': 70.52},
        'Ru': {'crystalstructure': 'hcp', 'a': 2.706, 'c': 4.282},
        'Zn': {'crystalstructure': 'hcp', 'a': 2.664, 'c': 4.947},
        'Co': {'crystalstructure': 'hcp', 'a': 2.507, 'c': 4.069},
        'Be': {'crystalstructure': 'hcp', 'a': 2.285, 'c': 3.584},
    }
    
    if dopant_symbol in gases:
        try:
            mol = molecule(gases[dopant_symbol])
        except KeyError:
            # ASE 的 G2 数据库中可能没有部分重卤素分子（如 I2, Br2），手动构建初始双原子结构
            bond_lengths = {'Br': 2.28, 'I': 2.67}
            d = bond_lengths.get(dopant_symbol, 2.0)
            mol = Atoms(gases[dopant_symbol], positions=[[0, 0, 0], [0, 0, d]])
            
        mol.set_cell([15.0, 15.0, 15.0])
        mol.center()
        mol.pbc = False
        return mol, 2
    
    elif dopant_symbol in SPECIAL_BULK:
        kwargs = SPECIAL_BULK[dopant_symbol]
        ref = bulk(dopant_symbol, **kwargs)
        return ref, len(ref)
        
    else:
        try:
            ref = bulk(dopant_symbol)
        except Exception:
            ref = bulk(dopant_symbol, crystalstructure='sc', a=3.0)
        return ref, len(ref)

def get_available_elements(setups_dir):
    """从 gpaw-setups 目录获取有 PBE 赝势的元素集合"""
    if not os.path.exists(setups_dir):
        return set()
    elements = set()
    for f in os.listdir(setups_dir):
        if f.endswith('.PBE.gz'):
            elem = f.split('.')[0]
            elements.add(elem)
    return elements

_mu_cache = {}

def calc_mu(element, name_prefix):
    """计算掺杂元素的化学势，包含体积弛豫"""
    if element in _mu_cache:
        print(f"[{element}] 使用缓存的化学势 = {_mu_cache[element]:.4f} eV")
        return _mu_cache[element]
        
    ref, natoms = get_dopant_structure(element)
    pbc_flag = any(ref.pbc)
    
    # 金属块体等小原胞在计算化学势时，必须用更密的 K 点（如 12x12x12），
    # 并且对于 GPAW，必须明确传入整数元组或列表来定义 K 点网格。
    mu_kpts = (12, 12, 12) if pbc_flag else None
    
    # 针对部分磁性金属赋予初始磁矩，防止 SCF 陷入非磁性局部极小值
    mag_elements = ['Fe', 'Co', 'Ni', 'Cr', 'Mn']
    if element in mag_elements:
        ref.set_initial_magnetic_moments([2.0] * natoms)
    
    if pbc_flag:
        # 先做体积弛豫（让晶格常数收敛）
        print(f"[{name_prefix}] 开始体积弛豫 (fmax=0.05, steps=100)...")
        print(f"[{name_prefix}] 初始体积: {ref.get_volume():.3f} Å³ (初始原子数: {natoms})")
        ref.calc = get_calculator(f'gpaw_{name_prefix}_vol.txt', elec_tol=1e-6, pbc=True, kpts=mu_kpts)
        ucf = UnitCellFilter(ref, scalar_pressure=0.0)
        dyn = BFGS(ucf, trajectory=f"{name_prefix}_vol.traj", logfile=f"{name_prefix}_vol.log")
        dyn.run(fmax=0.05, steps=100)
        
        # 弛豫后检查晶格是否合理
        print(f"[{name_prefix}] 体积弛豫后晶格参数: {ref.cell.lengths()}")
        print(f"[{name_prefix}] 体积弛豫后体积: {ref.get_volume():.3f} Å³")
        print(f"[{name_prefix}] 每原子体积: {ref.get_volume()/natoms:.3f} Å³")
        
    # 再做原子位置弛豫
    energy = two_stage_relax(ref, f"mu_{element}", kpts=mu_kpts)
    mu = energy / natoms
    _mu_cache[element] = mu
    return mu

def main():
    db = connect(DB_PATH)
    
    # 获取支持的元素
    setups_dir = '/root/autodl-tmp/2d-defect-dast/gpaw-setups'
    available_elements = get_available_elements(setups_dir)
    print(f"检测到支持的赝势元素数量: {len(available_elements)}")
    
    # 抽取特定测试样本
    target_formulas = [
        'Mo9AsTe18', 'RuSn9S18', 'HgW9Te18', 'CoMo9Te18',
        'Mo16ZnS32', 'Mo16RuSe32', 'BeW9Te18', 'Mo9Te19'
    ]
    samples = []
    found_formulas = set()
    for row in db.select():
        if row.formula in target_formulas and row.formula not in found_formulas:
            atoms = row.toatoms()
            # 检查是否所有元素都有对应的 PBE 赝势
            if available_elements and not set(atoms.symbols).issubset(available_elements):
                continue
            
            # 检查掺杂元素是否有赝势
            if available_elements and row.dopant not in available_elements:
                continue
                
            samples.append(row)
            found_formulas.add(row.formula)
            if len(samples) == len(target_formulas):
                break

    print(f"成功抽取了 {len(samples)} 个样本进行计算。")
    
    # 用于汇总所有结果的列表
    summary_results = []

    for i, row in enumerate(samples):
        print(f"\n================ 样本 {i+1}/{len(samples)}: {row.formula} (Dopant: {row.dopant}) ================")
        
        # 1. 缺陷超胞结构 (直接来自数据库)
        defect_atoms = row.toatoms()
        
        # 2. 构建本征超胞 (从缺陷结构中移除掺杂原子)
        # 注意: 对于 interstitial 缺陷，直接移除 dopant 即为本征
        # 对于 adsorbate 同样适用。如果是 substitution，需要将 dopant 替换回 host 原子，这里假设以 interstitial/adsorbate 为主
        host_atoms = defect_atoms.copy()
        
        # 为了应对同种元素自掺杂（如 MoTe2 中掺杂 Te，即 Te_interstitial/adsorbate）
        # 此时 defect_atoms 中包含多个 Te 原子。不能随便删除一个，必须删除那个作为掺杂物的原子。
        # 可以通过原子位置与晶胞中心或其它几何特征来判断，但最准确的做法是从数据库重新获取本征超胞的初始结构，
        # 或者识别出“多余”的那个原子。
        # 简单启发式：在 adsorbate 或 interstitial 中，掺杂原子的 z 坐标往往最高或最低，
        # 或者距离其他同类原子有特定的距离分布。这里我们尝试通过原子间距寻找最近邻或几何中心偏差，
        # 但最可靠的方法是只删除一个原子：通常插入的原子位于特定的层外或间隙。
        dopant_indices = [atom.index for atom in host_atoms if atom.symbol == row.dopant]
        if dopant_indices:
            # 如果只有一个掺杂原子（异质掺杂），直接删除
            if len(dopant_indices) == 1:
                del host_atoms[dopant_indices[0]]
            else:
                # 发生自掺杂（如同质元素间隙/吸附）。我们需要识别出哪个是多出来的原子。
                # 假设吸附原子/间隙原子在 Z 轴方向上偏离主要层，或者它是结构中最不稳定的那个。
                # 由于原胞在弛豫前是标准的 2D 晶格，通常掺杂原子的 z 坐标最大（表面吸附）或处于特殊的层间。
                # 为了不写复杂的几何算法，我们直接通过寻找 Z 坐标绝对偏差最大的那个原子作为掺杂原子
                z_coords = [host_atoms[i].position[2] for i in dopant_indices]
                z_mean = sum(z_coords) / len(z_coords)
                z_diffs = [abs(z - z_mean) for z in z_coords]
                target_idx = dopant_indices[z_diffs.index(max(z_diffs))]
                del host_atoms[target_idx]
        
        # ====== 开始 DFT 计算 ======
        try:
            print("-> 1/3 计算缺陷超胞能量 (en2)...")
            en2 = two_stage_relax(defect_atoms, f"sample_{i}_defect")
            
            print("-> 2/3 计算本征超胞能量 (hostenergy)...")
            hostenergy = two_stage_relax(host_atoms, f"sample_{i}_host")
            
            print(f"-> 3/3 计算掺杂元素单质能量 (dopant: {row.dopant})...")
            dopant_chemical_potential = calc_mu(row.dopant, f"mu_{row.dopant}")
        except Exception as e:
            print(f"[{row.formula}] 计算过程中发生错误，跳过该样本。错误信息: {e}")
            continue
        
        # ====== 计算形成能 ======
        # eform ≈ en2 - hostenergy - dopant_chemical_potential
        eform = en2 - hostenergy - dopant_chemical_potential
        ref_eform = row.eform
        
        diff = eform - ref_eform
        abs_diff = abs(diff)
        rel_error = (abs_diff / abs(ref_eform)) * 100 if ref_eform != 0 else 0.0
        
        print("\n--- 计算结果汇总与比对 ---")
        print(f"{'项':<30} | {'GPAW 计算值 (eV)':<18} | {'VASP 数据库参考值 (eV)':<18} | {'差值 (Diff eV)':<15}")
        print("-" * 90)
        print(f"{'en2 (缺陷超胞能量)':<28} | {en2:<20.4f} | {row.en2:<22.4f} | {en2 - row.en2:<15.4f}")
        print(f"{'hostenergy (本征超胞能量)':<26} | {hostenergy:<20.4f} | {row.hostenergy:<22.4f} | {hostenergy - row.hostenergy:<15.4f}")
        print(f"{'mu_dopant (掺杂元素化学势)':<26} | {dopant_chemical_potential:<20.4f} | {row.dopant_chemical_potential:<22.4f} | {dopant_chemical_potential - row.dopant_chemical_potential:<15.4f}")
        print("-" * 90)
        print(f"{'eform (最终形成能)':<27} | {eform:<20.4f} | {ref_eform:<22.4f} | {diff:<15.4f}")
        
        print(f"\n相对误差 = {rel_error:.2f}%")
        
        # 将结果存入汇总列表
        summary_results.append({
            'id': row.id,
            'formula': row.formula,
            'dopant': row.dopant,
            'en2': en2,
            'hostenergy': hostenergy,
            'mu_dopant': dopant_chemical_potential,
            'calc_eform': eform,
            'ref_eform': ref_eform,
            'diff': diff,
            'rel_error': rel_error
        })
        
    # ====== 打印最终汇总表格 ======
    print("\n\n" + "="*80)
    print(f"{'最终计算结果汇总 (共 ' + str(len(summary_results)) + ' 个样本)':^70}")
    print("="*80)
    header = f"{'ID':<6}{'Formula':<12}{'Dopant':<8}{'en2':<12}{'host_E':<12}{'mu_dopant':<12}{'calc_Eform':<12}{'ref_Eform':<12}{'Diff(eV)':<12}{'Error(%)':<10}"
    print(header)
    print("-" * 105)
    
    for res in summary_results:
        row_str = (
            f"{res['id']:<6}"
            f"{res['formula']:<12}"
            f"{res['dopant']:<8}"
            f"{res['en2']:<12.4f}"
            f"{res['hostenergy']:<12.4f}"
            f"{res['mu_dopant']:<12.4f}"
            f"{res['calc_eform']:<12.4f}"
            f"{res['ref_eform']:<12.4f}"
            f"{res['diff']:<12.4f}"
            f"{res['rel_error']:<10.2f}"
        )
        print(row_str)
    print("="*105)

if __name__ == "__main__":
    main()
