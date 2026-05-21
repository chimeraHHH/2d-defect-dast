import os
os.environ['GPAW_NEW'] = '1'
os.environ['GPAW_USE_GPUS'] = '0' # 强制使用 CPU，避免 cupy 报错

import numpy as np
from ase.db import connect
from ase.build import bulk
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac

DB_PATH = '/root/autodl-tmp/2d-defect-dast/imp2d_filtered.db'
E_CUTOFF = 600
XC_FUNC = 'PBE'

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

def get_dopant_structure(dopant_symbol):
    if dopant_symbol in SPECIAL_BULK:
        kwargs = SPECIAL_BULK[dopant_symbol]
        ref = bulk(dopant_symbol, **kwargs)
        return ref, len(ref)
    else:
        try:
            ref = bulk(dopant_symbol)
        except Exception:
            ref = bulk(dopant_symbol, crystalstructure='sc', a=3.0)
        return ref, len(ref)

def calc_mu(element):
    ref, natoms = get_dopant_structure(element)
    
    # 针对部分磁性金属赋予初始磁矩，防止收敛到非磁性态
    mag_elements = ['Fe', 'Co', 'Ni', 'Cr', 'Mn']
    if element in mag_elements:
        ref.set_initial_magnetic_moments([2.0] * natoms)
        
    mu_kpts = (12, 12, 12)
    print(f"\n[{element}] 初始晶格参数: {ref.cell.lengths()}")
    print(f"[{element}] 初始体积: {ref.get_volume():.3f} Å³")
    
    # 1. 体积弛豫
    calc_vol = GPAW(
        mode=PW(E_CUTOFF),
        xc=XC_FUNC,
        kpts=mu_kpts,
        spinpol=True,
        occupations=FermiDirac(0.05),
        convergence={'energy': 1e-4},
        parallel={'gpu': False},
        txt=f'test_mu_{element}_vol.txt'
    )
    ref.calc = calc_vol
    ucf = UnitCellFilter(ref, scalar_pressure=0.0)
    dyn_vol = BFGS(ucf, logfile=f'test_mu_{element}_vol.log')
    dyn_vol.run(fmax=0.05, steps=50)
    
    print(f"[{element}] 体积弛豫后晶格参数: {ref.cell.lengths()}")
    print(f"[{element}] 体积弛豫后体积: {ref.get_volume():.3f} Å³")

    # 2. 离子弛豫 (精细)
    calc_relax = GPAW(
        mode=PW(E_CUTOFF),
        xc=XC_FUNC,
        kpts=mu_kpts,
        spinpol=True,
        occupations=FermiDirac(0.05),
        convergence={'energy': 1e-6},
        parallel={'gpu': False},
        txt=f'test_mu_{element}_relax.txt'
    )
    ref.calc = calc_relax
    dyn_relax = BFGS(ref, logfile=f'test_mu_{element}_relax.log')
    dyn_relax.run(fmax=1e-4, steps=50)
    
    energy = ref.get_potential_energy()
    return energy / natoms

def main():
    db = connect(DB_PATH)
    target_dopants = ['Be', 'Ru', 'Zn']
    
    # 提取数据库参考值
    ref_mus = {}
    for row in db.select():
        if row.dopant in target_dopants and row.dopant not in ref_mus:
            ref_mus[row.dopant] = row.dopant_chemical_potential
        if len(ref_mus) == len(target_dopants):
            break
            
    results = []
    for dopant in target_dopants:
        mu_gpaw = calc_mu(dopant)
        mu_vasp = ref_mus[dopant]
        results.append((dopant, mu_gpaw, mu_vasp, mu_gpaw - mu_vasp))
        
    print("\n\n" + "="*70)
    print(f"{'元素':<10} | {'GPAW (eV)':<15} | {'VASP (eV)':<15} | {'Diff (eV)':<15}")
    print("-" * 70)
    for dopant, mu_g, mu_v, diff in results:
        print(f"{dopant:<10} | {mu_g:<15.4f} | {mu_v:<15.4f} | {diff:<15.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
