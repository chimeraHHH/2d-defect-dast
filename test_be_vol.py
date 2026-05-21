import os
os.environ['GPAW_NEW'] = '1'
# 暂时禁用 GPU 以避免 NVRTC 编译报错（可能是由于 cupy 编译缓存或环境问题）
os.environ['GPAW_USE_GPUS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

from ase.build import bulk
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac

ref = bulk('Be', crystalstructure='hcp', a=2.285, c=3.584)
print(f"初始原子数 = {len(ref)}")
print(f"初始体积 = {ref.get_volume():.3f} Å³")

calc = GPAW(
    mode=PW(600),
    xc='PBE',
    kpts=(1, 1, 1),
    spinpol=True,
    occupations=FermiDirac(0.05),
    convergence={'energy': 1e-4},
    parallel={'gpu': False},
    txt='test_be_vol.txt'
)
ref.calc = calc
ucf = UnitCellFilter(ref, scalar_pressure=0.0)
dyn = BFGS(ucf, logfile='test_be_vol.log')
dyn.run(fmax=0.05, steps=100)

print(f"体积弛豫后晶格参数: {ref.cell.lengths()}")
print(f"体积弛豫后体积: {ref.get_volume():.3f} Å³")
print(f"每原子体积: {ref.get_volume()/len(ref):.3f} Å³")
