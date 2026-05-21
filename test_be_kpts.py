import os
os.environ['GPAW_NEW'] = '1'
os.environ['GPAW_USE_GPUS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

from ase.build import bulk
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac

ref = bulk('Be', crystalstructure='hcp', a=2.285, c=3.584)
print(f"初始体积: {ref.get_volume():.3f}")

calc = GPAW(
    mode=PW(600),
    xc='PBE',
    kpts=(8, 8, 8),
    spinpol=True,
    occupations=FermiDirac(0.05),
    convergence={'energy': 1e-4},
    parallel={'gpu': False},
    txt='test_be_kpts.txt'
)
ref.calc = calc

ucf = UnitCellFilter(ref, scalar_pressure=0.0)
dyn = BFGS(ucf, logfile='test_be_kpts.log')
dyn.run(fmax=0.05, steps=100)

print(f"弛豫后体积: {ref.get_volume():.3f}")
print(f"每原子能量: {ref.get_potential_energy()/len(ref):.4f}")
