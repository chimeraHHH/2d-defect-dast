import os
os.environ['GPAW_NEW'] = '1'
os.environ['GPAW_USE_GPUS'] = '0'

from ase.build import bulk
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac

ref = bulk('Ru', crystalstructure='hcp', a=2.706, c=4.282)

calc = GPAW(
    mode=PW(600),
    xc='PBE',
    kpts=(12, 12, 12),
    spinpol=True,
    occupations=FermiDirac(0.05),
    convergence={'energy': 1e-4},
    parallel={'gpu': False},
    txt=None
)
ref.calc = calc
ucf = UnitCellFilter(ref, scalar_pressure=0.0)
dyn = BFGS(ucf, logfile=None)
dyn.run(fmax=0.05, steps=50)

energy = ref.get_potential_energy()
print(f"Ru 初始原子数: {len(ref)}")
print(f"Ru 弛豫后每原子能量: {energy/len(ref):.4f} eV")
