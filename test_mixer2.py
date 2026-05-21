from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac, MixerSum
import os

os.environ['GPAW_NEW'] = '1'
os.environ['GPAW_CPUPY'] = '1'

atom = bulk('V', 'bcc', a=3.0)
atom.set_initial_magnetic_moments([2.0])

calc = GPAW(
    mode=PW(300),
    xc='PBE',
    kpts=(4, 4, 4),
    spinpol=True,
    occupations=FermiDirac(0.1),
    mixer=MixerSum(beta=0.05, nmaxold=5, weight=50.0),
    maxiter=20,
    txt='test_mixer2.txt',
    parallel={}
)
atom.calc = calc
try:
    atom.get_potential_energy()
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
