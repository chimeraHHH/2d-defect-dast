from gpaw.utilities.acwf import work
import json
from pathlib import Path


def workflow():
    from myqueue.workflow import run
    for mode in ['pw', 'lcao']:
        with run(function=work, args=['FCC', 'Al'], kwargs={'mode': mode},
                 cores=8, tmax='1h', name=f'Al-{mode}'):
            run(function=check_lattice_constant, args=[mode],
                name=f'check-{mode}')


def check_lattice_constant(mode: str) -> None:
    dct = json.loads(Path(f'{mode}-FCC.json').read_text())
    strain = {'pw': 0.0003,
              'lcao': 0.0086}[mode]
    assert abs(dct['strain'] - strain) < 0.0001
