"""Test EXX/HFT implementation."""
import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.xc import XC
from gpaw.xc.hybrid import HybridXC


def xc1(name):
    return dict(name=name, stencil=1)


@pytest.mark.slow
@pytest.mark.libxc
@pytest.mark.hybrids
def test_exx_exx(in_tmp_dir):
    be2 = Atoms('Be2', [(0, 0, 0), (2.45, 0, 0)])
    be2.center(vacuum=2.0)

    ref_1871 = {  # Values from revision 1871. Not true reference values
        # xc         Energy          eigenvalue 0    eigenvalue 1
        'PBE': (5.424066548470926, -3.84092, -0.96192),
        'PBE0': (-790.919942, -4.92321, -1.62948),
        'EXX': (-785.5837828306236, -7.16802337336, -2.72602997017)}

    current = {}  # Current revision
    for xc in [XC(xc1('PBE')),
               HybridXC('PBE0', stencil=1, finegrid=True),
               HybridXC('EXX', stencil=1, finegrid=True),
               XC(xc1('PBE'))]:  # , 'oldPBE', 'LDA']:
        # Generate setup
        # g = Generator('Be', setup, scalarrel=True, nofiles=True, txt=None)
        # g.run(exx=True, **parameters['Be'])

        # switch to new xc functional
        calc = GPAW(mode='fd',
                    xc=xc,
                    h=0.21,
                    eigensolver='rmm-diis',
                    nbands=3,
                    convergence={'eigenstates': 1e-6},
                    txt='exx.txt')
        be2.calc = calc
        E = be2.get_potential_energy()
        if xc.name != 'PBE':
            E += calc.get_reference_energy()
        bands = calc.get_eigenvalues()[:2]  # not 3 as unocc. eig are random!?
        res = (E,) + tuple(bands)
        print(xc.name, res)

        if xc.name in current:
            for first, second in zip(current[xc.name], res):
                assert first == pytest.approx(second, abs=2.5e-3)
        else:
            current[xc.name] = res

    for name in current:
        for ref, cur in zip(ref_1871[name], current[name]):
            print(ref, cur, ref - cur)
            assert ref == pytest.approx(cur, abs=2.9e-3)
