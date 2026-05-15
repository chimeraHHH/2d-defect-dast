import pytest
import numpy as np

from gpaw.basis_data import Basis
from gpaw.mpi import world
from gpaw.lcao.generate_ngto_augmented import \
    create_GTO_dictionary as GTO, create_CGTO_dictionary as CGTO, \
    generate_nao_ngto_basis, read_gaussian_basis_file

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


def test_read_gaussian(in_tmp_dir):
    atom = 'Au'
    description = 'test description\nline2'
    gtos = [GTO(0, 0.2),
            GTO('s', 0.1),
            GTO(1, 0.05),
            CGTO(1, [9.2, 2.1, 0.54, 0.15], [0.02, 0.18, 0.3, 0.5]),
            CGTO(2, [0.05], [1.0])]

    with open('gbs.txt', 'w') as f:
        def w(s):
            f.write('%s\n' % s)
        w('****')
        for desc in description.split('\n'):
            w(f'! {desc}')
        w(f'{atom} 0')
        w('S   1   1.00')
        w('      0.2000000              1.000000D+00')
        w('S   1   1.00')
        w('      0.1000000              1.0000000')
        w('P   1   1.00')
        w('      0.0500000              1.0000000')
        w('P   4   1.00')
        w('      9.200000D+00           2.000000E-02')
        w('      2.100000E+00           1.800000D-01')
        w('      5.400000e-01           3.000000D-01')
        w('      1.500000D-01           5.000000e-01')
        w('D   1   1.00')
        w('      0.500000D-01           1.0000000')
        w('****')
        w('This line is never read')

    gbs_atom, gbs_description, gbs_gtos = read_gaussian_basis_file('gbs.txt')
    assert gbs_atom == atom
    assert gbs_description == description
    assert len(gbs_gtos) == len(gtos)
    for gbs_gto, gto in zip(gbs_gtos, gtos):
        for key in gto:
            assert np.allclose(gbs_gto[key], gto[key])


def test_generate(in_tmp_dir):
    with open('gbs.txt', 'w') as f:
        def w(s):
            f.write('%s\n' % s)
        w('C     0')
        w('S   1   1.00')
        w('      1.596000D-01           1.000000D+00')
        w('S   1   1.00')
        w('      0.0469000              1.0000000')
        w('P   4   1.00')
        w('      9.439000D+00           3.810900D-02')
        w('      2.002000D+00           2.094800D-01')
        w('      5.456000D-01           5.085570D-01')
        w('      1.517000D-01           4.688420D-01')
        w('P   1   1.00')
        w('      1.517000D-01           1.000000D+00')
        w('P   1   1.00')
        w('      0.0404100              1.0000000')
        w('D   1   1.00')
        w('      5.500000D-01           1.0000000')
        w('D   1   1.00')
        w('      0.1510000              1.0000000')

    gbs_atom, gbs_description, gtos = read_gaussian_basis_file('gbs.txt')

    assert gbs_atom == 'C'
    generate_nao_ngto_basis('C', xc='LDA', nao='dzp', name='NAO+NGTO',
                            gtos=gtos, gto_description=gbs_description)

    basis = Basis.read_path('C', 'NAO+NGTO', 'C.NAO+NGTO.dzp.basis')
    assert len(basis.bf_j) == 5 + 7
