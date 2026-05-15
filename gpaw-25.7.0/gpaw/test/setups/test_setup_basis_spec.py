"""This test looks for differently named setups and basis sets to ascertain
that correct filenames are constructed.  Generally speaking, the things
it looks for do not exist; we just verify the filenames.
"""
import pytest
from gpaw.basis_data import get_basis_name, parse_basis_name
from gpaw.setup import Setups


@pytest.mark.ci
@pytest.mark.parametrize(
    'setups, basis, refname',
    [({}, 'hello', 'Na.hello.basis'),
     ('hello', {}, 'Na.hello.LDA'),
     ('hello', 'dzp', 'Na.hello.dzp.basis'),
     ('hello', 'sz(dzp)', 'Na.hello.dzp.basis'),
     ('hello', 'world.dzp', 'Na.hello.world.dzp.basis'),
     ('hello', 'sz(world.dzp)', 'Na.hello.world.dzp.basis'),
     ('paw', 'world.dzp', 'Na.world.dzp.basis'),
     ('paw', 'sz(world.dzp)', 'Na.world.dzp.basis')])
def test_setup_basis_spec(setups, basis, refname):
    with pytest.raises(FileNotFoundError) as err:
        Setups([11], setups, basis, 'LDA')
    msg = str(err)
    fname = msg.split('"')[1]
    assert fname == refname, fname


@pytest.mark.parametrize('basisname',
                         'sz dz szp dzp dz2p dzdp tzqp dzsp'.split())
def test_basis_something_something(basisname):
    zetacount, polarizationcount = parse_basis_name(basisname)
    normalized_name = get_basis_name(zetacount, polarizationcount)
    zetacount2, polarizationcount2 = parse_basis_name(normalized_name)
    assert zetacount == zetacount2
    assert polarizationcount == polarizationcount2
    if (polarizationcount < 2 and
        len(basisname) < 4 and basisname.isalpha()):
        assert normalized_name == basisname
