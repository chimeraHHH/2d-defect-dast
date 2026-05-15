import pytest

from gpaw.atom.all_electron import ValenceData
from gpaw.atom.basis import BasisMaker
from gpaw.atom.configurations import parameters
from gpaw.atom.generator import Generator


def test_basismaker_onthefly_potentials():
    """Test on-the-fly potentials.

    Specifically, test that generated basis functions are
    approximately equal whether using on-the-fly potentials or not.

    We expect them to be the same except for a mixing step which the
    generator performs, which will affect historic basis sets but will
    not be present in basis sets generated directly from
    files/setupdata.

    """

    sym = 'Ti'
    gen = Generator(sym, xcname='PBE')
    setup = gen.run(write_xml=False, **parameters[sym])

    valdata1 = ValenceData.from_setupdata_and_potentials(
        setup, vr_g=gen.vr, r2dvdr_g=gen.r2dvdr, scalarrel=gen.scalarrel)
    valdata2 = ValenceData.from_setupdata_onthefly_potentials(setup)

    basis1 = BasisMaker(valdata1).generate()
    basis2 = BasisMaker(valdata2).generate()

    assert len(basis1.bf_j) == len(basis2.bf_j)
    for bf1, bf2 in zip(basis1.bf_j, basis2.bf_j):
        assert bf1.l == bf2.l
        assert bf1.n == bf2.n
        assert bf1.rc == pytest.approx(bf2.rc)
        assert bf1.phit_g == pytest.approx(bf2.phit_g, abs=1e-6, rel=1e-6)
