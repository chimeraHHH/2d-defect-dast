import pytest
from gpaw.lrtddft import LrTDDFT


@pytest.mark.lrtddft
def test_rpa(H2, in_tmp_dir):
    lr = LrTDDFT(H2.calc, xc='RPA', restrict={'to': [1]})
    assert len(lr) == 1

    fname = 'rpa.dat.gz'
    lr.write(fname)

    lr2 = LrTDDFT.read(fname)
    assert len(lr2) == 1
