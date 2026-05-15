import pytest
from gpaw.xc import XC


@pytest.mark.libxc
def test_pbe0():
    xc = XC('PBE0:backend=pw')
    assert xc.exx_fraction == 0.25
    xc = XC('HYB_GGA_XC_PBEH:backend=pw:omega=0:fraction=0.25')
    assert xc.exx_fraction == 0.25
