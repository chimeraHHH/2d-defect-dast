import pytest

from gpaw.new.density import atomic_occupation_numbers
from gpaw.setup import create_setup
from gpaw.spinorbit import soc, projected_soc
from gpaw.xc import XC


def test_rad_pot():
    xc = XC('LDA', collinear=False)
    setup = create_setup('Cr', xc=xc)
    mz = 1.0
    f_si = atomic_occupation_numbers(setup, [0, 0, mz], 4)
    print(f_si)
    D_sp = setup.initialize_density_matrix(f_si)
    dv_vii = soc(setup, xc, D_sp)
    # Reversing magmom should give same potential
    D_sp[3] *= -1
    dv_vii -= soc(setup, xc, D_sp)
    assert abs(dv_vii).max() == pytest.approx(0.0, abs=1e-12)


def test_proj_soc():
    xc = XC('LDA', collinear=False)
    setup = create_setup('Cr', xc=xc)
    mz = 1.0
    f_si = atomic_occupation_numbers(setup, [0, 0, mz], 4)
    D_sp = setup.initialize_density_matrix(f_si)
    dv_vii = projected_soc(soc(setup, xc, D_sp),
                           theta=0, phi=0)
    # Reversing normal plane direction should give same potential
    dv_vii -= projected_soc(soc(setup, xc, D_sp),
                            theta=180, phi=0)
    assert abs(dv_vii).max() == pytest.approx(0.0, abs=1e-12)
