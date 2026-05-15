import pytest
from gpaw.core import UGDesc, PWDesc
from gpaw.new.pw.hamiltonian import apply_local_potential_gpu
from gpaw.gpu import cupy as cp


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('nbands', [1, 2, 3, 5])
def test_apply_loc_pot(dtype, nbands):
    a = 1.5
    n = 8
    vt_R = UGDesc(cell=[a, a, a], size=(n, n, n)).empty(xp=cp)
    v0 = 1.3
    vt_R.data[:] = v0
    if dtype == complex:
        kpt = [0, 0, 0.5]
    else:
        kpt = None
    pw = PWDesc(cell=vt_R.desc.cell, ecut=15.0, kpt=kpt)
    psit_nG = pw.empty(nbands, xp=cp)
    p0 = 1.2 - 2.0j
    psit_nG.data[:] = p0
    if dtype == float:
        psit_nG.data[:, 0] = -1.1
    out_nG = pw.empty(nbands, xp=cp)
    apply_local_potential_gpu(vt_R,
                              psit_nG,
                              out_nG,
                              blocksize=3)
    error_nG = (cp.asnumpy(out_nG.data) -
                (v0 + pw.ekin_G) * cp.asnumpy(psit_nG.data))
    assert abs(error_nG).max() == pytest.approx(0.0)


if __name__ == '__main__':
    test_apply_loc_pot(float, 1)
