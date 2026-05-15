import pytest
from ase.build import bulk
from gpaw.new.ase_interface import GPAW
from gpaw.mpi import world

# Prevent grid-dependent crash:
parallel = dict(band=1 if world.size < 8 else 4)


@pytest.fixture(scope='module', params=['fd', 'lcao', 'pw'])
def noprojs_gpw(module_tmp_path, request):
    mode = request.param
    atoms = bulk('Al')
    if mode in {'fd', 'lcao'}:
        kwargs = dict(mode=mode, gpts=(8, 8, 8))
    else:
        kwargs = dict(mode={'name': 'pw', 'ecut': 200.0})
    atoms.calc = GPAW(kpts=[2, 2, 2], txt=None, parallel=parallel,
                      convergence={'density': 1e6, 'eigenstates': 1e6},
                      **kwargs)
    atoms.get_potential_energy()
    gpw_path = module_tmp_path / f'gs_noprojs_{mode}.gpw'
    atoms.calc.write(gpw_path, include_projections=False)
    return gpw_path


def test_no_save_projections(noprojs_gpw):
    calc = GPAW(noprojs_gpw, parallel=parallel)
    ibzwfs = list(calc.dft.ibzwfs)
    assert len(ibzwfs) > 0
    for wfs in ibzwfs:
        assert wfs._P_ani is None


def test_nice_error_message(noprojs_gpw):
    if 'lcao' in noprojs_gpw.name:
        pytest.skip('LCAO is not quite ready for this')
    # We want there to be a good error message when we do not have
    # projections.  This only tests the most obvious case of .P_ani access,
    # but there could be code paths that will crash less controllably.
    calc = GPAW(noprojs_gpw, parallel=parallel)

    wfs = next(iter(calc.dft.ibzwfs))
    with pytest.raises(RuntimeError, match='There are no proj'):
        wfs.P_ani


def test_fixed_density_bandstructure(noprojs_gpw):
    calc = GPAW(noprojs_gpw, parallel=parallel)

    fixed_calc = calc.fixed_density(
        parallel=parallel,
        kpts=[[0., 0., 0.], [0., 0., 0.5]], symmetry='off')

    bs = fixed_calc.band_structure()
    assert len(bs.path.kpts) == 2
    ibzwfs = list(fixed_calc.dft.ibzwfs)

    for wfs in ibzwfs:
        assert len(wfs.P_ani) == len(calc.get_atoms())
    # Should we test something else here?
    # If we calculate a full bandstructure, it looks realistic.
    # We could compare to an "ordinary" (with projections) gpw file
    # to see that the numbers are in fact unaffected by the distinction.
