from ase import Atoms
from ase.units import Ha
from ase.parallel import parprint

from gpaw import GPAW

from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.pes.state import BoundState, H1s
from gpaw.pes.ds_beta import CrossSectionBeta


def test_ds_beta(in_tmp_dir):
    ngauss = 2

    h = .3
    box = 3.

    gpwname = 'H1s.gpw'
    if 1:
        c = GPAW(mode='fd', xc='PBE', nbands=-1, h=h)
        s = Atoms('H')
        adjust_cell(s, box, h=h)
        c.calculate(s)
        c.write(gpwname, 'all')
    else:
        c = GPAW(gpwname)
        s = c.get_atoms()
        c.converge_wave_functions()
    cm = s.get_center_of_mass()
    Ekin = 1.

    for form, title in [('L', 'length form'),
                        ('V', 'velocity form')]:
        parprint('--', title)
        ds = []
        for analytic in [True, False]:
            if analytic:
                initial = H1s(c.density.gd, cm)
            else:
                initial = BoundState(c, 0, 0)
                initial.set_energy(- Ha / 2.)

            csb = CrossSectionBeta(initial=initial,
                                   final=None,
                                   r0=cm, ngauss=ngauss, form=form)
            if analytic:
                ds.append(initial.get_ds(Ekin, form))
                parprint('analytic 1s energy, beta, ds %5.3f' %
                         (Ekin + Ha / 2.), end='')
                parprint(f'{2:8.4f} {ds[-1]:12.5f}')
            ds.append(csb.get_ds(Ekin))
            parprint('numeric  1s energy, beta, ds %5.3f' %
                     (Ekin + Ha / 2.), end='')
            parprint(f'{csb.get_beta(Ekin):8.4f} {ds[-1]:12.5f}')
        parprint('error analytic GS:',
                 int(100 * abs(ds[1] / ds[0] - 1.) + .5), '%')
        assert abs(ds[1] / ds[0] - 1.) < 0.31
        parprint('error numeric GS:',
                 int(100 * abs(ds[2] / ds[0] - 1.) + .5), '%')
        assert abs(ds[2] / ds[0] - 1.) < 0.2
