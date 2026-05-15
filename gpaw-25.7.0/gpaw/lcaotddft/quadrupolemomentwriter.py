import re
import numpy as np

from ase.utils import IOContext

from gpaw.lcaotddft.observer import TDDFTObserver


def calculate_quadrupole_moments(gd, rho_g, center_v):
    # Center relative to the cell center
    # center_v += 0.5 * gd.cell_cv.sum(0)
    r_vg = gd.get_grid_point_coordinates()
    qm_i = np.zeros(6, dtype=float)
    i = 0
    for v1 in range(3):
        x1_g = r_vg[v1] - center_v[v1]
        for v2 in range(v1, 3):
            x2_g = r_vg[v2] - center_v[v2]
            qm_i[i] = -gd.integrate(x1_g * x2_g * rho_g)
            i += 1
    return qm_i


class QuadrupoleMomentWriter(TDDFTObserver):
    version = 1

    def __init__(self, paw, filename, center=[0, 0, 0], density='comp',
                 interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.ioctx = IOContext()
        if paw.niter == 0:
            # Initialize
            if isinstance(center, str) and center == 'center':
                self.center_v = 0.5 * paw.density.finegd.cell_cv.sum(0)
            else:
                assert len(center) == 3
                self.center_v = center
            self.density_type = density
            self.fd = self.ioctx.openfile(filename, comm=paw.world, mode='w')
        else:
            # Read and continue
            self.read_header(filename)
            self.fd = self.ioctx.openfile(filename, comm=paw.world, mode='a')

    def _write(self, line):
        self.fd.write(line)
        self.fd.flush()

    def _write_header(self, paw):
        if paw.niter != 0:
            return
        line = f'# {self.__class__.__name__}[version={self.version}]'
        line += '(center=[%.6f, %.6f, %.6f]' % tuple(self.center_v)
        line += ', density=%s)\n' % repr(self.density_type)
        line += ('# %15s %15s %22s %22s %22s %22s %22s %22s\n' %
                 ('time', 'norm', 'x^2', 'xy', 'xz', 'y^2', 'yz', 'z^2'))
        self._write(line)

    def read_header(self, filename):
        with open(filename) as f:
            line = f.readline()
        regexp = (r"^(?P<name>\w+)\[version=(?P<version>\d+)\]\(center=\["
                  r"(?P<center0>[-+0-9\.]+), "
                  r"(?P<center1>[-+0-9\.]+), "
                  r"(?P<center2>[-+0-9\.]+)\], "
                  r"density='(?P<density>\w+)'\)$")
        m = re.match(regexp, line[2:])
        assert m is not None, 'Unknown fileformat'
        assert m.group('name') == self.__class__.__name__
        assert int(m.group('version')) == self.version
        self.center_v = [float(m.group('center%d' % v)) for v in range(3)]
        self.density_type = m.group('density')

    def _write_kick(self, paw):
        line = '# Kick: %s' % paw.kick_ext
        line += '; Time = %.8lf' % paw.time
        line += '\n'
        self._write(line)

    def _write_dm(self, paw):
        time = paw.time
        density = paw.density
        if self.density_type == 'comp':
            rho_g = density.rhot_g
            gd = density.finegd
        elif self.density_type == 'pseudo':
            rho_g = density.nt_sg.sum(axis=0)
            gd = density.finegd
        elif self.density_type == 'pseudocoarse':
            rho_g = density.nt_sG.sum(axis=0)
            gd = density.gd
        else:
            raise RuntimeError('Unknown density type: %s' % self.density_type)

        norm = gd.integrate(rho_g)
        qm = calculate_quadrupole_moments(gd, rho_g, center_v=self.center_v)
        line = (('%20.8lf %20.8le' + ' %22.12le' * len(qm) + '\n') %
                ((time, norm) + tuple(qm)))
        self._write(line)

    def _update(self, paw):
        if paw.action == 'init':
            self._write_header(paw)
        elif paw.action == 'kick':
            self._write_kick(paw)
        self._write_dm(paw)

    def __del__(self):
        self.ioctx.close()
