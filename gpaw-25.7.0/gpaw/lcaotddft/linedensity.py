import numpy as np

import ase.io.ulm as ulm

from gpaw.io import Writer
from gpaw.lcaotddft.observer import TDDFTObserver


class LineDensityReader:
    def __init__(self, filename):
        self.reader = ulm.Reader(filename)
        tag = self.reader.get_tag()
        if tag != LineDensityWriter.ulmtag:
            raise RuntimeError('Unknown tag %s' % tag)
        self.filename = filename

    def __getattr__(self, attr):
        return getattr(self.reader, attr)

    @property
    def kick_strength(self):
        for ri in range(1, len(self.reader)):
            r = self.reader[ri]
            if r.action == 'kick':
                return r.kick_strength

    def read(self, skip_duplicates=True, zero_pad='both'):
        assert zero_pad in ['both', False, None]
        time_t = []
        rho_tx = []
        time0 = -np.inf
        for ri in range(1, len(self.reader)):
            r = self.reader[ri]
            time = r.time
            if time > time0 or not skip_duplicates:
                time_t.append(time)
                rho_x = r.rho_x
                if zero_pad == 'both':
                    rho_x = np.concatenate(([0], rho_x, [0]))
                rho_tx.append(rho_x)
                time0 = time
        time_t = np.array(time_t)
        rho_tx = np.array(rho_tx)
        return time_t, rho_tx

    def close(self):
        self.reader.close()


class LineDensityWriter(TDDFTObserver):
    version = 1
    ulmtag = 'LineDensity'

    def __init__(self, paw, filename, c=0, density_type='comp', interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        if paw.niter == 0:
            self.density_type = density_type
            self.c = c
            density = paw.density
            if 'coarse' in density_type:
                gd = density.gd
            else:
                gd = density.finegd
            h_v = gd.h_cv[self.c]

            self.writer = Writer(filename, paw.world, mode='w',
                                 tag=self.__class__.ulmtag)
            self.writer.write(version=self.__class__.version,
                              density_type=self.density_type,
                              c=self.c, h_v=h_v)
            self.writer.sync()
        else:
            # Read settings
            reader = LineDensityReader(filename)
            self.density_type = reader.density_type
            self.c = reader.c
            h_v = reader.h_v

            # Append to earlier file
            self.writer = Writer(filename, paw.world, mode='a',
                                 tag=self.__class__.ulmtag)
        self.dx = np.linalg.norm(h_v)
        assert self.density_type in ['comp', 'pseudo', 'pseudocoarse'], \
            f'Unknown density type: {self.density_type}'

    def _update(self, paw):
        # Write metadata to main writer
        self.writer.write(niter=paw.niter, time=paw.time, action=paw.action)
        if paw.action == 'kick':
            self.writer.write(kick_strength=paw.kick_strength)

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

        rho_g = gd.collect(rho_g)
        # TODO: instead of collecting on master,
        # do integration in each domain and gather/sum to master
        if gd.comm.rank == 0:
            sum_axis = tuple({0, 1, 2}.difference({self.c}))
            rho_x = rho_g.sum(axis=sum_axis) * (gd.dv / self.dx)
        else:
            rho_x = None
        self.writer.write(rho_x=rho_x)

        # Sync the main writer
        self.writer.sync()

    def __del__(self):
        self.writer.close()
