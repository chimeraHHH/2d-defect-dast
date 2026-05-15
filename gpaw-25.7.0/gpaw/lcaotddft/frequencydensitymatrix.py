import numpy as np

from ase.io.ulm import Reader
from gpaw.io import Writer

from gpaw.tddft.folding import Frequency
from gpaw.tddft.folding import FoldedFrequencies
from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import read_wuMM
from gpaw.lcaotddft.utilities import write_uMM
from gpaw.lcaotddft.utilities import write_wuMM


def generate_freq_w(foldedfreqs_f):
    freq_w = []
    for ff in foldedfreqs_f:
        for f in ff.frequencies:
            freq_w.append(Frequency(f, ff.folding, 'au'))
    return freq_w


class FrequencyDensityMatrixReader:
    def __init__(self, filename, ksl, kpt_u):
        self.ksl = ksl
        self.kpt_u = kpt_u
        self.reader = Reader(filename)
        tag = self.reader.get_tag()
        if tag != FrequencyDensityMatrix.ulmtag:
            raise RuntimeError('Unknown tag %s' % tag)
        self.version = self.reader.version

        # Read small vectors
        self.time = self.reader.time
        self.foldedfreqs_f = [FoldedFrequencies(**ff)
                              for ff in self.reader.foldedfreqs_f]
        self.freq_w = generate_freq_w(self.foldedfreqs_f)
        self.Nw = len(self.freq_w)

    def __getattr__(self, attr):
        if attr in ['rho0_uMM']:
            return read_uMM(self.kpt_u, self.ksl, self.reader, attr)
        if attr in ['FReDrho_wuMM', 'FImDrho_wuMM']:
            reim = attr[1:3]
            wlist = range(self.Nw)
            return self.read_FDrho(reim, wlist)

        try:
            return getattr(self.reader, attr)
        except KeyError:
            pass

        raise AttributeError('Attribute %s not defined in version %s' %
                             (repr(attr), repr(self.version)))

    def read_FDrho(self, reim, wlist):
        assert reim in ['Re', 'Im']
        attr = 'F%sDrho_wuMM' % reim
        return read_wuMM(self.kpt_u, self.ksl, self.reader, attr, wlist)

    def close(self):
        self.reader.close()


class FrequencyDensityMatrix(TDDFTObserver):
    version = 1
    ulmtag = 'FDM'

    def __init__(self,
                 paw,
                 dmat,
                 filename=None,
                 frequencies=None,
                 restart_filename=None,
                 interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.has_initialized = False
        self.dmat = dmat
        self.filename = filename
        self.restart_filename = restart_filename
        self.world = paw.world
        self.ksl = paw.wfs.ksl
        self.kd = paw.wfs.kd
        self.kpt_u = paw.wfs.kpt_u
        self.log = paw.log
        if self.ksl.using_blacs:
            ksl_comm = self.ksl.block_comm
            kd_comm = self.kd.comm
            assert self.world.size == ksl_comm.size * kd_comm.size

        assert self.world.rank == self.ksl.world.rank

        if filename is not None:
            self.read(filename)
            return

        self.time = paw.time
        if isinstance(frequencies, FoldedFrequencies):
            frequencies = [frequencies]
        self.foldedfreqs_f = frequencies
        self.freq_w = generate_freq_w(self.foldedfreqs_f)
        self.Nw = np.sum([len(ff.frequencies) for ff in self.foldedfreqs_f])

    def initialize(self):
        if self.has_initialized:
            return

        if self.kd.gamma:
            self.rho0_dtype = float
        else:
            self.rho0_dtype = complex

        self.rho0_uMM = []
        for kpt in self.kpt_u:
            self.rho0_uMM.append(self.dmat.zeros(self.rho0_dtype))
        self.FReDrho_wuMM = []
        self.FImDrho_wuMM = []
        for w in range(self.Nw):
            self.FReDrho_wuMM.append([])
            self.FImDrho_wuMM.append([])
            for kpt in self.kpt_u:
                self.FReDrho_wuMM[-1].append(self.dmat.zeros(complex))
                self.FImDrho_wuMM[-1].append(self.dmat.zeros(complex))
        self.has_initialized = True

    def _update(self, paw):
        if paw.action == 'init':
            if self.time != paw.time:
                raise RuntimeError('Timestamp do not match with '
                                   'the calculator')
            self.initialize()
            if paw.niter == 0:
                rho_uMM = self.dmat.get_density_matrix(paw.niter)
                for u, kpt in enumerate(self.kpt_u):
                    rho_MM = rho_uMM[u]
                    if self.rho0_dtype == float:
                        assert np.max(np.absolute(rho_MM.imag)) == 0.0
                        rho_MM = rho_MM.real
                    self.rho0_uMM[u][:] = rho_MM
            return

        if paw.action == 'kick':
            return

        assert paw.action == 'propagate'

        time_step = paw.time - self.time
        self.time = paw.time

        # Complex exponentials with envelope
        exp_w = []
        for ff in self.foldedfreqs_f:
            exp_i = (np.exp(1.0j * ff.frequencies * self.time) *
                     ff.folding.envelope(self.time) * time_step)
            exp_w.extend(exp_i.tolist())

        rho_uMM = self.dmat.get_density_matrix((paw.niter, paw.action))
        for u, kpt in enumerate(self.kpt_u):
            Drho_MM = rho_uMM[u] - self.rho0_uMM[u]
            for w, exp in enumerate(exp_w):
                # Update Fourier transforms
                self.FReDrho_wuMM[w][u] += Drho_MM.real * exp
                self.FImDrho_wuMM[w][u] += Drho_MM.imag * exp

    def write_restart(self):
        if self.restart_filename is None:
            return
        self.write(self.restart_filename)

    def write(self, filename):
        self.log(f'{self.__class__.__name__}: Writing to {filename}')
        writer = Writer(filename, self.world, mode='w',
                        tag=self.__class__.ulmtag)
        writer.write(version=self.__class__.version)
        writer.write(time=self.time)
        writer.write(foldedfreqs_f=[ff.todict() for ff in self.foldedfreqs_f])
        write_uMM(self.kd, self.ksl, writer, 'rho0_uMM', self.rho0_uMM)
        wlist = range(self.Nw)
        write_wuMM(self.kd, self.ksl, writer, 'FReDrho_wuMM',
                   self.FReDrho_wuMM, wlist)
        write_wuMM(self.kd, self.ksl, writer, 'FImDrho_wuMM',
                   self.FImDrho_wuMM, wlist)
        writer.close()

    def read(self, filename):
        reader = FrequencyDensityMatrixReader(filename, self.ksl, self.kpt_u)
        self.time = reader.time
        self.foldedfreqs_f = reader.foldedfreqs_f
        self.freq_w = reader.freq_w
        self.Nw = reader.Nw
        self.rho0_uMM = reader.rho0_uMM
        self.rho0_dtype = self.rho0_uMM[0].dtype
        self.FReDrho_wuMM = reader.FReDrho_wuMM
        self.FImDrho_wuMM = reader.FImDrho_wuMM
        reader.close()
        self.has_initialized = True
