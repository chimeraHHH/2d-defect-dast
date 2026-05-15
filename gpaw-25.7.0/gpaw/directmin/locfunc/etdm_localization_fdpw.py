from gpaw.directmin.fdpw.pz_localization import PZLocalization
from gpaw.directmin.fdpw.etdm_inner_loop import ETDMInnerLoop
import numpy as np


class FDPWETDMLocalize:

    def __init__(self, obj_f, wfs, maxiter=300, g_tol=1.0e-3, randval=0.01):

        self.obj_f = obj_f
        self.randval = randval

        if obj_f.name == 'Zero':
            self.iloop = ETDMInnerLoop(
                obj_f, wfs, 'all', 5.0e-4, maxiter,
                g_tol=g_tol, useprec=True)
        else:
            self.iloop = PZLocalization(
                obj_f, wfs, maxiter=maxiter,
                g_tol=g_tol)

    def run(self, wfs, dens, ham=None, rewritepsi=True, log=None, seed=None):

        wfs.timer.start('Inner loop')

        if self.obj_f.name == 'Zero':
            etotal, counter = self.iloop.run(wfs, dens, log, 0, ham=ham)
        else:
            counter = self.iloop.run(0.0, wfs, dens, log, 0,
                                     randvalue=self.randval, seed=seed)

        if rewritepsi:
            for kpt in wfs.kpt_u:
                k = self.iloop.n_kps * kpt.s + kpt.q
                dim1 = self.iloop.U_k[k].shape[0]
                kpt.psit_nG[:dim1] = np.tensordot(
                    self.iloop.U_k[k].T, kpt.psit_nG[:dim1], axes=1)
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        wfs.timer.stop('Inner loop')

        return counter
