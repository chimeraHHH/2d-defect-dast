from gpaw.directmin.locfunc.etdm_localization_fdpw import FDPWETDMLocalize
from gpaw.directmin.locfunc.etdm_localization_lcao import LCAOETDMLocalize
from gpaw.directmin.fdpw.er_localization import ERLocalization as ERL
from gpaw.directmin.functional.fdpw import get_functional \
    as get_functional_fdpw
from gpaw.pipekmezey.pipek_mezey_wannier import PipekMezey
from gpaw.pipekmezey.wannier_basic import WannierLocalization
import numpy as np


def localize_orbitals(
        wfs, dens, ham, log, localizationtype, tol=None, seed=None,
        func_settings=None):
    io = localizationtype

    if io is None:
        return

    locnames = io.lower().split('_')
    if tol is not None:
        tols = {name: tol for name in locnames}
    else:
        tols = {'er': 5.0e-5,
                'pm': 1.0e-10,
                'fb': 1.0e-10,
                'pz': 5.0e-4,
                'ks': 5.0e-4}

    log("Initial localization: ...", flush=True)
    wfs.timer.start('Initial Localization')
    for name in locnames:
        tol = tols[name]
        if name == 'er':
            if wfs.mode == 'lcao':
                log('Edmiston-Ruedenberg localization is not supported '
                    'in LCAO',
                    flush=True)
                continue
            log('Edmiston-Ruedenberg localization started',
                flush=True)
            dm = FDPWETDMLocalize(
                ERL(wfs, dens), wfs,
                maxiter=200, g_tol=tol, randval=0.1)
            dm.run(wfs, dens, seed=seed)
            log('Edmiston-Ruedenberg localization finished',
                flush=True)
            del dm
        elif name == 'pz':
            if wfs.mode != 'lcao':
                log('Perdew-Zunger localization started', flush=True)
                PZC = get_functional_fdpw(func_settings, wfs, dens, ham)
                dm = FDPWETDMLocalize(
                    PZC, wfs, maxiter=200, g_tol=tol, randval=0.1)
                dm.run(wfs, dens, log=log)
                log('Perdew-Zunger localization finished', flush=True)
            else:
                dm = LCAOETDMLocalize(
                    wfs.eigensolver, wfs, log,
                    tol=wfs.eigensolver.subspace_convergence)
                dm.run(ham, dens)
        elif name == 'ks':
            log('ETDM minimization using occupied and virtual orbitals',
                flush=True)
            if wfs.mode == 'lcao':
                raise NotImplementedError
            else:
                KS = get_functional_fdpw('ks', wfs, dens, ham)
            dm = FDPWETDMLocalize(KS, wfs, maxiter=200, g_tol=tol, randval=0)
            dm.run(wfs, dens, ham=ham, log=log)
            log('ETDM minimization finished', flush=True)
        else:
            for k, kpt in enumerate(wfs.kpt_u):
                if sum(kpt.f_n) < 1.0e-3:
                    continue
                if name == 'pm':
                    if k == 0:
                        log('Pipek-Mezey localization started',
                            flush=True)
                    lf_obj = PipekMezey(
                        wfs=wfs, spin=kpt.s, dtype=wfs.dtype, seed=seed)
                    lf_obj.localize(tolerance=tol)
                    if k == 0:
                        log('Pipek-Mezey localization finished',
                            flush=True)
                    U = np.ascontiguousarray(
                        lf_obj.W_k[kpt.q].T)
                elif name == 'fb':
                    if k == 0:
                        log('Foster-Boys localization started',
                            flush=True)
                    lf_obj = WannierLocalization(
                        wfs=wfs, spin=kpt.s, seed=seed)
                    lf_obj.localize(tolerance=tol)
                    if k == 0:
                        log('Foster-Boys localization finished',
                            flush=True)
                    U = np.ascontiguousarray(
                        lf_obj.U_kww[kpt.q].T)
                    if wfs.dtype == float:
                        U = U.real
                else:
                    raise ValueError('Check localization type.')
                wfs.gd.comm.broadcast(U, 0)
                dim = U.shape[0]
                if wfs.mode == 'fd':
                    kpt.psit_nG[:dim] = np.einsum(
                        'ij,jkml->ikml', U, kpt.psit_nG[:dim])
                elif wfs.mode == 'pw':
                    kpt.psit_nG[:dim] = U @ kpt.psit_nG[:dim]
                else:
                    kpt.C_nM[:dim] = U @ kpt.C_nM[:dim]

                del lf_obj

    wfs.timer.stop('Initial Localization')
    log("Done", flush=True)
