from gpaw.lcaotddft.observer import TDDFTObserver


class RestartFileWriter(TDDFTObserver):
    """Observer for writing restart files periodically.

    At the given interval, the calculator restart file is written and
    ``write_restart()`` of every attached observer is called.

    The observer attaches to the TDDFT calculator during creation.

    Parameters
    ----------
    paw
        TDDFT calculator
    restart_filename
        File for writing the calculator object
    interval
        Update interval. The restart files are written every
        that many propagation steps.
    """
    def __init__(self, paw, restart_filename, interval=100):
        TDDFTObserver.__init__(self, paw, interval)
        self.restart_filename = restart_filename

    def _update(self, paw):
        if paw.niter == 0:
            return
        paw.log('%s activated' % self.__class__.__name__)
        for obs, n, args, kwargs in paw.observers:
            if isinstance(obs, TDDFTObserver):
                obs.write_restart()
        paw.write(self.restart_filename, mode='all')
