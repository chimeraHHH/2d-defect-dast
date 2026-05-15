from gpaw.analyse.observers import Observer


class TDDFTObserver(Observer):

    def __init__(self, paw, interval):
        super().__init__(interval)
        self.timer = paw.timer
        if hasattr(paw, 'time') and hasattr(paw, 'niter'):
            paw.attach(self, interval, paw)

    def update(self, paw):
        self.timer.start('%s update' % self.__class__.__name__)
        self._update(paw)
        self.timer.stop('%s update' % self.__class__.__name__)

    def _update(self, paw):
        raise NotImplementedError()

    def write_restart(self):
        """Write restart file.

        Optional method that will be called by RestartFileWriter
        when writing the calculator restart file.
        """

    def __str__(self):
        return self.__class__.__name__
