class Observer:

    def __init__(self, interval=1):
        object.__init__(self)
        self.niter = 0
        self.interval = interval

    def __call__(self, *args, **kwargs):
        self.niter += self.interval
        self.update(*args, **kwargs)

    def update(self):
        raise RuntimeError('Virtual member function called.')


class WritableObserver(Observer):

    def __init__(self, w, interval=1):
        super().__init__(interval)
        self.w = w

    def __del__(self):
        self.w.close()
