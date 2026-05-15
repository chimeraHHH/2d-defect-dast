def __getattr__(attr):
    if attr == 'scipy':
        import gpaw.gpu.cpupyx.scipy as scipy
        return scipy
    raise AttributeError(attr)
