from gpaw import GPAW_NO_C_EXTENSION

if GPAW_NO_C_EXTENSION:
    have_magma = False
    import gpaw.purepython as _gpaw
else:
    import _gpaw  # type: ignore[no-redef]

    # Do not force users to recompile due to merging magma support to master
    have_magma = getattr(_gpaw, 'have_magma', False)


def __getattr__(name):
    return getattr(_gpaw, name)
