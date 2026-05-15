import atexit
import numpy as np
import gpaw.cgpaw as cgpaw


def _elpaconstants():
    consts = cgpaw.pyelpa_constants()
    assert consts[0] == 0, f'ELPA_OK is {consts[0]}, expected 0'
    return {'elpa_ok': consts[0],
            '1stage': consts[1],
            '2stage': consts[2]}


class LibElpa:
    @staticmethod
    def have_elpa():
        return hasattr(cgpaw, 'pyelpa_allocate')

    _elpa_initialized = False

    @staticmethod
    def api_version():
        return cgpaw.pyelpa_version()

    @classmethod
    def ensure_elpa_initialized(cls):
        if not cls._elpa_initialized:
            cgpaw.pyelpa_init()
            atexit.register(cgpaw.pyelpa_uninit)
            cls._elpa_initialized = True

    def __init__(self, desc, nev=None, solver='1stage'):
        if nev is None:
            nev = desc.gshape[0]

        ptr = np.zeros(1, np.intp)

        if not self.have_elpa():
            raise ImportError('GPAW is not running in parallel or otherwise '
                              'not compiled with Elpa support')

        if desc.nb != desc.mb:
            raise ValueError('Row and column block size must be '
                             'identical to support Elpa')

        self.ensure_elpa_initialized()
        cgpaw.pyelpa_allocate(ptr)
        self._ptr = ptr
        cgpaw.pyelpa_set_comm(ptr,
                              desc.blacsgrid.comm.get_c_object())
        self._parameters = {}

        self._consts = _elpaconstants()
        elpasolver = self._consts[solver]

        bg = desc.blacsgrid
        self.elpa_set(na=desc.gshape[0],
                      local_ncols=desc.shape[0],
                      local_nrows=desc.shape[1],
                      nblk=desc.mb,
                      process_col=bg.myrow,  # XXX interchanged
                      process_row=bg.mycol,
                      blacs_context=bg.context)
        self.elpa_set(nev=nev, solver=elpasolver)
        self.desc = desc

        cgpaw.pyelpa_setup(self._ptr)

    @property
    def description(self):
        solver = self._parameters['solver']
        if solver == self._consts['1stage']:
            pretty = 'Elpa one-stage solver'
        else:
            assert solver == self._consts['2stage']
            pretty = 'Elpa two-stage solver'
        return pretty

    @property
    def nev(self):
        return self._parameters['nev']

    def _is_complex(self, array):
        if array.dtype == np.complex128:
            return True
        if array.dtype == np.float64:
            return False

        raise TypeError(f'Unsupported dtype {array.dtype} for Elpa interface')

    def diagonalize(self, A, C, eps):
        assert self.nev == len(eps)
        self.desc.checkassert(A)
        self.desc.checkassert(C)
        cgpaw.pyelpa_diagonalize(self._ptr, A, C, eps, self._is_complex(A))

    def general_diagonalize(self, A, S, C, eps, is_already_decomposed=0):
        assert self.nev == len(eps)
        self.desc.checkassert(A)
        self.desc.checkassert(S)
        self.desc.checkassert(C)
        cgpaw.pyelpa_general_diagonalize(self._ptr, A, S, C, eps,
                                         is_already_decomposed,
                                         self._is_complex(A))

    def elpa_set(self, **kwargs):
        for key, value in kwargs.items():
            # print('pyelpa_set {}={}'.format(key, value))
            cgpaw.pyelpa_set(self._ptr, key, value)
            self._parameters[key] = value

    def __repr__(self):
        return f'LibElpa({self._parameters})'

    def __del__(self):
        if hasattr(self, '_ptr'):
            # elpa_deallocate has no error flag so we don't check it
            cgpaw.pyelpa_deallocate(self._ptr)
            self._ptr[0] = 0
