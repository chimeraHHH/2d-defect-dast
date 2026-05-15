from contextlib import contextmanager
from functools import wraps
from io import StringIO
from typing import Callable, TypeVar, Union, overload, Optional
from gpaw import GPAW_TRACE
import inspect


class GlobalTimer:
    def __init__(self):
        from gpaw.utilities.timing import Timer
        self._timers = [Timer()]

    @contextmanager
    def context(self, timer):
        # XXX We need to decide what "timer contexts" are,
        # at least that will be necessary if we want a default
        # behaviour which would then be in effect during runs of the
        # test suite.
        if not GPAW_TRACE:
            raise RuntimeError('You need to set environment variable '
                               'GPAW_TRACE=1 in order to utilize '
                               'tracing via GlobalTimer.')
        self._timers.append(timer)
        try:
            yield
        finally:
            self._timers.pop()

    def start(self, name, **kwargs):
        timer = self._timers[-1]
        n_params = len(inspect.signature(timer.start).parameters)
        if n_params == 1:
            timer.start(name)
        else:
            timer.start(name, **kwargs)

    def stop(self, name=None, **kwargs):
        timer = self._timers[-1]
        n_params = len(inspect.signature(timer.stop).parameters)
        if n_params == 1:
            timer.stop(name)
        else:
            timer.stop(name, **kwargs)

    def tostring(self):
        buf = StringIO()
        self._timers[-1].write(out=buf)
        return buf.getvalue()


T = TypeVar('T')
F = Callable[..., T]


@overload
def _trace(meth: F, **timer_params) -> F:
    ...


@overload
def _trace(meth: None = None, **timer_params) -> Callable[[F], F]:
    ...


def _trace(meth: Optional[F] = None,
           **timer_params) -> Union[F, Callable[[F], F]]:
    """Decorator for telling global timer to trace a function or method."""

    def get_wrapper(method: Callable[..., T]) -> Callable[..., T]:
        modname = method.__module__
        methname = method.__qualname__
        name = f'{modname}.{methname}'

        @wraps(method)
        def wrapper(*args, **kwargs) -> T:
            global_timer.start(name, **timer_params)
            try:
                return method(*args, **kwargs)
            finally:
                global_timer.stop(**timer_params)

        return wrapper

    if meth:
        return get_wrapper(meth)

    return get_wrapper


def dummy_trace(meth: Union[Callable[..., T], None] = None,
                **timer_params) -> Callable[..., T]:
    if meth:
        return meth

    def wrapper(method):
        return method

    return wrapper


@contextmanager
def _tracectx(name, gpu=False):
    global_timer.start(name, gpu=gpu)
    try:
        yield
    finally:
        global_timer.stop(gpu=gpu)


@contextmanager
def dummy_tracectx(name, gpu=False):
    yield


trace = _trace if GPAW_TRACE else dummy_trace
tracectx = _tracectx if GPAW_TRACE else dummy_tracectx
global_timer = GlobalTimer()
