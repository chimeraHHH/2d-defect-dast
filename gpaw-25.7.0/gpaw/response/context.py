from __future__ import annotations
from typing import Union
from pathlib import Path
from time import ctime
from sys import stdout

from inspect import isgeneratorfunction
from functools import wraps

from ase.utils import IOContext
from ase.utils.timing import Timer

import gpaw.mpi as mpi


TXTFilename = Union[Path, str]
ResponseContextInput = Union['ResponseContext', dict, TXTFilename]


class ResponseContext:
    def __init__(self, txt: TXTFilename = '-',
                 timer=None, comm=mpi.world, mode='w'):
        self.comm = comm
        self.iocontext = IOContext()
        self.open(txt, mode)
        self.set_timer(timer)

    @staticmethod
    def from_input(context: ResponseContextInput) -> ResponseContext:
        if isinstance(context, ResponseContext):
            return context
        elif isinstance(context, dict):
            return ResponseContext(**context)
        elif isinstance(context, (Path, str)):  # TXTFilename
            return ResponseContext(txt=context)
        raise ValueError('Expected ResponseContextInput, got', context)

    def open(self, txt, mode):
        if txt is stdout and self.comm.rank != 0:
            txt = None
        self.fd = self.iocontext.openfile(txt, self.comm, mode)

    def set_timer(self, timer):
        self.timer = timer or Timer()

    def close(self):
        self.iocontext.close()

    def __del__(self):
        self.close()

    def with_txt(self, txt, mode='w'):
        return ResponseContext(txt=txt, comm=self.comm, timer=self.timer,
                               mode=mode)

    def print(self, *args, flush=True, **kwargs):
        print(*args, file=self.fd, flush=flush, **kwargs)

    def new_txt_and_timer(self, txt, timer=None):
        self.write_timer()
        # Close old output file and create a new
        self.close()
        self.open(txt, mode='w')
        self.set_timer(timer)

    def write_timer(self):
        self.timer.write(self.fd)
        self.print(ctime())


class timer:
    """Decorator for timing a method call.
    NB: Includes copy-paste from ase, which is suboptimal...

    Example::

        from gpaw.response.context import timer

        class A:
            def __init__(self, context):
                self.context = context

            @timer('Add two numbers')
            def add(self, x, y):
                return x + y

        """
    def __init__(self, name):
        self.name = name

    def __call__(self, method):
        if isgeneratorfunction(method):
            @wraps(method)
            def new_method(slf, *args, **kwargs):
                gen = method(slf, *args, **kwargs)
                while True:
                    slf.context.timer.start(self.name)
                    try:
                        x = next(gen)
                    except StopIteration:
                        break
                    finally:
                        slf.context.timer.stop()
                    yield x
        else:
            @wraps(method)
            def new_method(slf, *args, **kwargs):
                slf.context.timer.start(self.name)
                x = method(slf, *args, **kwargs)
                try:
                    slf.context.timer.stop()
                except IndexError:
                    pass
                return x
        return new_method
