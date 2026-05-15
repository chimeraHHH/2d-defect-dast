from __future__ import annotations

import contextlib
import io
import os
import sys
from functools import cache
from pathlib import Path
from typing import IO, Any, Sequence

from gpaw.mpi import MPIComm, world


def indent(text: Any, indentation='  ') -> str:
    r"""Indent text blob.

    >>> indent('line 1\nline 2', '..')
    '..line 1\n..line 2'
    """
    if not isinstance(text, str):
        text = str(text)
    return indentation + text.replace('\n', '\n' + indentation)


class Logger:
    def __init__(self,
                 filename: str | Path | IO[str] | None = '-',
                 comm: MPIComm | Sequence[int] | None = None):
        if comm is None:
            comm = world
        elif not hasattr(comm, 'rank'):
            comm = world.new_communicator(list(comm))

        self.comm: MPIComm = comm  # type: ignore

        self.fd: IO[str]

        if self.comm.rank > 0 or filename is None:
            self.fd = open(os.devnull, 'w', encoding='utf-8')
            self.close_fd = True
        elif filename == '-':
            self.fd = sys.stdout
            self.close_fd = False
        elif isinstance(filename, (str, Path)):
            self.fd = open(filename, 'w', encoding='utf-8')
            self.close_fd = True
        else:
            self.fd = filename
            self.close_fd = False

        self.indentation = ''

        self.use_colors = can_colorize(file=self.fd)
        if self.use_colors:
            self.green = '\x1b[32m'
            self.reset = '\x1b[0m'
        else:
            self.green = ''
            self.reset = ''

    def __del__(self):
        self.close()

    def close(self) -> None:
        if self.close_fd:
            self.fd.close()

    @contextlib.contextmanager
    def indent(self, text):
        self(text)
        self.indentation += '  '
        yield
        self.indentation = self.indentation[2:]

    def __call__(self, *args, end=None, flush=False) -> None:
        if self.fd.closed:
            return
        i = self.indentation
        text = ' '.join(str(arg) for arg in args)
        if i:
            text = (i + text.replace('\n', '\n' + i)).rstrip(' ')
        print(text, file=self.fd, end=end, flush=flush)


def can_colorize(*, file: IO[str] | IO[bytes] | None = None) -> bool:
    """Code from Python 3.14b1: cpython/Lib/_colorize.py."""
    ok = _can_colorize()
    if ok is not None:
        return ok

    if file is None:
        file = sys.stdout

    if not hasattr(file, 'fileno'):
        return False

    try:
        return os.isatty(file.fileno())
    except io.UnsupportedOperation:
        return hasattr(file, 'isatty') and file.isatty()


@cache
def _can_colorize() -> bool | None:
    """Check standard envvars for colors.

    See https://docs.python.org/3/using/cmdline.html#controlling-color

    Returns None if undecided.
    """
    if not sys.flags.ignore_environment:
        if os.environ.get('PYTHON_COLORS') == '0':
            return False
        if os.environ.get('PYTHON_COLORS') == '1':
            return True
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    if os.environ.get('TERM') == 'dumb':
        return False
    if sys.platform == 'win32':
        try:
            import nt

            if not nt._supports_virtual_terminal():
                return False
        except (ImportError, AttributeError):
            return False
    return None
