from __future__ import annotations

import os
import sys
from typing import Literal
from textwrap import fill

from ase.utils import import_module, search_current_git_hash

import gpaw.cgpaw as cgpaw
import gpaw
import gpaw.fftw as fftw
from gpaw.mpi import have_mpi, rank
from gpaw.new.c import GPU_AWARE_MPI, GPU_ENABLED
from gpaw.utilities import compiled_with_libvdwxc, compiled_with_sl
from gpaw.utilities.elpa import LibElpa
from gpaw.gpu import cupy, cupy_is_fake, __file__ as gpaw_gpu_filename


Color = Literal['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'none']

# Background: +10
COLORS: dict[Color, int] = {'r': 31, 'g': 32, 'b': 34,
                            'c': 36, 'm': 35, 'y': 33,
                            'k': 30, 'w': 37,
                            'none': 39}


def highlight(text: str,
              foreground: Color = 'none',
              background: Color = 'none',
              *,
              bright: bool = True) -> str:
    sgr = '\x1b[{}m'.format
    colors = [COLORS[background] + 10,
              COLORS[foreground],
              COLORS['none'],
              COLORS['none'] + 10]
    if bright:
        colors[0] += 60
        colors[1] += 60
    color_codes = [sgr(c) for c in colors]
    return '{1[0]}{1[1]}{0}{1[2]}{1[3]}'.format(text, color_codes)


def warn(text: str,
         foreground: Color = 'w',
         background: Color = 'r',
         *,
         width: int | None = None,
         **kwargs) -> str:
    pad = ' ' * ((width or 0) - len(text))
    return highlight(text,
                     foreground=foreground,
                     background=background,
                     **kwargs) + pad


def info() -> None:
    """Show versions of GPAW and its dependencies."""
    results: list[tuple[str, str | bool]] = [
        ('python-' + sys.version.split()[0], sys.executable)]
    warnings = {}
    for name in ['gpaw', 'ase', 'numpy', 'scipy', 'gpaw_data']:
        try:
            module = import_module(name)
        except ImportError:
            results.append((name, False))
        else:
            # Search for git hash
            githash = search_current_git_hash(module)
            if githash is None:
                githash = ''
            else:
                githash = f'-{githash:.10}'
            results.append(
                (name + '-' + module.__version__ + githash,
                 module.__file__.rsplit('/', 1)[0] + '/'))  # type: ignore

    libs = gpaw.get_libraries()

    libxc = libs['libxc']
    if libxc:
        results.append((f'libxc-{libxc}', True))
    else:
        results.append(('libxc', False))
        warnings['libxc'] = ('GPAW not compiled with LibXC support; '
                             'though not a requirement, '
                             'it is recommended that LibXC be installed and '
                             'GPAW be recompiled with support therefor')

    if hasattr(cgpaw, 'githash'):
        githash = f'-{cgpaw.githash():.10}'
    else:
        githash = ''

    results.append(('_gpaw' + githash,
                    os.path.normpath(getattr(cgpaw._gpaw, '__file__',
                                             'built-in'))))

    results.append(('MPI enabled', have_mpi))
    results.append(('OpenMP enabled', cgpaw.have_openmp))
    results.append(('GPU enabled', GPU_ENABLED))
    results.append(('GPU-aware MPI', GPU_AWARE_MPI))
    cupy_version = 'cupy-' + cupy.__version__
    results.append((cupy_version, cupy.__file__))
    if cupy_is_fake and (GPU_ENABLED or GPU_AWARE_MPI):
        warnings[cupy_version] = ('GPAW compiled with GPU support, '
                                  'but the requisite CuPy is not found or '
                                  'cannot be set up (see gpaw.gpu at '
                                  f'{gpaw_gpu_filename!r}); '
                                  'GPU calculations will fail, '
                                  'unless the user explicitly set the '
                                  'environment variable GPAW_CPUPY=1, '
                                  'which uses GPAW\'s fake CuPy '
                                  '(gpaw.gpu.cpupy) for testing purposes')
    results.append(('MAGMA', cgpaw.have_magma))
    if have_mpi:
        have_sl = compiled_with_sl()
        have_elpa = LibElpa.have_elpa()
        if have_elpa:
            version = LibElpa.api_version()
            if version is None:
                version = 'unknown, at most 2018.xx'
            have_elpa = f'yes; version: {version}'
    else:
        have_sl = have_elpa = 'no (MPI unavailable)'

    if not hasattr(cgpaw, 'mmm'):
        results.append(('BLAS', 'using scipy.linalg.blas and numpy.dot()'))
        warnings['BLAS'] = ('GPAW not compiled with native BLAS support; '
                            'though not a requirement, '
                            'it is recommended that BLAS be installed and '
                            'GPAW be recompiled with support therefor')

    results.append(('scalapack', have_sl))
    results.append(('Elpa', have_elpa))

    have_fftw = fftw.have_fftw()
    results.append(('FFTW', have_fftw))
    results.append(('libvdwxc', compiled_with_libvdwxc()))

    for i, path in enumerate(gpaw.setup_paths):
        results.append((f'PAW-datasets ({i + 1})', str(path)))

    if rank != 0:
        return

    lines = [(a, b if isinstance(b, str) else ['no', 'yes'][b])
             for a, b in results]
    n1 = max(len(a) for a, _ in lines)
    n2 = max(len(b) for _, b in lines)
    output_width = n1 + 6 + n2
    box_edge = ' ' + '-' * (output_width - 2)
    print(box_edge)
    for a, b in lines:
        if a in warnings:
            a, b = warn(a, width=n1), warn(b, width=n2)
        else:
            a, b = f'{a:{n1}}', f'{b:{n2}}'
        print(f'| {a}  {b} |')
    print(box_edge)

    if not warnings:
        return
    warning_header = 'WARNING ({}):'.format
    header_width = max(len(warning_header(item)) for item in warnings) + 1
    for item, message in warnings.items():
        topic = 'WARNING ({}):'.format(item)
        message = fill(message,
                       initial_indent=' ' * header_width,
                       subsequent_indent=' ' * header_width,
                       width=output_width)
        print(warn(topic,
                   foreground='y',
                   background='k',
                   bright=False,
                   width=header_width),
              message[header_width:],
              sep='')


class CLICommand:
    """Show versions of GPAW and its dependencies"""

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        info()
