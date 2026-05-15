# Copyright (C) 2006 CSC-Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.
import os
import sys
import re
import shlex
from sysconfig import get_platform
from subprocess import run
from pathlib import Path
from stat import ST_MTIME


def mtime(path, name, mtimes):
    """Return modification time.

    The modification time of a source file is returned.  If one of its
    dependencies is newer, the mtime of that file is returned.
    This function fails if two include files with the same name
    are present in different directories."""

    include = re.compile(r'^#\s*include "(\S+)"', re.MULTILINE)

    if name in mtimes:
        return mtimes[name]
    t = os.stat(os.path.join(path, name))[ST_MTIME]
    for name2 in include.findall(open(os.path.join(path, name)).read()):
        path2, name22 = os.path.split(name2)
        if name22 != name:
            t = max(t, mtime(os.path.join(path, path2), name22, mtimes))
    mtimes[name] = t
    return t


def check_dependencies(sources):
    # Distutils does not do deep dependencies correctly.  We take care of
    # that here so that "python setup.py build_ext" always does the right
    # thing!
    mtimes = {}  # modification times

    # Remove object files if any dependencies have changed:
    plat = get_platform() + '-{maj}.{min}'.format(maj=sys.version_info[0],
                                                  min=sys.version_info[1])
    remove = False
    for source in sources:
        path, name = os.path.split(source)
        t = mtime(path + '/', name, mtimes)
        o = 'build/temp.%s/%s.o' % (plat, source[:-2])  # object file
        if os.path.exists(o) and t > os.stat(o)[ST_MTIME]:
            print('removing', o)
            os.remove(o)
            remove = True

    so = 'build/lib.{}/_gpaw.so'.format(plat)
    if os.path.exists(so) and remove:
        # Remove shared object C-extension:
        # print 'removing', so
        os.remove(so)


def write_configuration(define_macros, include_dirs, libraries, library_dirs,
                        extra_link_args, extra_compile_args,
                        runtime_library_dirs, extra_objects, compiler):

    # Write the compilation configuration into a file
    try:
        out = open('configuration.log', 'w')
    except IOError as x:
        print(x)
        return
    print("Current configuration", file=out)
    print("compiler", compiler, file=out)
    print("libraries", libraries, file=out)
    print("library_dirs", library_dirs, file=out)
    print("include_dirs", include_dirs, file=out)
    print("define_macros", define_macros, file=out)
    print("extra_link_args", extra_link_args, file=out)
    print("extra_compile_args", extra_compile_args, file=out)
    print("runtime_library_dirs", runtime_library_dirs, file=out)
    print("extra_objects", extra_objects, file=out)
    out.close()


def build_interpreter(
        compiler, extension, extension_objects, *,
        link_extra_preargs, link_extra_postargs,
        build_temp, build_bin, debug):
    exename = compiler.executable_filename('gpaw-python')
    print(f'building {repr(exename)} executable', flush=True)

    macros = extension.define_macros.copy()
    for undef in extension.undef_macros:
        macros.append((undef,))

    # Compile the sources that define GPAW_INTERPRETER
    sources = ['c/main.c']
    objects = compiler.compile(
        sources,
        output_dir=str(build_temp),
        macros=macros,
        include_dirs=extension.include_dirs,
        debug=debug,
        extra_postargs=extension.extra_compile_args)
    objects += extension_objects

    # Link the custom interpreter
    compiler.link_executable(
        objects, exename,
        output_dir=str(build_bin),
        extra_preargs=link_extra_preargs,
        libraries=extension.libraries,
        library_dirs=extension.library_dirs,
        runtime_library_dirs=extension.runtime_library_dirs,
        extra_postargs=link_extra_postargs + extension.extra_link_args,
        debug=debug,
        target_lang=extension.language)
    return build_bin / exename


def build_gpu(gpu_compiler, gpu_compile_args, gpu_include_dirs,
              define_macros, undef_macros, build_temp):
    print('building gpu kernels', flush=True)

    kernels_dpath = Path('c/gpu/kernels')

    # Create temp build directory
    build_temp_kernels_dpath = build_temp / kernels_dpath
    if not build_temp_kernels_dpath.exists():
        print(f'creating {build_temp_kernels_dpath}', flush=True)
        build_temp_kernels_dpath.mkdir(parents=True)

    # Glob all kernel files, but remove those included by other kernels
    kernels = sorted(kernels_dpath.glob('*.cpp'))
    for name in ['interpolate-stencil.cpp',
                 'lfc-reduce.cpp',
                 'lfc-reduce-kernel.cpp',
                 'reduce.cpp',
                 'reduce-kernel.cpp',
                 'restrict-stencil.cpp']:
        kernels.remove(kernels_dpath / name)

    # Compile GPU kernels
    objects = []
    for src in kernels:
        obj = build_temp / src.with_suffix('.o')
        objects.append(str(obj))
        run_args = [gpu_compiler]
        run_args += gpu_compile_args
        for (name, value) in define_macros:
            arg = f'-D{name}'
            if value is not None:
                arg += f'={value}'
            run_args += [arg]
        run_args += [f'-U{name}' for name in undef_macros]
        run_args += [f'-I{dpath}' for dpath in gpu_include_dirs]
        run_args += ['-c', str(src)]
        run_args += ['-o', str(obj)]
        print(shlex.join(run_args), flush=True)
        p = run(run_args, check=False, shell=False)
        if p.returncode != 0:
            print(f'error: command {repr(gpu_compiler)} failed '
                  f'with exit code {p.returncode}',
                  file=sys.stderr, flush=True)
            sys.exit(1)

    return objects
