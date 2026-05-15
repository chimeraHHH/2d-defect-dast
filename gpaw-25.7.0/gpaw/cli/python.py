import argparse
import runpy
import sys


class CLICommand:
    """Run GPAW's Python interpreter."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            '--dry-run', '-z', type=int, default=0,
            metavar='NCPUS',
            help='Dry run on NCPUS cpus.')
        parser.add_argument(
            '-d', '--debug', action='store_true',
            help='Run in debug-mode.')

        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--command', '-c',
            dest='cmd',
            nargs=argparse.REMAINDER,
            help='Program passed in as string (terminates option list).')
        group.add_argument(
            '--module', '-m',
            nargs=argparse.REMAINDER,
            help='Run library module as a script (terminates option list).')

        parser.add_argument(
            'arguments', metavar='ARG',
            help='Arguments passed to program in '
            'sys.argv[1:].  '
            'Use -- to force all remaining arguments to be '
            'passed to target program (useful for passing '
            'options starting with -).',
            nargs='*')

    @staticmethod
    def run(args):
        from gpaw import all_lazy_imports, broadcast_imports, __getattr__
        with broadcast_imports:
            for attr in all_lazy_imports:
                __getattr__(attr)

        if args.cmd:
            sys.argv[:] = ['-c'] + args.cmd[1:]
            d = {}
            exec(args.cmd[0], d, d)
        elif args.module:
            sys.argv[:] = args.module
            runpy.run_module(args.module[0],
                             run_name='__main__',
                             alter_sys=True)
        else:
            sys.argv[:] = args.arguments
            runpy.run_path(args.arguments[0], run_name='__main__')
