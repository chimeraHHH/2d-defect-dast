import subprocess
import sys
import os

usage = """gpaw sbatch [-0] -- [sbatch options] script.py [script options]
   or: gpaw sbatch [-0] -- [sbatch options] python -m module [module options]
"""


class CLICommand:
    """Submit a GPAW Python script via sbatch.


   If a virtual environment is active when submitting, it will be activated for
   the job as well.
   """

    @staticmethod
    def add_arguments(parser):
        parser.usage = usage
        parser.add_argument('-0', '--test', action='store_true',
                            help='Dry run: Print driver script.')
        parser.add_argument('arguments', nargs='*')

    @staticmethod
    def run(args):
        script = '#!/bin/bash -l\n'
        for i, arg in enumerate(args.arguments):
            if arg.endswith('.py'):
                break
        else:
            for i, arg in enumerate(args.arguments):
                if (arg.startswith('python') and
                    len(args.arguments) > i + 1 and
                    args.arguments[i + 1].startswith('-m')):
                    del args.arguments[i]
                    break
            else:
                print('No script.py found!', file=sys.stderr)
                return

        if arg.endswith('.py'):
            for line in open(arg):
                if line.startswith('#SBATCH'):
                    script += line
        script += ('cd $SLURM_SUBMIT_DIR\n')
        venv = os.getenv('VIRTUAL_ENV')
        if venv:
            print('Detected virtual environment:', venv)
            script += f'source {venv}/bin/activate\n'
        script += ('OMP_NUM_THREADS=1 '
                   'mpiexec `echo $GPAW_MPI_OPTIONS` gpaw python {}\n'
                   .format(' '.join(args.arguments[i:])))
        cmd = ['sbatch'] + args.arguments[:i]
        if args.test:
            print('sbatch command:')
            print(' '.join(cmd))
            print('\nscript:')
            print(script, end='')
        else:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, env=os.environ)
            p.communicate(script.encode())
