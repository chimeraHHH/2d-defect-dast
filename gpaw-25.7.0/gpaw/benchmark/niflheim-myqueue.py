from gpaw.benchmark import benchmark_main, get_benchmarks
from pathlib import Path
from os import getcwd

platforms = [('xeon24el8_test', 24, 24, 0, '10m'),
             ('a100', 8, 1, 1, '1h'),
             ('a100', 128, 4, 4, '1h'),
             ('sm3090el8', 8, 1, 1, '1h'),
             ('epyc96', 96, 96, 0, '2h'),
             ('epyc96', 96 * 2, 96 * 2, 0, '4h')]


def workflow():
    from myqueue.workflow import run
    path = Path(getcwd())
    for partition, ncores, nprocs, ngpus, time in platforms:
        for benchmark in get_benchmarks(
                cores=nprocs, memory='10000G', gpus=ngpus):
            (path / f'{partition}-{nprocs}').mkdir(exist_ok=True)
            run(function=benchmark_main, args=(benchmark,),
                nodename=partition, cores=ncores, processes=nprocs,
                gpus=ngpus, tmax=time,
                name=f'{benchmark}-{partition}-{nprocs}',
                folder=f'{partition}-{nprocs}')
            if ngpus == 0:
                run(function=benchmark_main, args=(f'{benchmark}#old',),
                    nodename=partition, cores=ncores, processes=nprocs,
                    gpus=ngpus, tmax=time,
                    name=f'{benchmark}-{partition}-{nprocs}-old',
                    folder=f'{partition}-{nprocs}')
