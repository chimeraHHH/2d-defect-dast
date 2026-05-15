# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import sys
import time
import math

import numpy as np
from ase.utils.timing import Timer

import gpaw.mpi as mpi


class NullTimer:
    """Compatible with Timer and StepTimer interfaces.  Does nothing."""
    def __init__(self):
        pass

    def print_info(self, calc):
        pass

    def start(self, name):
        pass

    def stop(self, name=None):
        pass

    def get_time(self, name):
        return 0.0

    def write(self, out=sys.stdout):
        pass

    def write_now(self, mark=''):
        pass

    def add(self, timer):
        pass

    def __call__(self, name):
        return self

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


nulltimer = NullTimer()


class DebugTimer(Timer):
    def __init__(self, print_levels=1000, comm=mpi.world, txt=sys.stdout):
        Timer.__init__(self, print_levels)
        ndigits = 1 + int(math.log10(comm.size))
        self.srank = '%0*d' % (ndigits, comm.rank)
        self.txt = txt

    def start(self, name):
        Timer.start(self, name)
        abstime = time.time()
        t = self.timers[tuple(self.running)] + abstime
        self.txt.write('T%s >> %15.8f %s (%7.5fs) started\n'
                       % (self.srank, abstime, name, t))

    def stop(self, name=None):
        if name is None:
            name = self.running[-1]
        abstime = time.time()
        t = self.timers[tuple(self.running)] + abstime
        self.txt.write('T%s << %15.8f %s (%7.5fs) stopped\n'
                       % (self.srank, abstime, name, t))
        Timer.stop(self, name)


class GPUEvent:
    def __init__(self, name):
        self.name = name
        import cupy
        default = dict(block=False,
                       disable_timing=False,
                       interprocess=False)
        self.stop_event = cupy.cuda.Event(**default)
        self.start_event = cupy.cuda.Event(**default)
        self.start_event.record()

    def stop(self):
        self.stop_event.record()

    def get_time(self):
        import cupy
        return cupy.cuda.get_elapsed_time(self.start_event,
                                          self.stop_event) / 1000


class GPUTimerBase:

    def __init__(self, max_stack=10):
        self.event_queue = []
        self.event_stack = []
        self.max_stack = max_stack
        from collections import defaultdict
        self.gpu_timers = defaultdict(float)

    def gpu_start(self, key):
        self.event_stack.append(GPUEvent(key))

    def gpu_stop(self):
        gpu_event = self.event_stack.pop()
        gpu_event.stop()
        self.event_queue.append(gpu_event)
        if len(self.event_queue) > self.max_stack:
            self.handle_events()

    def handle_events(self):
        while len(self.event_queue):
            event = self.event_queue[0]
            if not event.stop_event.done:
                break
            del self.event_queue[0]
            time = event.get_time()
            self.gpu_timers[event.name] += time
            self.handle_event_hook(event)

    def handle_event_hook(self, event):
        pass

    def gpu_write(self, out=sys.stdout):
        import cupy
        event = cupy.cuda.Event(block=True)
        event.synchronize()
        self.handle_events()

        timers, self.timers = self.timers, self.gpu_timers
        Timer.write(self, out)
        self.timers = timers


class GPUTimer(Timer, GPUTimerBase):
    def __init__(self, *args, **kwargs):
        Timer.__init__(self, *args, **kwargs)
        GPUTimerBase.__init__(self)

    def start(self, name):
        Timer.start(self, name)
        GPUTimerBase.gpu_start(self, name)

    def stop(self, name=None):
        Timer.stop(self, name)
        GPUTimerBase.gpu_stop(self)

    def write(self, out=sys.stdout):
        print('CPU event timings:', file=out)
        Timer.write(self, out)
        print('GPU event timings:', file=out)
        GPUTimerBase.gpu_write(self, out)


def ranktxt(comm, rank=None):
    rank = comm.rank if rank is None else rank
    ndigits = len(str(comm.size - 1))
    return '%0*d' % (ndigits, rank)


class ParallelTimer(DebugTimer):
    """Like DebugTimer but writes timings from all ranks.

    Each rank writes to timings.<rank>.txt.  Also timings.metadata.txt
    will contain information about the parallelization layout.  The idea
    is that the output from this timer can be used for plots and to
    determine bottlenecks in the parallelization.

    See the tool gpaw-plot-parallel-timings."""
    def __init__(self, prefix='timings', flush=False):
        fname = f'{prefix}.{ranktxt(mpi.world)}.txt'
        txt = open(fname, 'w', buffering=1 if flush else -1)
        DebugTimer.__init__(self, comm=mpi.world, txt=txt)
        self.prefix = prefix

    def print_info(self, calc):
        """Print information about parallelization into a file."""
        fd = open('%s.metadata.txt' % self.prefix, 'w')
        DebugTimer.print_info(self, calc)
        wfs = calc.wfs

        # We won't have to type a lot if everyone just sends all their numbers.
        myranks = np.array([wfs.world.rank, wfs.kd.comm.rank,
                            wfs.bd.comm.rank, wfs.gd.comm.rank])
        allranks = None
        if wfs.world.rank == 0:
            allranks = np.empty(wfs.world.size * 4, dtype=int)
        wfs.world.gather(myranks, 0, allranks)
        if wfs.world.rank == 0:
            for itsranks in allranks.reshape(-1, 4):
                fd.write('r=%d k=%d b=%d d=%d\n' % tuple(itsranks))
        fd.close()


class Profiler(Timer):
    def __init__(self, prefix, comm=mpi.world):
        import atexit

        self.prefix = prefix
        self.comm = comm
        self.ranktxt = ranktxt(comm)
        fname = f'{prefix}.{self.ranktxt}.json'
        self.txt = open(fname, 'w', buffering=-1)
        self.pid = 0  # os.getpid() creates more confusing output
        atexit.register(Profiler.finish_trace, self)
        # legacy json format for perfetto always assumes microseconds
        self.u = 1_000_000

        self.synchronize()
        Timer.__init__(self, 1000)

    def synchronize(self):
        # Synchronize in order to have same time reference
        ref = np.zeros(1)
        if self.comm.rank == 0:
            ref[0] = time.time()
        self.comm.broadcast(ref, 0)
        self.ref = ref[0]

    def finish_trace(self):
        self.txt.close()
        self.comm.barrier()
        if self.comm.rank == 0:
            out = open(self.prefix + '.json', 'w')
            out.write("""{
    "traceEvents":
    [ """)
            for i in range(self.comm.size):
                fname = f'{self.prefix}.{ranktxt(self.comm, rank=i)}.json'
                print('Processing', fname)
                with open(fname, 'r') as f:
                    out.writelines(f.readlines())
            out.write("] }\n")
            out.close()
        self.comm.barrier()

    def start(self, name):
        Timer.start(self, name)
        self.txt.write(
            f"""{{"name": "{name}", "cat": "PERF", "ph": "B","""
            f""" "pid": {self.pid}, "tid": {self.ranktxt}, """
            f""""ts": {int((time.time() - self.ref) * self.u)} }},\n""")

    def stop(self, name=None):
        if name is None:
            name = self.running[-1]
        self.txt.write(
            f"""{{"name": "{name}", "cat": "PERF", "ph": "E", """
            f""""pid": {self.pid}, "tid": {self.ranktxt}, """
            f""""ts": {int((time.time() - self.ref) * self.u)}}},\n""")
        Timer.stop(self, name)


class GPUProfiler(Profiler, GPUTimerBase):
    def __init__(self, prefix, comm=mpi.world):
        Profiler.__init__(self, prefix, comm=comm)
        GPUTimerBase.__init__(self)

    def synchronize(self):
        from cupy.cuda import Event
        # Make sure GPU gets here
        event = Event(block=True)
        event.record()
        event.synchronize()

        # Wait all CPUs
        self.comm.barrier()

        # Now all GPUs and CPUs are somewhat simultaneous
        # So, record the reference event and time
        event = Event(block=True)
        event.record()
        self.ref = time.time()
        self.ref_event = event

        # Broadcast CPU time reference (possibly problematic)
        buf = np.zeros(1)
        buf[0] = self.ref
        self.comm.broadcast(buf, 0)
        self.ref = buf[0]

    def start(self, name, gpu=False):
        Profiler.start(self, name)
        if gpu:
            GPUTimerBase.gpu_start(self, name)

    def stop(self, name=None, gpu=False):
        Profiler.stop(self, name)
        if gpu:
            GPUTimerBase.gpu_stop(self)

    def handle_event_hook(self, event):
        import cupy

        def get_time(e):
            t = cupy.cuda.get_elapsed_time
            return t(self.ref_event, e)

        ms_start = get_time(event.start_event)
        ms_stop = get_time(event.stop_event)
        self.txt.write(
            f"""{{"name": "{event.name}", "cat": "PERF", "ph": "B","""
            f""" "pid": {self.pid}, "tid": "GPU {self.ranktxt}", """
            f""""ts": {int(ms_start * 1000)} }},\n""")
        self.txt.write(
            f"""{{"name": "{event.name}", "cat": "PERF", "ph": "E", """
            f""""pid": {self.pid}, "tid": "GPU {self.ranktxt}", """
            f""""ts": {int(ms_stop * 1000)} }},\n""")


class HPMTimer(Timer):
    """HPMTimer requires installation of the IBM BlueGene/P HPM
    middleware interface to the low-level UPC library. This will
    most likely only work at ANL's BlueGene/P. Must compile
    with GPAW_HPM macro in customize.py. Note that HPM_Init
    and HPM_Finalize are called in cgpaw.c and not in the Python
    interface. Timer must be called on all ranks in node, otherwise
    HPM will hang. Hence, we only call HPM_start/stop on a list
    subset of timers."""

    top_level = 'GPAW.calculator'  # HPM needs top level timer
    compatible = ['Initialization', 'SCF-cycle']

    def __init__(self):
        Timer.__init__(self)
        from gpaw.cgpaw import hpm_start, hpm_stop
        self.hpm_start = hpm_start
        self.hpm_stop = hpm_stop
        hpm_start(self.top_level)

    def start(self, name):
        Timer.start(self, name)
        if name in self.compatible:
            self.hpm_start(name)

    def stop(self, name=None):
        Timer.stop(self, name)
        if name in self.compatible:
            self.hpm_stop(name)

    def write(self, out=sys.stdout):
        Timer.write(self, out)
        self.hpm_stop(self.top_level)


class CrayPAT_timer(Timer):
    """Interface to CrayPAT API. In addition to regular timers,
    the corresponding regions are profiled by CrayPAT. The gpaw-python has
    to be compiled under CrayPAT.
    """

    def __init__(self, print_levels=4):
        Timer.__init__(self, print_levels)
        from gpaw.cgpaw import craypat_region_begin, craypat_region_end
        self.craypat_region_begin = craypat_region_begin
        self.craypat_region_end = craypat_region_end
        self.regions = {}
        self.region_id = 5  # leave room for regions in C

    def start(self, name):
        Timer.start(self, name)
        if name in self.regions:
            id = self.regions[name]
        else:
            id = self.region_id
            self.regions[name] = id
            self.region_id += 1
        self.craypat_region_begin(id, name)

    def stop(self, name=None):
        Timer.stop(self, name)
        id = self.regions[name]
        self.craypat_region_end(id)
