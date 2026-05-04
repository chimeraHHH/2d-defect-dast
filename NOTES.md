# Notes / future work

## GPAW + CUDA on RTX 5090 (sm_120)

**Status (2026-05-04):** all C18 DFT validation runs used **CPU-only** GPAW
(apt-installed `gpaw 22.1.0`). The rented 5090 GPU sat idle at 0 MiB / 7%
util while a single Python process saturated 15/16 CPU cores.

**Cost paid:** top-10 doped DFT 80 min + 5 pristine 29 min + 8 atomic μ
2-22 min + 7 bulk μ ~10 min ≈ **~140 min wall-clock** on a 16-core CPU.

**Estimated speedup with GPU:** for our PW(300 eV) PBE workload,
CUDA-enabled GPAW typically gives **3-10× wall-clock speedup**, dominated
by FFTs on the dense plane-wave grid. The 81-atom Mo2CO2 calcs (~20 min
each on CPU) are the most attractive targets — could drop to 3-5 min.

**Why not done now:** the apt build is the 2022 LTS, predates GPAW's
mature GPU backend. The path to a working GPU build:

1. Install CUDA toolkit ≥ 12.8 (sm_120 needs CUDA 12.8+; older toolkits
   silently downgrade to PTX, killing performance).
2. `pip install cupy-cuda12x` matching the toolkit minor version.
3. Build GPAW master from source with `--with-gpu` configure flag,
   linking against MPI + libxc + cuFFT + cuBLAS.
4. Verify `from gpaw import gpu` imports (this module is missing in
   22.1.0) and `mode=PW(...)` accepts `parallel={'gpu': True}`.
5. Sanity check: same single-point energy on a 27-atom test should
   match the CPU reference to within 1 meV.

**When to do this:** if/when a follow-up validation needs N≥50
candidates, a GPU build is mandatory — CPU time would scale to 12+
hours per batch. For the current N=10 paper §5.22, CPU is fine.

**Tracking:** no formal issue tracker; this note is the record. Update
or remove once a GPU build lands.
