# Notes / future work

## Batch 70-89 recovered from raw server backup (2026-07-06)

原始服务器数据 (`backup_data 2/`, ~420MB) 找回后, 用
`generate_tmd_summary_70_89.py` 从 GPAW 原始输出提取了第 70-89 批
(new_tmd_samples_20.db, VS2/CrS2 × K/Mg/Ca/Sc/Ti/Cr(V)/Mn/Fe/Cu/Zn)
的形成能: **16/20 完成**, 见
[tmd_samples_70_89_summary.md](tmd_samples_70_89_summary.md)。

- 未完成: #77/#87 (Fe, 缺 mu_Fe), #88/#89 (CrS2+Cu/Zn, defect 未跑完)。
  按决定不再重算。
- #74/#79/#86 仅 stage1 收敛 (精度略低, 已标注)。
- #81 (CrS2+Mg) E_form=+5.28 eV 异常偏高, defect 能量 -309.66 明显高于
  host -313.32, 疑似 SCF 收敛到高能态, 使用前需人工复核
  `gpaw_sample_81_defect_stage2.txt`。
- `backup_data 2/` 含 6 个 ASE .db (含 imp2d.db 全量 17364 条 +
  filtered 10885 条)、35 种元素 mu 参考计算、全部 sample 0-86 的
  GPAW 日志与 traj, 是唯一原始数据副本, 勿删。

## v2 architecture (2026-05-04)

The v2 work introduces three additive attention-bias channels (no new tokens,
to avoid the DAST virtual-node failure mode of v1):

- ``PeriodicFourierBias`` (`src/models/attention_v2.py`): integer-k truncated
  Fourier basis on minimum-image fractional displacement; exact periodicity
  under any lattice shift, encodes direction the scalar distance discards.
- ``MultiScaleDistanceBias``: short-range Gaussian RBF + long-range
  ``[1/(r+δ)^n, exp(-r/λ)]`` with smooth cut-offs at ``r_short`` and ``r_max``.
- ``DefectAwareBias``: 4-entry categorical bias keyed on
  (defect_i, defect_j) ∈ {0,1}².

Single-source ablation (h=128, 50ep, leak-free aug, seed=42) showed all 5
combinations land in 0.519–0.551 eV — within 2σ of baseline 0.516 — so the
single-source PFA path is **net-neutral**. The win comes from combining the
PFA backbone with **multi-source training** (4 DBs × per-source readout
heads): seed=42 gives test MAE 0.4929 eV (−11.2% vs v1 multi-source 0.555);
4-seed mean 0.486 ± 0.025 eV.

**Important caveat**: multi-source uses ``split_indices(seed=N)`` for IMP2D
test, which differs from the leak-free 1065 sample set used by single-source
baselines. Apples-to-apples comparison only holds against v1 multi-source.
A leak-free re-evaluation of v2 multi-source is on the to-do list.

## v2 multi-source LOHO — incomplete; ID/OOD trade-off discovered (2026-05-04)

The 5-host LOHO run was cancelled at 3/5 (MoS₂, MoSSe, TaSe₂ done; Cr₂I₆
in-progress and C₂H₂ not yet started). The completed hosts show that
v2-multi-source **degrades** leave-one-host-out test MAE by 22-65% vs the
v1-single-source LOHO baselines, with a 1.32-1.76× val→test gap, signalling
poor OOD generalisation. The hypothesis is that the ~18 k JARVIS DFT-3D
bulk-pristine samples in the multi-source corpus pull the shared backbone
toward a 3D-bulk representation that doesn't transfer to held-out 2D hosts.

The fair comparison for v2-multi-LOHO is **v2-single-LOHO** (using the
already-prepared `configs/v2_loho_*.yaml`) and/or **v1-multi-LOHO** — neither
exists on disk yet. Until those controls are run, the v2-multi-LOHO numbers
are treated as a methodological observation rather than a publishable
result. See [results/V2_LOHO_FINDINGS.md](results/V2_LOHO_FINDINGS.md) for
the full record + cancellation receipt.

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
