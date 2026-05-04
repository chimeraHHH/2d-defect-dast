# v2 multi-source LOHO — partial run + decision to halt (2026-05-04)

## TL;DR

We ran 3 of the 5 hosts in the v2-PFA-multi-source LOHO queue and
cancelled the remaining two (Cr₂I₆ in-progress at ep 16/50 and C₂H₂
not yet started). The cancellation was made on the basis that the
3 completed hosts already showed an unambiguous and reproducible
**ID/OOD trade-off**: multi-source training, while it improved the
in-distribution test MAE by 11.2% (0.555 → 0.4929 eV at seed=42),
**degraded** the leave-one-host-out test MAE by 22-65% across all 3
completed hosts. This is a substantive negative result that needs to
be re-investigated with fair single-source vs multi-source LOHO
comparisons before any further GPU is spent.

## Numbers

| host  | n_test | v2_multi_test | v1_single_test | Δ vs v1-single | best_val | val→test gap |
|-------|--------|---------------|----------------|----------------|----------|--------------|
| MoS₂  | 308    | 0.8496        | 0.5163         | **+64.6%**     | 0.4817   | 1.76×        |
| MoSSe | 464    | 0.7280        | 0.4479         | **+62.5%**     | 0.5517   | 1.32×        |
| TaSe₂ | 222    | 0.7738        | 0.6364         | **+21.6%**     | 0.5620   | 1.38×        |
| Cr₂I₆ | —      | (cancelled)   | 0.8401         | —              | —        | —            |
| C₂H₂  | —      | (cancelled)   | 2.3987         | —              | —        | —            |

The val→test gap of 1.32–1.76× indicates that the model converges to a
representation that fits the train + val distribution (non-host
samples) but transfers poorly to the held-out host.

## Root-cause hypothesis (untested)

The multi-source pipeline mixes IMP2D (8 512 train samples × 3 aug ≈
25 k 2D-defect samples) with JARVIS-2D (70), JARVIS-3D (381), and
**JARVIS DFT-3D (17 912 bulk pristine 3D crystals)**. The shared
backbone receives ~40 % of its gradient signal from the DFT-3D head,
whose chemistry (mean Ef = −0.82 eV/atom; 3D bulk crystals) is
fundamentally different from the IMP2D LOHO test domain (2D defect
supercells with held-out host chemistry). Three observations support
this hypothesis:

1. The training is stable and val_mae converges (0.48–0.56 across
   hosts), so the issue is not optimisation noise.
2. The val→test gap appears specifically in OOD evaluation; the
   in-distribution test (4-seed mean 0.486 ± 0.025) is competitive.
3. The degradation correlates inversely with chemical similarity
   between held-out host and training distribution: TaSe₂ (Group V/VI
   transition-metal dichalcogenide, well-represented in training)
   degrades only 22 %, while MoS₂ (Group VI, also well-represented but
   with a notably distinct vacancy chemistry) degrades 65 %.

## What we have NOT done yet (and why we should)

The "fair" comparison for v2 multi-source LOHO is **not** v1
single-source LOHO (which is what we have on file). It is one of:

- **v2 single-source LOHO** (use existing
  `configs/v2_loho_*.yaml`, 5×~50 min). Apples-to-apples for "what
  does PFA buy us at the LOHO data scale".
- **v1 multi-source LOHO** (5×~35 min). Apples-to-apples for "is the
  ID/OOD trade-off intrinsic to the multi-source recipe, or specific
  to the v2 backbone".

Either or both are required before drawing publishable conclusions.
For now we **freeze** the multi-source LOHO at 3/5 hosts and treat the
finding as an unresolved methodological observation rather than a
headline result.

## Corollary for the paper

Until the controls above are run, paper §5.4 (LOHO) cannot be filled
in with the v2 multi-source numbers as a positive result. The
cleanest framing once data is available is:

> *"Multi-source joint training, while it improves in-distribution
> accuracy by ~11%, comes at the cost of out-of-distribution
> generalisation: leave-one-host-out test MAE degrades by 20–65% on
> three 2D hosts. This is a non-trivial trade-off for high-throughput
> deployment and motivates a single-source backbone in OOD-critical
> applications."*

## Cancellation receipt

- Run started: 2026-05-04 13:28 UTC (queue PID 3898, python PID 4789)
- Run cancelled: 2026-05-04 15:33 UTC (manual SIGTERM + SIGKILL)
- GPU usage at cancellation: 7 %, 0 MiB (clean idle)
- Cumulative GPU time spent on this LOHO run: ~125 minutes (3 finished
  hosts at ~35 min each + 16 epochs of Cr₂I₆ at ~42 s/ep ≈ 11 min).
- Files preserved: `results/v2_loho_multi_{MoS2, MoSSe, TaSe2}.json`,
  `results/v2_loho_multi_{MoS2, MoSSe, TaSe2}/best.pt`,
  `results/v2_loho_multi_queue.log`.
- Files NOT created: `results/v2_loho_multi_Cr2I6.json`,
  `results/v2_loho_multi_C2H2.json`,
  `results/v2_loho_multi_queue.done`.
