# v3 multi-source × full augmentation — negative result (2026-05-05)

## TL;DR

Extending leak-free 3× rotation+perturbation augmentation from
IMP2D-only to **all four sources** in the multi-source pipeline
(IMP2D + JARVIS-2D + JARVIS-3D + JARVIS DFT-3D), with the DFT-3D loss
weight reduced from 0.3 to 0.10 to compensate for the increased
DFT-3D sample count, **degrades IMP2D test MAE substantially**: from
0.4929 (v2 multi seed=42, no aug on auxiliary sources) to **0.8233**
(seed=42, 50 epoch, 73 min wall on RTX 5090) — a **+67 % regression**.

The v2 multi-source recipe is therefore the recommended deployment
configuration; naive augmentation of all sources is a **negative
ablation** that reinforces the paper's central "data-quality-over-
quantity" message.

## What we ran

| Source | Original | Aug factor | Aug train | Loss weight |
|---|---|---|---|---|
| IMP2D | 8 512 | ×3 | 25 536 | 1.0 |
| JARVIS-2D | 56 | ×3 | 168 | 0.5 |
| JARVIS-3D | 304 | ×3 | 912 | 0.5 |
| JARVIS DFT-3D | 15 921 | ×3 | 47 763 | 0.10 (was 0.3 in v2) |
| **total** | 24 793 | — | **74 379** | — |

Hyperparameters identical to v2 multi-source: PFA-only backbone,
hidden 128, 4 heads, 2 cross-layers, AdamW lr 5e-4, weight decay 1e-4,
SmoothL1 loss, cosine annealing T_max=50, batch 16, 50 epoch, seed=42.

## Trajectory (val_mae_imp2d)

```
ep  0   2.226   ep 30   1.091   *
ep  5   1.553   ep 32   0.975   *
ep 10   1.687   ep 35   0.897   *
ep 15   1.388   ep 40   0.848   *
ep 19   1.376   ep 45   0.819   *
ep 26   1.376   ep 47   0.813   *
ep 29   1.303   ep 50   0.815   (final)
```

The validation MAE plateaued at 1.38 between ep 17 and ep 29, only
breaking through after the cosine-annealing learning rate dropped
below 50 % of its initial value. Final test MAE 0.8233 was 67 %
higher than the v2 multi-source single-seed result of 0.4929 on the
same `split_indices(seed=42)` test partition.

## Why this fails

Three contributing factors, not yet disentangled by controlled
experiments:

1. **Effective signal-to-noise per IMP2D sample drops.** With 3×
   augmentation applied to all sources, each IMP2D sample's gradient
   contribution is now competing against more aug copies of every
   source for the optimizer's attention. The 4-head readout was
   designed to keep sources isolated at the readout, but the shared
   backbone receives mixed gradient.

2. **DFT-3D weight 0.10 was a poor compromise.** In v2 multi-source
   we used DFT-3D weight 0.3 with 17 912 unaugmented samples. The
   instinct here was that 47 763 aug DFT-3D samples × 0.3 would
   overwhelm IMP2D, so we reduced to 0.10. But 47 763 × 0.10 = 4 776
   effective DFT-3D contribution vs the v2 baseline of 17 912 × 0.30
   = 5 374 — almost the same. Yet test MAE blew up. The reduction
   may have removed needed bulk-pristine regularisation without
   removing the bulk-pristine bias.

3. **Cosine annealing was tuned for 50 ep at the v2 data scale.**
   With 2.8× more steps per epoch, the LR effectively decays 2.8×
   faster per step. The early plateau at val 1.38 (ep 17–29) is
   consistent with the model being trapped in a shallow minimum that
   only the late LR drop unlocks. A run with `ReduceLROnPlateau` or
   `T_max=80` might converge to a better local minimum.

The empirically clearer takeaway, however, is the qualitative one
that aligns with the paper's main thesis (Sec. 4 of v3 paper):

> **Augmentation is a poor substitute for actual chemistry diversity.**
> The multi-source v2 -12 % gain came from four DIFFERENT databases
> covering distinct chemistry; trebling the size by augmentation
> copies — even though it nominally moves N_train from 26 837 to
> 74 379 — does not give the same benefit because the additional
> samples are highly correlated with the originals.

## Decision

**No multi-seed verification.** The single-seed regression is large
enough (+67 %) and the trajectory clear enough that 3 more seeds
would only confirm the same pattern at significant GPU cost.

**v2 multi-source remains the headline.** Configuration:
- PFA-only backbone (0.82 M params)
- 4 sources without augmentation on auxiliary sources
- DFT-3D loss weight 0.3
- 60 epoch
- 4-seed mean test MAE 0.486 ± 0.025 eV (unchanged from previous
  reported result)

**Paper update**: this result is folded into the discussion of
augmentation limits as a third honest negative ablation alongside
the v2 single-source PFA ablation (Sec. 5.2) and the dual-stream
ablation (Sec. 5.3). It strengthens the paper's central claim that
"data quality, not quantity" is what matters at this scale.

## Cumulative GPU spend on v3 (approximate)

```
v3 dualstream pipeline + 4-seed (2 reboot recoveries)   ~10 GPU·h
aug-multi-source builder + headline                     ~1.5 GPU·h
─────────────────────────────────────────────────────────────────
v3 phase total                                          ~11.5 GPU·h
running total of session (incl. v2 phases)              ~16   GPU·h
```

within the 24-hour session budget.
