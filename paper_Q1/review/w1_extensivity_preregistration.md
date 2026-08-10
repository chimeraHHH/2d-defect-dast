# W1 preregistration: extensivity diagnostic for the readout conclusion

Frozen: 2026-08-10, before any W1 outcome was computed or viewed.
Milestone: `J-R3-physics-error-map` (W1 companion to the G3 preregistration;
kept as a separate document because its inputs are archived SchNet
predictions, not the DART OOF archives the G3 CLI consumes).

## Hypothesis under test

The impurity formation energy is a localized, non-extensive quantity (a
difference of extensive totals). If that framing is correct:

- **H-sum**: the signed OOF residual of the additive-readout SchNet has a
  nonzero slope in system size;
- **H-mean**: the mean-readout arm has a near-zero size slope in the signed
  residual, while its absolute error may grow with size through dilution of
  the defect signal;
- **H-DART**: the defect-centered readout has a near-zero size slope.

These are hypotheses to be tested, not conclusions. If H-sum fails, the
manuscript keeps C3 as a purely empirical recipe result and does not use the
extensivity explanation.

## Frozen inputs

- Archived, hash-verified additive-readout SchNet predictions from the
  completed 48-run campaign (host-CV primary; pair-CV as sensitivity where
  archived);
- the archived 15-run mean-readout host-CV sensitivity arm;
- DART OOF predictions: legacy archives for the exploratory tier; repaired G2
  OOF for the canonical tier;
- the canonical protocol-v2 sample table for atom counts (total and
  host-only) and target magnitudes.

Primary panel: host-CV, the only protocol where all three readout arms exist.

## Frozen analysis

For each arm, a Huber regression of the signed OOF residual on total atom
count, pooled over folds with fold indicators; host-clustered bootstrap with
2,000 draws and seed 20260810. Secondary regressor: host-only atom count.
Sensitivity: adjust for target magnitude; and the same regressions on
absolute error to expose the mean-readout dilution term. No other outcomes,
subgroups, or transformations may be added after results are seen.

## Frozen gates

- The extensivity explanation is "established" only when: the additive-arm
  size slope's 95% clustered interval excludes zero; its sign agrees in at
  least four of five host folds; and the additive-minus-DART slope-difference
  interval excludes zero.
- The mean-readout panel is reported descriptively (signed slope plus
  absolute-error trend) and carries no standalone verdict.
- A failed gate is reported as "not established", never dropped.

## Tier and precondition

- **Pipeline-independence audit (precondition for any canonical SchNet
  panel):** one bounded code audit confirming the SchNet input pipeline
  shares neither the 32-pair ordered triplet cap nor the componentwise
  fractional wrapping of the DART builder. The audit report is committed
  before any W1 outcome is viewed. If shared code is found, all SchNet panels
  drop to exploratory until G2.
- The DART panel is exploratory (legacy graph) until an accepted
  `prm_g2_acceptance_v1` exists; it is then recomputed from the repaired OOF
  before any manuscript use. Legacy output carries the standard watermark and
  stays outside paper-facing directories.

## Implementation

A new script `scripts/prm_w1_extensivity.py` with unit tests, run on the
accepted server profile from a pushed clean commit; CPU-only with an empty
CUDA mask. The joined table construction reuses the G3 provenance join rules
(audited `sample_index`, hash-checked inputs); every output is
exclusive-create with a manifest of input hashes.
