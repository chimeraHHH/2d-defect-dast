# EV-A preregistration: convention-anchor reproduction

Frozen: 2026-08-10
Milestone: `J-EV-A-convention-anchors`
Authorization: **granted 2026-08-10 for six anchor DFT runs only.** EV-B/C/D
(the blind candidate campaign) is a separate unapproved gate.

## Purpose and boundary

EV-A tests exactly one thing: whether the team can reproduce the IMP2D
formation-energy convention on structures whose database values are known. It
is a calibration of the DFT pipeline, stated openly as such. Anchor labels are
known in advance, so EV-A is not blind and supports no model claim, no
external-validity claim, and no acceleration claim. Its only admissible
outcomes are "convention reproduced within the frozen tolerance" or
"convention mismatch; EV stops".

## Frozen anchor selection rule

Population: the 10,224 canonical rows. Anchors are selected deterministically,
before any DFT input is written, by the following rule:

1. Strata, in frozen order: (chalcogenide, adsorbate), (chalcogenide,
   interstitial), (carbide/MXene-like, adsorbate), (carbide/MXene-like,
   interstitial), (elemental/hydrogenated, adsorbate),
   (elemental/hydrogenated, interstitial).
2. Within each stratum, exclude pathological rows by the exact P4
   non-pathological rule (finite `XF<=2`, finite `|conv2|`, not in the frozen
   103-row top-1% `|conv2|` set).
3. Restrict to rows whose target lies within the stratum interquartile range
   and whose total atom count lies within the stratum interquartile range, so
   anchors are typical, bounded-cost cases.
4. Select the row with the lowest `sample_index`, skipping any row whose host
   or impurity identity was already used by an earlier stratum (iterate
   ascending `sample_index`).
5. If a stratum empties, draw from the frozen fallback stratum order
   (halide/halochalcogenide, adsorbate) then (halide/halochalcogenide,
   interstitial). Fewer than six anchors fails EV-A closed; the rule is never
   relaxed by hand.

The selected `sample_index` values, identities, and the full elimination
ledger are committed before any DFT run.

## Frozen computation contract

- One DFT code and one resolved settings file reproducing the documented
  IMP2D workflow: functional, pseudopotential/PAW set, plane-wave cutoff,
  k-mesh density rule, vacuum thickness, smearing, spin treatment,
  relaxation and electronic convergence thresholds, and the formation-energy
  reference convention (host reference and impurity chemical-potential terms)
  as defined after Eq. (1) of the manuscript.
- The resolved settings file is committed before the first run; it is never
  edited per anchor.
- All inputs, outputs, convergence logs, and wall times are archived; a
  failed or unconverged relaxation is retained and reported, never resampled.
- Runs execute on the accepted server profile from a pushed clean commit;
  receipts carry aliases and hashes only.

## Frozen acceptance tolerances

Over the six anchor formation energies, comparing reproduced values against
the database values:

- median absolute difference at most 0.10 eV;
- maximum absolute difference at most 0.20 eV;
- every relaxation converged under the frozen force and energy criteria.

Rationale, fixed now: the repaired-model evaluation this gate ultimately
serves reports errors near 1 eV on hard transfer axes, so the convention
check must be several times tighter to be informative, and about 0.1 eV is
the typical cross-setup reproducibility of defect formation energies.

Any tolerance failure is reported as a convention mismatch. In that case EV-B
is not requested, and the manuscript states that external validation was
attempted at the convention-anchor stage and stopped there. Tolerances are
never widened after results exist.

## Outputs

- `artifacts/prm_ev/ev_a_manifest.json`: anchor `sample_index` list, settings
  hash, per-anchor reproduced value, database value, difference, convergence
  status, wall time;
- the elimination ledger from the selection rule;
- a short EV-A report stating pass or mismatch against the frozen
  tolerances.

## Explicitly not established by an EV-A pass

External validity of any model. That requires the separate EV-B/C/D blind
campaign: candidates frozen before labels, predictions committed before
labels, one consistent computation, no post-hoc exclusion.
