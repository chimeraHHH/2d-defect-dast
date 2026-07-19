# PRM revision plan

## Objective

Revise the Q1 manuscript into a reproducible *Physical Review Materials*
Regular Article about impurity-incorporation energetics and applicability
domains in two-dimensional materials. New DFT calculations and claims of
prospective DFT validation are explicitly out of scope.

## Scientific mainline

The paper will test when a defect-aware graph transformer transfers across
host and dopant chemistry, whether its three architectural components provide
reproducible gains under one training protocol, and whether validation-calibrated
uncertainty identifies a useful selective-prediction domain.

## Evidence contract

- Dataset: 10,641 source-filtered IMP2D rows in `cleaned_dataset.pkl`, reduced
  to a canonical 10,224-structure modeling set after excluding four rows with
  non-reconstructable raw energy components, 349 rows without a
  permutation-invariant impurity identity, and 64 redundant structures.
- Development benchmark: the historical seed-42 80/10/10 split, retained only
  for comparison with archived runs and labelled as previously inspected.
- Confirmatory evidence: paired random repeats, random out-of-fold prediction,
  balanced host-, dopant-, and host--dopant-pair-group cross-validation, plus a
  predefined host-by-dopant chemistry block holdout.
- Architecture test: full 2^3 factorial for gated readout, local-environment
  enrichment, and the composite pre-norm distance-gated local block, five
  paired repeats, identical trainer.
- Baselines: mean/ridge, random forest and histogram gradient boosting on
  reproducible chemistry/structure descriptors; SchNet as the neural comparator.
- Metrics: MAE, RMSE, bias, Spearman correlation, macro-group MAE, paired
  bootstrap confidence intervals, and low-energy ranking metrics.
- UQ: checkpoint selection uses validation data, while variance scaling and
  conformal quantiles use a distinct calibration partition; test data are
  evaluation only. Report coverage, interval width, Gaussian NLL, CRPS,
  risk-coverage and AURC.
- Every run must preserve commit, dirty state, command, resolved config, dataset
  identity, split identity, seed, environment, sample indices and predictions.

## Work packages

1. Freeze the repository, data manifest, split schema and result schema.
2. Audit targets, metadata, duplicates and split leakage.
3. Run the architecture factorial and fair baselines.
4. Run host/dopant/block transfer experiments with the selected architecture.
5. Run calibrated ensemble UQ, selective prediction and materials analyses.
6. Generate paper-facing tables and figures from canonical result bundles.
7. Rewrite Q1 as the PRM manuscript, compile it, and perform a skeptical audit.

## Go/no-go rules

- Architecture superiority is claimed only when the paired 95% interval excludes
  zero on confirmatory splits. Otherwise the architecture result is reported as
  inconclusive and the paper is framed around applicability domains.
- No model or ensemble member may be selected using test labels.
- Attention or occlusion statistics are descriptive, never causal evidence.
- Existing QE/GPAW calculations are not described as validation of IMP2D
  formation energies and do not support a discovery-rate claim.
- The paper is complete only when every number in the abstract, main tables and
  figure captions is linked to a canonical machine-readable result.

## Current route

The final 10,224-structure protocol v2, periodic SchNet implementation, result
collectors, and manuscript protocol sections are complete. All 27 descriptor
splits have been rerun from a clean commit and passed content-level validation,
artifact-hash collection, and a strict-JSON normalization in which undefined
constant-baseline correlations are represented as `null`. The verified 55-file
descriptor archive is mirrored byte-for-byte to the formal result store.
Validation selects LightGBM for the random, pair, and chemistry-block regimes
and histogram gradient boosting for host- and dopant-held-out evaluation. The
two DART initialization assets are pinned by SHA-256, and formal runs must record
a fail-closed parameter-load report. On WHUServer-L40S, clean training commit
`af6eeec` passed bounded protocol-v2 DART and SchNet CPU smoke runs. All
9,871,460 stored periodic graph edges were independently reconstructed without
topology or cutoff violations. GitHub code and evidence commit `866fcd4` passes
all 133 tests in an isolated remote checkout. UQ and materials collectors reject
non-finite inputs, undefined rank statistics, non-standard JSON numbers, and
trivial or tied decision sets where the corresponding accuracy is not defined.
The stale protocol-v1 queue was stopped before it launched any jobs. The
systemd-managed `factorial-v2` queue waits for the declared GPU idle thresholds;
the latest audit found all 40 paired runs pending, no attempts consumed, and no
failures. After completion, the route is validation-only promotion, transfer,
SchNet, ensemble UQ, and paper result generation.
