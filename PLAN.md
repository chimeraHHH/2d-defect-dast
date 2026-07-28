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

## Controlled execution sequence

All formal commands run from a clean, commit-pinned worktree with
`/storage/ssd/metaiot_guest/yiminghua/prm_env/bin/python`.  GPU 2 is excluded
from every queue.

1. Allow `factorial-v4-indexed` to reach 40 complete runs with no failed or
   stopped jobs.  Validate every run against its versioned YAML, 150-epoch
   history, split hash, asset hashes, prediction-derived metrics, and declared
   output hashes.
2. From a separate clean collector worktree, run
   `scripts.prm_collect_factorial` against
   `/storage/ssd/metaiot_guest/yiminghua/prm_runs_v2`.  Commit the archived
   numerical evidence, factorial contrasts, and validation-only
   `selection.json` before generating any selected-model configuration.
3. Run `scripts.prm_promote_selected` from the committed selection.  It must
   produce 43 transfer configurations and five UQ-member configurations for
   exactly one `g[01][01][01]` variant.  Commit and push these controlled YAML
   files before training.
4. Run bounded GPU smoke jobs for the promoted DART and periodic SchNet paths.
   A smoke result is operational evidence only; it must not enter a paper
   metric.  Verify configuration identity, split identity, complete declared
   outputs, checkpoint readability, and DART calibration output where
   applicable.
5. Launch the promoted-DART and SchNet campaigns from two clean, frozen
   worktrees.  Assign disjoint GPU sets using complementary
   `--exclude-gpus` lists, keep GPU 2 excluded from both, and do not modify
   either worktree while its scheduler is active.  The result directories and
   scheduler queue IDs must also be disjoint.
6. Admit neural results only after `scripts.prm_collect_results` verifies the
   expected 48 SchNet runs, five selected factorial repeats, and 43 promoted
   DART transfer runs.  Descriptor families and the DART architecture remain
   validation-selected; no test metric is used for promotion.
7. Run `scripts.prm_uq_analysis` and `scripts.prm_materials_analysis` from a
   clean collector commit.  Then run `scripts.prm_make_result_assets`, which
   writes the result manifest before creating the manuscript readiness marker.
8. Run the bounded post-hoc SchNet readout sensitivity on host-CV only. Change
   atomwise graph readout from sum to mean while holding the other 15
   configurations fixed; archive it separately from the prespecified main
   comparison.
9. Rebuild the REVTeX manuscript, inspect every rendered page, cross-check all
   prose numbers against machine-readable bundles, and perform a final
   review-style audit before tagging an archival release.

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
two DART initialization assets are pinned by SHA-256, and formal runs record a
fail-closed parameter-load report. All 9,871,460 stored periodic graph edges
were independently reconstructed without topology or cutoff violations.

The corrected 40-run factorial completed from clean commit `7a6593a`; a
separate collector selected `g111` by validation MAE only. The promoted DART
and periodic-SchNet queues then completed 48 runs each from clean commit
`6c40137`, with GPU 2 excluded. The complete aligned comparison, five-member
held-out UQ analysis, and pair-OOF materials screening analysis are archived
under `artifacts/prm_results/`. No partial queue metric entered the manuscript.

The bounded post-hoc SchNet mean-readout sensitivity completed all 15 host-CV
runs from clean commit `91cb8ac` on GPUs 1, 4, and 5, with GPU 2 excluded.
The hash-verified archive is under
`artifacts/prm_results/sensitivity/schnet_readout/`. Replacing additive with
mean readout reduces pooled host-CV MAE from 6.703 to 2.552 eV; the paired
mean-minus-add difference is -4.151 eV with a host-cluster bootstrap 95%
interval of [-7.704, -1.256] eV and improvement in all five folds. The
remaining 2.552 eV error is still substantially above DART's 0.938 eV, so the
result identifies readout as an important part of the complete-recipe gap but
does not isolate a backbone-only effect. The prespecified additive SchNet
remains the main comparator.

Fail-closed asset generation now validates 23 paper outputs, and the populated
main manuscript and separate Supplemental Material compile to visually checked
12-page and one-page PDFs. An independent audit directly reaggregates archived
predictions, reproduces all paper-facing bootstrap analyses, verifies every
LaTeX macro and generated table row, and passes 1,265 checks with no failures;
its report is `artifacts/prm_results/paper/numeric_audit.json`. The remaining
scientific gate is the skeptical manuscript review. Administrative release
work remains blocked on verified author/contact/funder metadata and a
rights-holder-approved code/data license.
