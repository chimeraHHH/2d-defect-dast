# PRB revision plan

## Current AutoSci stage

The project has entered the user-authorized J-R1/G1 correctness gate and the
J-R3/G3 analysis-build gate.  The completed E1--E9 record remains frozen.  G1
is authorized for the bounded permutation/MIC diagnostic, the invariant graph
repair, property tests, and one repaired-fold pilot.  Its source-task lineage
is now part of the same correctness gate: the frozen JARVIS subset must be
rebuilt without changing membership, a corrected pretraining checkpoint must
be versioned and hash-bound, and the legacy-initialization, no-pretraining, and
corrected-pretraining arms must remain distinguishable.  G3 is authorized for
pre-registration, implementation, and server-side exploratory analysis on the
frozen current OOF predictions.  Until G2 is separately authorized and produces
complete repaired-model OOF predictions, every old-OOF G3 result is explicitly
exploratory and is prohibited from supporting a canonical manuscript claim.

The user also authorized claim-bounded prose and figure restructuring for PRB.
This includes replacing unsupported cost-saving language with retrospective
energy estimation or reranking of available DFT-relaxed geometries, revising
the evidence order, and reserving a main-text physical-analysis figure without
inventing results.

On 2026-08-10 the user additionally approved gate **G2** (repaired full core
OOF) and gate **EV-A** (six convention-anchor DFT runs) as defined in
`paper_Q1/review/reviewer_revision_plan_2026-08-10_v2.md`.  G2 launches only
after completed G1/G1C server acceptance from a pushed clean commit, per
`paper_Q1/review/g2_execution_runbook.md`; EV-A executes only under its frozen
preregistration `paper_Q1/review/ev_a_preregistration.md`.

G1 was accepted on 2026-08-10 (`artifacts/prm_g1/g1_acceptance.json`, commit
88d71cc) and the 35-run repaired core OOF queue completed with the G2
acceptance recorded on 2026-08-11 (`artifacts/prm_g2/g2_acceptance.json`,
runs at commit 45997d3).  On 2026-08-11 the user approved **canonical G3**
promotion on the repaired OOF evidence.  The EV-B/C/D blind candidate
campaign, any prospective geometry campaign, release, and submission remain
unapproved transitions under manual control.

## Objective

Decide whether to (A) repair the graph representation, regenerate the core OOF
evidence, and build a physics-first *Physical Review B* Regular Article, or (B)
retain a strictly retrospective IMP2D benchmark and retarget it. New DFT is not
part of the active route and remains a separately authorized optional gate.

The controlling reviewer-response artifacts are:

- `paper_Q1/review/reviewer_comment_matrix_2026-08-10.md`;
- `paper_Q1/review/reviewer_revision_plan_2026-08-10_v2.md`, which supersedes
  the same-day v1 plan; v1 is retained as history and its G0/G1/G2/G4/G5 gate
  definitions remain valid where v2 cites them.

The minimum truthful task description is “retrospective energy estimation or
reranking of available DFT-relaxed geometries.” General DFT-acceleration claims
remain prohibited unless a later target-matched early-snapshot campaign passes.

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
7. Rewrite Q1 for PRB, compile it, and perform a skeptical scientific and
   submission audit.

## Controlled execution sequence

All formal commands run on the `WHUServer-L40S` server profile from a clean,
commit-pinned worktree.  The private server profile exports `PRM_PYTHON`,
`PRM_DATASET_PATH`, `PRM_RAW_DB_PATH`, and `PRM_RUN_ROOT`; public contracts bind
their content hashes and stable aliases, not account-specific absolute paths.
GPU 2 is excluded from every queue.

1. Allow `factorial-v4-indexed` to reach 40 complete runs with no failed or
   stopped jobs.  Validate every run against its versioned YAML, 150-epoch
   history, split hash, asset hashes, prediction-derived metrics, and declared
   output hashes.
2. From a separate clean collector worktree, run
   `scripts.prm_collect_factorial` against
   `$PRM_RUN_ROOT`.  Commit the archived
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
artifact-hash collection, and strict-JSON normalization in which undefined
constant-baseline correlations are represented as `null`. The verified 55-file
descriptor archive is mirrored byte-for-byte to the formal result store.
Hyperparameters and learned descriptor families are selected independently
inside every split using only that split's validation rows. LightGBM is selected
in all random and pair folds and the chemistry block; host and impurity folds
each select histogram gradient boosting four times and LightGBM once. The two
DART initialization assets are pinned by SHA-256, and formal runs record a
fail-closed parameter-load report. All 9,871,460 stored periodic graph edges
were independently reconstructed without topology or cutoff violations.

The corrected 40-run factorial completed from clean commit `7a6593a`; a
separate collector selected `g111` by validation MAE only. The promoted DART
and periodic-SchNet queues then completed 48 runs each from clean commit
`6c40137`, with GPU 2 excluded. The complete aligned comparison, five-member
internal UQ analysis on a dedicated ensemble split, and pair-OOF materials
screening analysis are archived under `artifacts/prm_results/`. No partial
queue metric entered the manuscript.

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

The following paragraph records the **pre-J-R1 paper snapshot** and is retained
only as historical provenance. Fail-closed asset generation validated 23 paper outputs. An independent audit
directly reaggregates archived predictions, reproduces all paper-facing
bootstrap analyses including screening references, verifies every LaTeX macro
and generated table row. The PRB J-stage audit extends this verifier to the
pair-held-out constituent exception and passes 1,346 checks with no failures.
The then-current 13-page main manuscript and 3-page Supplemental Material compiled
without undefined references or overfull boxes and have passed page-by-page
layout QA. Submission is nevertheless NO-GO: the DART graph builder retains the
first 32 ordered neighbor pairs at each center, so exact atom-permutation
robustness is not established, and the global radial bias uses componentwise
fractional wrapping rather than a guaranteed shortest image in oblique cells.
The manuscript disclosed both facts. J-R1 now implements the user-approved
diagnostic and invariant rebuild, but the historical numerical evidence remains
legacy-graph evidence until G2 is separately authorized and completed.

The active reconstructed package is 12 main-text pages with five main figures
and 7 Supplemental pages with four Supplemental figures. Its numeric audit,
final hashes, and graph-correctness acceptance remain pending server execution.
Author metadata, figure provenance, licensing, and the public archival release
are additional submission blockers.

The historical JARVIS checkpoint is also legacy evidence.  Its source graph
builder placed the triplet center in a column not consumed as the center by the
local layer, selected a bounded neighbor subset in a stored-order-dependent
way, and used nonperiodic Cartesian pair distances for three-dimensional
periodic crystals.  A repaired IMP2D graph combined with that checkpoint is a
legacy-initialization sensitivity arm, not an end-to-end corrected model.  G1C
must reconstruct the hash-frozen 19,902-record source subset, rerun the original
pretraining recipe on the repaired graph, and issue dataset/checkpoint receipts
before any later G2 run can satisfy the corrected-lineage gate.
