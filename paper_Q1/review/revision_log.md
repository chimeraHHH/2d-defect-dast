# Revision log

## Current state

- Draft: result-gated PRM LaTeX manuscript.
- Follow-up policy: execute all required non-DFT work.
- Blocking items: E5 and E7 in `paper_experiment_matrix.md`, full author and
  contact metadata, and final release metadata.

## Issue log

### REV-001: Unverified DefiNet reference

- Severity: critical integrity issue.
- Fix type: literature positioning and claim removal.
- Change: removed the unresolvable DOI
  `10.1021/acs.chemmater.4c02907`, its fabricated-looking metadata, and the
  dependent equivariant-formation-energy phrase.
- Status: complete in commit `22cbba8`.
- Blocks finalization: no.

### REV-002: Numerical sections lack complete downstream evidence

- Severity: critical.
- Fix type: required experiment and analysis completion.
- Change: Results, Discussion, and Conclusion remain behind the readiness
  marker until complete collector-validated assets exist.
- Status: in progress.
- Blocks finalization: yes.

### REV-003: Novelty boundary

- Severity: major.
- Fix type: literature positioning and claim downgrade.
- Change: introduction and review materials position the contribution as the
  audited IMP2D protocol, paired factorial, chemical transfer, held-out UQ, and
  screening analysis; first-of-kind graph-model claims are excluded.
- Status: complete, subject to final numerical rewrite.
- Blocks finalization: no.

### REV-004: Baseline breadth

- Severity: major.
- Fix type: claim boundary.
- Change: use periodic SchNet, validation-selected descriptors, and the full
  factorial as aligned controls; explicitly state that these do not exhaust
  atomistic architectures.
- Status: implemented in Methods and Discussion; final paired evidence pending.
- Blocks finalization: yes, through E4 rather than a new comparator campaign.

### REV-005: Author and archival metadata

- Severity: administrative blocker.
- Fix type: final packaging.
- Change: replace the author TODO and provisional release sentence only from
  verified author-supplied metadata and the actual tagged release; cite the
  released data/software in the reference list and Data Availability Statement.
- Status: pending.
- Blocks finalization: yes.

### REV-006: Source-task chemistry overlap

- Severity: major claim-boundary issue.
- Fix type: provenance analysis and claim downgrade.
- Change: added a clean, hash-bound JARVIS--IMP2D overlap audit and revised
  Methods and Discussion so grouped holdout means absence from IMP2D target
  supervision, not complete exclusion from source-task formula or element
  exposure.
- Evidence: `artifacts/prm_results/operations/pretraining_overlap_3a98513.json`.
- Status: complete.
- Blocks finalization: no.

### REV-007: Calibration-subset exchangeability

- Severity: major statistical-design issue.
- Fix type: prespecified analysis correction before result exposure.
- Change: replaced source-index alternation with a label-independent random
  partition of the dedicated calibration rows using fixed seed 6201. The
  frozen train/validation/calibration/test split and all trained models remain
  unchanged.
- Status: implemented before UQ collection.
- Blocks finalization: through E5 only.

### REV-008: Five-repeat factorial interval strength

- Severity: major interpretation issue.
- Fix type: existing-result robustness reporting and claim boundary.
- Change: preserve the canonical percentile intervals and selection bundle,
  but add repeat-level directional consistency for every interval-directional
  contrast and state that five-repeat intervals are descriptive uncertainty
  summaries rather than large-sample hypothesis tests.
- Status: implemented in the result-asset generator and Evaluation Protocol.
- Blocks finalization: through E7 only.

### REV-009: Feature and capacity reproducibility

- Severity: major methods-reporting issue.
- Fix type: writing and code-documentation correction.
- Change: enumerate all nine node attributes and all 80 descriptor dimensions,
  state the actual fallback/padding rules, correct the false median-imputation
  docstring, and disclose trainable parameter counts for DART and SchNet.
- Status: complete.
- Blocks finalization: no.

### REV-010: Public repository front page

- Severity: major submission-package integrity issue.
- Fix type: repository documentation.
- Change: replace the historical 982-line README, which mixed withdrawn scores,
  stale novelty claims, and prospective DFT statements, with a concise PRM
  evidence map, asset reconstruction route, verification commands, and scope
  boundary. Correct the JARVIS conversion script's stale chemistry-disjoint
  claim.
- Status: complete.
- Blocks finalization: no.

### REV-011: Source-dataset citation

- Severity: major data-provenance issue.
- Fix type: reference and availability correction.
- Change: cite the exact IMP2D dataset version used in addition to the
  accompanying article: Interstitial and Adsorbate Structure Database,
  version 2, DOI `10.11583/DTU.19692238.v2`.
- Status: complete.
- Blocks finalization: no.

### REV-012: Empty nonpositive-target stratum

- Severity: major result-packaging issue.
- Fix type: analysis edge-case correction.
- Change: represent a pooled regime with no test targets satisfying
  $E_{\mathrm f}\leq0$ by an eligible count of zero and an undefined stratum
  MAE, rendered as a dash with an explicit table note, instead of aborting the
  complete comparison collector.
- Status: complete before exposure of pooled metrics.
- Blocks finalization: no.
