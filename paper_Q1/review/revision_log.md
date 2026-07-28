# Revision log

## Current state

- Draft: result-gated PRM LaTeX manuscript.
- Follow-up policy: execute all required non-DFT work.
- Blocking items: E4--E7 in `paper_experiment_matrix.md`, full author and
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
