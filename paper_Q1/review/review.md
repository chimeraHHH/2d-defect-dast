# Final skeptical review

> Historical PRM-stage record (2026-07-29). It is superseded for current
> decisions by `j_stage_autosci_review.md` and the 2026-08-10 PRB full review in
> `paper_Q1/ccfa-review-reports/`.

Review date: 2026-07-29

## Review basis

This review treats `paper_Q1/main.tex`, `paper_Q1/supplement.tex`, the canonical
result bundles under `artifacts/prm_results/`, and the independent numerical
audit as the submission evidence. It does not credit historical exploratory
results or any prospective DFT calculation.

The review was performed after:

- fold-specific validation selection replaced regime-wide descriptor-family
  selection;
- trivial screening references and paired model-minus-reference intervals were
  added;
- the UQ test was reclassified as an internal dedicated ensemble split rather
  than an untouched outer holdout;
- exact model, optimization, seed, and descriptor-search specifications were
  added to the Supplemental Material.

## Judgment

No unresolved critical or major scientific defect remains under the manuscript's
current claims. The scientific content is suitable to proceed as a
*Physical Review Materials* Regular Article. Submission remains blocked by
author-supplied metadata and archival-release actions, not by another
non-DFT experiment.

The defensible claim is an audited, split-aligned IMP2D evaluation with a paired
factorial, bounded chemical-transfer tests, internal uncertainty calibration,
and retrospective screening decisions. The paper does not establish universal
architecture superiority, prospective materials discovery, or performance on
an external target-matched DFT set.

## Residual risks

### R1: Comparator breadth

- Severity: moderate, disclosed.
- Evidence: periodic SchNet, four learned descriptor families, a mean
  predictor, and the full DART factorial are aligned to the same partitions.
  They do not exhaust modern equivariant atomistic architectures.
- Resolution: all superiority statements are restricted to the evaluated
  comparators. DART--SchNet is explicitly an end-to-end recipe comparison.
  The host-only mean-readout sensitivity is labeled post hoc.
- Decision: no additional comparator is required for the present scoped PRM
  claim. A modern equivariant comparator would strengthen impact but is not
  needed to make the reported comparison valid.

### R2: UQ is internal

- Severity: moderate, corrected.
- Evidence: the 1,023 test rows are excluded from ensemble fitting, checkpoint
  selection, and calibration, but many occur in other roles during earlier
  architecture development on the same IMP2D corpus.
- Resolution: the abstract, Methods, Results, Discussion, Conclusion, table
  caption, and review records now call this an internal dedicated ensemble
  split and exclude an outer-holdout guarantee.
- Decision: acceptable as an internal calibration and selective-prediction
  analysis; not evidence of chemically shifted calibration.

### R3: Screening evidence is retrospective

- Severity: moderate, disclosed.
- Evidence: pair-out-of-fold predictions support incorporation-class, exact
  site, top-two, and regret metrics within IMP2D. Empirical-majority and
  analytic candidate-uniform references now quantify the trivial decision
  floor, with pair-cluster bootstrap intervals for model gains.
- Resolution: no synthesis, thermodynamic prevalence, or new-material claim is
  made. External DFT validation remains outside scope.
- Decision: sufficient for a materials-facing utility analysis, not for a
  discovery claim.

### R4: Dataset and transfer boundary

- Severity: moderate, disclosed.
- Evidence: the protocol removes four non-reconstructable targets, 349 rows
  without a permutation-invariant impurity identity, and 64 redundant rows.
  JARVIS pretraining overlaps some host formulas and all impurity elements.
- Resolution: grouped tests mean absence from IMP2D target supervision, not
  element- or formula-disjoint pretraining. Claims are limited to 10,224
  neutral adsorbate/interstitial structures.
- Decision: the exclusions reduce coverage but prevent a more serious identity
  and leakage problem.

### R5: Submission metadata

- Severity: administrative blocker.
- Missing verified inputs: complete author order and affiliations,
  corresponding-author email, funder wording, rights-holder-approved license,
  and a tagged archival release with persistent identifier.
- Decision: do not infer these fields from older drafts. Replace the explicit
  placeholders only after author confirmation.

## Verification

- Remote scientific environment: 179 tests passed; two PyTorch JIT deprecation
  warnings only.
- Independent paper audit: 1,331 checks passed with zero failures.
- Paper assets: 23 hash-bound generated outputs.
- LaTeX: main manuscript and Supplemental Material compile without undefined
  references, missing citations, overfull boxes, or fatal errors.
- Visual QA: all rendered pages were checked for clipping, overlap, unreadable
  text, broken glyphs, and figure/table legibility.

## PRM fit

The paper fits the PRM Regular Article route: APS lists no fixed length limit
for that article type and asks authors to provide broad materials context.
Section M3-A, “Development of new methods for materials,” is the most direct
fit. PRM is a hybrid journal, so the requested subscription-funded,
non-open-access route can be selected during submission.

APS also requires publicly shared data and software to be cited in the
reference list and named in the Data Availability Statement. The manuscript's
provisional repository sentence is scientifically adequate for review, but the
final tagged release, citation, version, and persistent identifier remain
mandatory packaging actions.

## Final decision

Proceed to PRM submission after the administrative substitutions and archival
release. No further non-DFT scientific experiment is required under the claims
currently written. Do not broaden the claims unless a new aligned comparator,
chemically shifted UQ evaluation, or target-matched external validation set is
added.
