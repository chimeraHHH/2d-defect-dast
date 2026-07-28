# Revision log

## Current state

- Draft: scientific-content-complete PRM LaTeX manuscript.
- Follow-up policy: execute all required non-DFT work.
- Blocking items: verified author/contact/funder metadata,
  rights-holder-approved licensing, and final release metadata.

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
- Status: complete; the readiness marker is bound to the full canonical
  comparison, UQ, and materials bundles.
- Blocks finalization: no.

### REV-003: Novelty boundary

- Severity: major.
- Fix type: literature positioning and claim downgrade.
- Change: introduction and review materials position the contribution as the
  audited IMP2D protocol, paired factorial, chemical transfer, internal UQ, and
  screening analysis; first-of-kind graph-model claims are excluded.
- Status: complete.
- Blocks finalization: no.

### REV-004: Baseline breadth

- Severity: major.
- Fix type: claim boundary.
- Change: use periodic SchNet, validation-selected descriptors, and the full
  factorial as aligned controls; explicitly state that these do not exhaust
  atomistic architectures.
- Status: complete for the prespecified comparison; the post-hoc readout
  sensitivity is tracked separately as REV-014.
- Blocks finalization: no.

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
- Blocks finalization: no.

### REV-008: Five-repeat factorial interval strength

- Severity: major interpretation issue.
- Fix type: existing-result robustness reporting and claim boundary.
- Change: preserve the canonical percentile intervals and selection bundle,
  but add repeat-level directional consistency for every interval-directional
  contrast and state that five-repeat intervals are descriptive uncertainty
  summaries rather than large-sample hypothesis tests.
- Status: implemented in the result-asset generator and Evaluation Protocol.
- Blocks finalization: no.

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

### REV-013: Dedicated uncertainty evidence

- Severity: critical evidence gate.
- Fix type: complete-run collection and calibration analysis.
- Change: collected all five promoted DART ensemble members, split the
  dedicated calibration partition label-independently into variance-scaling
  and conformal subsets, and evaluated all uncertainty metrics only on the
  dedicated 1,023-structure test partition. These rows were excluded from
  ensemble fitting, checkpoint selection, and calibration, but not from the
  earlier architecture-development corpus. The complete archive records
  member, split, prediction, checkpoint, and collector hashes.
- Evidence: `artifacts/prm_results/uq/`.
- Status: complete before paper asset generation.
- Blocks finalization: no.

### REV-014: SchNet readout interpretation

- Severity: major comparator-interpretation issue.
- Fix type: explicit recipe boundary plus bounded post-hoc sensitivity.
- Change: disclose that the prespecified periodic SchNet comparator uses
  atomwise additive readout, avoid attributing its host-held-out error to the
  interaction backbone alone, and rerun only the 15 host-CV configurations
  with mean readout while holding all other controlled fields fixed.
- Evidence: `artifacts/prm_results/sensitivity/schnet_readout/` and
  `configs/prm/sensitivity/schnet_mean_host/`.
- Result: all 15 runs completed from clean commit `91cb8ac`; pooled host-CV MAE
  changes from 6.703 eV (additive) to 2.552 eV (mean), with paired difference
  -4.151 eV, 95% host-cluster bootstrap interval [-7.704, -1.256] eV, and
  improvement in all five folds.
- Status: complete and written in Discussion and Supplemental Material.
- Blocks finalization: no.

### REV-015: Final scientific-content manuscript package

- Severity: major packaging gate.
- Fix type: fail-closed asset regeneration, compilation, and visual QA.
- Change: generated schema-v2 assets with 23 hash-bound outputs, populated the
  main manuscript, added a separate Supplemental Material file, and compiled
  visually inspected main and Supplemental Material PDFs.
- Evidence: `artifacts/prm_results/paper/result_assets.json`,
  `paper_Q1/main.pdf`, and `paper_Q1/supplement.pdf`.
- Status: complete; no undefined references, content-affecting warnings,
  clipping, overlap, or unreadable result elements remain.
- Blocks finalization: no.

### REV-016: Independent numerical audit

- Severity: critical evidence gate.
- Fix type: independent prediction-level recomputation and manuscript contract
  verification.
- Change: added a standalone audit that reaggregates archived DART, SchNet, and
  descriptor predictions; reproduces paired and clustered bootstrap results,
  factorial summaries, UQ, materials screening, and readout sensitivity; and
  verifies all asset hashes, result macros, and generated table rows. Report
  floats are serialized to 15 significant digits so platform-specific libm
  tails do not change the audit artifact.
- Evidence: `scripts/prm_verify_paper_numbers.py` and
  `artifacts/prm_results/paper/numeric_audit.json`.
- Status: complete; 1,331 checks pass with zero failures.
- Blocks finalization: no.

### REV-017: Descriptor-family selection leakage

- Severity: critical comparison-design issue.
- Fix type: split-local model selection and archive recollection.
- Change: replaced one descriptor family selected from mean validation MAE
  across all folds with independent family selection inside each split using
  only that split's validation rows. The pooled descriptor prediction now
  concatenates those foldwise selected test predictions; the mean predictor is
  retained only as a separate baseline.
- Evidence: `artifacts/prm_results/descriptors/selection.json` and
  `artifacts/prm_results/comparison/descriptor_selection.json`.
- Result: LightGBM is selected in all random and pair folds and the chemistry
  block; host and impurity folds each select histogram gradient boosting four
  times and LightGBM once.
- Status: complete and independently audited.
- Blocks finalization: no.

### REV-018: Screening reference baselines

- Severity: major interpretation issue.
- Fix type: decision-level reference construction.
- Change: added an empirical majority-class reference for incorporation class
  and analytic candidate-uniform references for global exact-site,
  within-class exact-site, and top-two decisions. Added row-aligned
  model-minus-reference gains with host--impurity-pair cluster-bootstrap
  intervals.
- Evidence: `artifacts/prm_results/materials/summary.json` and
  `paper_Q1/generated/tab_screening.tex`.
- Status: complete and independently audited.
- Blocks finalization: no.

### REV-019: UQ development-scope correction

- Severity: major claim-boundary issue.
- Fix type: manuscript-wide wording correction.
- Change: removed every claim that the dedicated UQ test is untouched by the
  full development process. The paper now distinguishes exclusion from
  ensemble fitting/checkpoint selection/calibration from reuse of the same
  canonical corpus during earlier architecture development.
- Status: complete in the abstract, Methods, Results, Discussion, Conclusion,
  generated table caption, and review records.
- Blocks finalization: no.

### REV-020: Final skeptical review and reproducibility table

- Severity: final scientific-content gate.
- Fix type: independent review, reproducibility disclosure, compilation, and
  visual QA.
- Change: completed the post-result skeptical review, added exact DART,
  SchNet, optimization, seed, and descriptor-search specifications to the
  Supplemental Material, reran the complete test and numerical audit gates,
  and inspected the rebuilt PDFs.
- Evidence: `paper_Q1/review/review.md`, `paper_Q1/supplement.tex`,
  `paper_Q1/main.pdf`, and `paper_Q1/supplement.pdf`.
- Status: complete. Only author-supplied metadata and archival-release actions
  remain.
- Blocks finalization: no scientific blocker; administrative blockers remain.
