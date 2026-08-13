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

### REV-021: J-stage narrative convergence and submission accessibility

- Severity: major presentation and claim-boundary issue.
- Fix type: writing, terminology, figure accessibility, and submission-package
  preparation; no new experiment or model run.
- Change: reorganized the title, abstract, introduction, transfer results, and
  conclusion around three durable findings: paired G/P support with an
  inconclusive independent E effect; axis-resolved chemical transfer; and
  bounded retrospective screening with internal-only uncertainty evidence.
  Replaced broad transferability and preferred-class language, identified the
  prespecified SchNet comparator as an additive-readout learning recipe, and
  corrected the non-trained decision-reference terminology.  Removed the
  REVTeX/float-package conflict, repaired the Supplemental Material citation,
  and added grayscale-redundant hatching/line styles to the affected figures.
- Evidence: `paper_Q1/main.tex`, `paper_Q1/sections/`,
  `paper_Q1/supplement.tex`, and the rebuilt paper-facing assets.
- Verification: 20 targeted paper-asset/protocol/audit tests pass; the
  independent numerical audit again passes 1,331 of 1,331 checks; the rebuilt
  13-page main PDF and 2-page Supplemental Material compile without undefined
  references, overfull boxes, or float-package conflicts and pass page-by-page
  visual inspection.
- Status: complete.
- Blocks finalization: no scientific blocker; author-controlled submission
  metadata, licensing, and archival-release actions remain.

### REV-022: Journal-calibrated figure program and final J-stage rewrite

- Severity: major presentation and evidence-architecture issue.
- Fix type: PRM/PRB/PRL figure survey, vector figure generation, manuscript
  restructuring, and render-level QA; no new experiment or model run.
- Change: added a config- and source-bound DART architecture figure; replaced
  code-only factorial labels with readable G/E/P combinations; rebuilt the
  transfer figure around axis-resolved DART error and paired comparator
  contrasts with an explicit broken axis; moved descriptive error heterogeneity
  to Supplemental Fig. S1; narrowed the title to impurity formation energies;
  and aligned the Introduction, Methods, Results, captions, cover letter, and
  Supplemental Material description with the new six-figure sequence.
- Evidence: `paper_Q1/review/j_stage_figure_redesign.md`,
  `paper_Q1/review/figure_catalog.json`,
  `artifacts/prm_results/paper/protocol_figure.json`, and
  `artifacts/prm_results/paper/result_assets.json`.
- Verification: 20 targeted tests pass; the numeric audit passes 1,331 of
  1,331 checks; the 13-page main PDF and 3-page Supplemental Material compile
  without undefined references, overfull boxes, clipping, or unembedded fonts;
  every paper-facing figure passes direct and grayscale render inspection.
- Status: complete.
- Blocks finalization: no scientific blocker; author-controlled byline,
  funding, licensing, and archival-release metadata remain.

### REV-023: AutoFigure editor-only presentation pass

- Severity: presentation refinement; no change to scientific evidence.
- Fix type: local AutoFigure-Edit SVG editing, provenance binding, and
  render-level QA; no model generation, experiment, inference, or numerical
  analysis.
- Change: created separate presentation derivatives for the DART architecture
  and factorial figures.  Figure 1 gains stronger panel-title hierarchy, an
  in-box wrap for “9 normalized attributes,” and SVG accessibility metadata.
  Figure 3 removes the duplicated left-panel legend and retains the complete
  upper-right key.  Canonical scientific-source PDF/PNG files remain intact.
- Guard: the Figure 1 derivative preserves all 91 paths exactly.  After the
  allowlisted removal of `legend_1`, Figure 3 preserves every remaining path,
  use, rectangle, text item, data point, interval, axis, and label exactly.
  Both promoted SVG candidates contain live text and zero raster images.
- Evidence: `paper_Q1/review/autofigure_edit_contract.md`,
  `paper_Q1/review/autofigure_edit_manifest.json`, and
  `paper_Q1/review/figure_catalog.json`.
- Verification: the 13-page main PDF compiles without undefined references or
  overfull boxes; embedded-font and raster checks pass; the frozen numerical
  audit passes 1,331 of 1,331 checks; 22 targeted protocol/result-asset/paper
  tests pass; final-size pages 3 and 8 pass visual inspection.
- Status: integrated and QA-passed; local milestone commit remains under user
  control.
- Blocks finalization: no scientific blocker.  The generative AutoFigure chain
  remains intentionally blocked pending credentials, local segmentation/model
  assets, and explicit authorization for any remote transmission.

### REV-024: User-supplied schematic integration

- Severity: presentation and method-communication revision; no change to the
  frozen scientific evidence.
- Fix type: manual raster-asset integration, scientifically bounded captions,
  LaTeX placement, provenance capture, and render-level QA; no experiment,
  inference, model selection, or numerical analysis.
- Change: replaced the former Main Figure 1 architecture presentation with
  `figure1_manual.png`; inserted `figure2_manual.png` after the first Methods A
  definition of the two incorporation classes; and inserted
  `figure3_manual.png` after the G/E/P module description in Methods C.  Natural
  source order assigns these assets Main Figures 1, 2, and 4, respectively;
  the protocol overview remains Main Figure 3 and the four quantitative main
  figures become Figures 5--8.
- Scientific guards: the captions state that the impurity-status embedding is
  added after the 137-to-128 projection, 12~\AA{} is a radial-grid endpoint
  rather than an attention cutoff, class screening compares retained
  class-wise candidate minima, the two colored impurities in Figure 2 are
  alternative rather than simultaneous configurations, and the E descriptor
  is normalized, projected, and added only to the impurity node.
- Provenance: `paper_Q1/review/manual_figure_manifest.json` records source and
  paper hashes, pixel hashes, dimensions, DPI metadata, LaTeX labels, formal
  numbers, and pixel-exact source comparisons.  The three manuscript PNGs are
  explicitly allowlisted in `.gitignore` so a later user-controlled milestone
  commit cannot silently omit them.
- Verification: the rebuilt main paper remains 13 pages and compiles without
  undefined references, overfull boxes, or clipped figures.  Pages containing
  Figures 1--4 pass color and grayscale inspection; Figure 2 has a documented
  grayscale advisory because position and labels carry part of its redundant
  encoding.  The frozen numerical audit passes 1,331 of 1,331 checks and all
  22 targeted protocol/result-asset/paper-number tests pass.
- Status: integrated and QA-passed; local milestone commit and push remain
  under user control.
- Blocks finalization: no scientific blocker.  The raster artwork contains
  baked-in text, so correcting source-image wording and adopting a
  colorblind-redundant Figure 2 encoding remain advisable before camera-ready
  production.

### REV-025: AutoSci PRB redirection, physics-first rewrite, and red-team gate

- Severity: target-journal revision plus newly identified scientific blocker;
  no change to frozen model predictions or reference data.
- Fix type: PRB requirements audit, full scientific/writing review,
  evidence-bounded prose revision, executable claim audit, and final-size PDF
  inspection. No training, inference, DFT, or GPU work was performed.
- Change: migrated the manuscript and Supplemental Material to the PRB REVTeX
  option; rewrote the title, abstract, Introduction, Discussion, Conclusion,
  cover letter, DAS, and checklist around chemical-axis transfer and
  incorporation-geometry asymmetry; defined the IMP2D chemical-potential
  convention; promoted the 0.719/0.322 eV class-specific errors; narrowed
  screening to existing relaxed candidates; and added current APS AI-use
  disclosures.
- Integrity corrections: the pair-held-out narrative now records the single
  `Ti2CO2|Na` singleton-host exception. Methods and Discussion disclose that
  the angle branch retains the first 32 ordered neighbor pairs and that the
  global radial bias uses componentwise fractional wrapping rather than a
  guaranteed shortest image in oblique cells. Strict atom-permutation
  robustness and minimum-image claims were withdrawn.
- Evidence: the independent numerical audit now checks the 10,223/10,224
  pair-constituent statement and passes 1,346 of 1,346 checks; 22 targeted
  protocol/result/paper tests pass.
- Build: the 13-page main PDF and 3-page Supplemental Material compile without
  overfull boxes, unresolved citations/references, fatal errors, or unembedded
  fonts. All 16 pages were inspected; no overlap, clipping, broken glyph, or
  unreadable equation/table was found.
- Decision: PRB submission remains NO-GO. The order-sensitive triplet cap is a
  model-level risk that prose cannot repair; the user must choose a targeted
  permutation-sensitivity gate, an invariant rebuild with a new evidence
  milestone, or stop/retarget. Figure-source corrections, provenance, author
  metadata, licensing, and the archival release also remain open.
- Control: no commit, push, release, or upload was performed; progress remains
  under manual user control.

### REV-026: PRM V19 figure-story reconstruction

- Severity: presentation, figure provenance, and PRB package consistency; no
  new training, inference, DFT, or numerical evidence was introduced.
- Change: replaced the first two user schematics with author-directed GPT
  Image 2 revisions; merged pair-held-out parity and the five-regime transfer
  hierarchy into a deterministic quantitative Figure 3; retained the physical
  error map, retrospective reranking, and permutation diagnostic as Figures
  4--6; moved protocol, benchmark, periodic-table, factorial, UQ,
  heterogeneity, and case-level support to the Supplemental Material.
- Scientific guards: Figure 1 distinguishes the 12-Angstrom radial-grid end
  from an attention cutoff and ends at per-structure energy prediction;
  Figure 2 shows the incorporation classes as alternative single-impurity
  configurations with redundant color and shape. No image model touched a
  quantitative figure.
- Provenance: `v19_image2_manifest.json` records authorization, model, input
  scope, source and derivative hashes, and author verification;
  `v19_visual_contract.md` records the role and evidence boundary of every
  main-text figure. Captions and Acknowledgments disclose Image 2 assistance.
- Build and QA: the PRB-profile main and Supplemental PDFs each contain 13
  letter-size pages. The main PDF has six figures, no unresolved references,
  no overfull boxes, and embedded fonts; all pages passed rendered contact-sheet
  inspection. SHA-256 values are `ddbb3c02...109f06c` (main) and
  `25b5f584...5b6a20b` (Supplemental).

### REV-027: Figure 1 local-environment inset

- Change: replaced the first V19 architecture draft with an author-directed
  GPT Image 2 edit that uses two callout leaders from the impurity node to a
  true 5-\AA{} local-neighborhood inset. The inset displays coordination
  number, mean and maximum neighbor distances, and electronegativity contrast;
  the former standalone E-module figure remains absent.
- Integrity boundary: all backbone, radial-grid, local-cutoff, readout, and
  per-structure output labels are unchanged. No quantitative result or model
  claim was added.
- QA: the revised 13-page PRB build has no unresolved reference, overfull box,
  clipping, or page-count change; the final-size Figure 1 page passed visual
  inspection.

### REV-028: Figure 1 paper-infographic style transfer

- Change: used GPT Image 2 to redraw the verified Figure 1 topology in the
  broad visual language of an author-supplied paper figure: warm paper ground,
  subtle hatch texture, hand-drawn rounded cards, stronger panel headings, and
  a slim architecture takeaway strip.
- Source separation: the supplied DriveCache image was used only as a style
  reference. No vehicles, robots, icons, labels, data, or exact composition
  were copied. The DART diagram remained the sole source of scientific content.
- Integrity boundary: every architecture module, numerical setting, callout,
  and arrow direction was preserved. The takeaway strip restates the caption's
  local-plus-global, non-extensive prediction principle and is not a new module.

### REV-029: Figure 1 alignment and operation pictograms

- Change: aligned all three panel cards to one grid, centered headings and box
  labels, made the readout branches symmetric, and regularized the vertical
  backbone spacing. Removed the in-image sentence about 12~\AA{} not being a
  cutoff while retaining the correct 0--12~\AA{} radial-basis-grid label.
- Visual enrichment: added subordinate pictograms for the latent vector,
  elemental attributes, three repeated local blocks, four-head attention,
  radial basis functions, attention pooling, channelwise maximum, scalar gate,
  and scalar formation-energy output.
- Integrity boundary: each pictogram illustrates an operation already named in
  its enclosing module. No model branch, training signal, metric, or scientific
  claim was introduced.
