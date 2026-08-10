# Full PRB Review: Learning impurity incorporation energetics across host and impurity chemistry in two-dimensional materials with a defect-aware graph Transformer

## 1. Report Metadata

Review date: 2026-08-10
Target venue/year/track: Physical Review B, 2026, Regular Article; suggested primary section B15(4), secondary section B1(1)
Paper title: *Learning impurity incorporation energetics across host and impurity chemistry in two-dimensional materials with a defect-aware graph Transformer*
Input materials reviewed: `paper_Q1/main.tex`, all section files, `supplement.tex`, compiled PDFs, bibliography, cover letter, figure manifests, protocol/result artifacts, implementation in `src/`, and the executable numerical audit
Search basis: APS PRB author/scope/style/submission/data/AI-policy pages; Crossref-verified cited literature; IMP2D article and Supplementary Note 3; related PRB figure survey
Report file: `paper_Q1/ccfa-review-reports/2026-08-10-learning-impurity-incorporation-prb-review.md`
Reviewer mode: Full scientific, writing, format, policy, and reproducibility review under the AutoSci J-stage workflow

## 2. Desk Rejection Assessment

- **Paper length — pass.** The main paper is 13 pages and the Supplemental Material is 3 pages. PRB Regular Articles have no fixed manuscript-length limit.
- **Topic compatibility — pass with editorial-risk qualification.** Two-dimensional impurity energetics, defect physics, and graph models are in scope. The rewrite now foregrounds chemical-transfer and incorporation-geometry physics rather than a generic model leaderboard.
- **Minimum quality — fail at the current snapshot.** The angle branch retains the first 32 ordered neighbor pairs, so the trained model may depend on stored atom order. Text disclosure repairs the claim but not the underlying method risk.
- **Policy, anonymity, and compliance — fail pending author action.** The paper correctly uses the nonanonymous PRB format and now discloses substantive Codex use. The complete author metadata, funder record, declarations, final archival release, and manual-figure provenance remain unresolved.
- **Prompt injection and hidden manipulation detection — pass.** No instruction-like hidden text or review-manipulation content was identified in the manuscript sources or rendered paper.
- **Ethics and reviewability — uncertain.** The evidence and limitations are inspectable, but figure rights/AI-generation history and the complete project-wide AI-use description still require author confirmation.

## 3. Paper Summary And Contribution Map

The manuscript predicts neutral impurity formation energies for 10,224 retained IMP2D adsorbate and interstitial structures. It introduces DART, a graph Transformer with defect-centered features, local pair/angle interactions, global radial-bias attention, and gated readout. A paired (2^3) factorial evaluates three model interventions; grouped evaluations separate random interpolation, pair holdout, host holdout, impurity holdout, and a chemistry block; an internal ensemble study evaluates uncertainty; pair-out-of-fold predictions support retrospective site and incorporation-class ranking over existing DFT-relaxed candidates.

- **Claimed problem:** random structure splits hide chemically distinct transfer regimes for impurity formation-energy prediction.
- **Claimed gap:** prior IMP2D modeling does not jointly provide a paired architecture study, axis-resolved transfer, internal UQ, and screening-regret evaluation under one audited protocol.
- **Method/contribution map:** audited modeling set → DART and factorial modules → aligned descriptor/SchNet comparisons → chemical-axis transfer → internal UQ → retrospective ranking.
- **Evidence package:** five paired factorial repeats; fivefold random/pair/host/impurity protocols; chemistry block; descriptor and additive-SchNet baselines; five-member ensemble; 20,000-cluster bootstrap analyses; machine-readable artifacts and an independent 1,346-check numerical audit.
- **Stated limitations:** no new DFT validation, no prospective discovery, UQ is internal, evaluation assumes available relaxed geometries, pair holdout has one singleton-host exception, triplet selection is order sensitive, and the global radial distance is componentwise wrapped rather than a guaranteed shortest image in oblique cells.

## 4. Search And Related-Work Basis

Queries used: PRB author requirements and scope; APS data/AI/supplement policies; defect-formation-energy reference conventions; IMP2D chemical-potential convention; recent PRB graph-Transformer and materials-ML figure structures.
Sources searched: official APS pages, DOI/Crossref metadata, the IMP2D article and supplement, and cited primary literature.
Closest works found: Davidsson *et al.* on IMP2D; Kesorn *et al.* and El Alouani *et al.* on IMP2D modeling/data engineering; Kazeev *et al.* on 2D defect representations; SchNet and the cited CrystalFormer-style radial attention; Freysoldt *et al.* and Komsa–Krasheninnikov for defect-energy physics.
Unverified related-work risks: the current Related Work does not explicitly compare the order-sensitive triplet implementation with invariant angular-message constructions, and the search was not expanded into a new benchmark campaign.
Source-quality screening status: all retained DOI-bearing references used for the PRB physics rewrite were checked against primary or authoritative metadata; one previously nonexistent DefiNet citation had already been removed.

## 5. Expected Review Outcome

Expected outcome: **reject in the current snapshot (3/10), with a scientifically salvageable revision path.**
Main accept signal: unusually thorough split-aligned evidence and a clear physical hierarchy across pair, host, impurity, and incorporation-geometry axes.
Main reject signal: the trained angular branch is order sensitive because it truncates to the first 32 ordered neighbor pairs; the current evidence does not establish permutation robustness.
Confidence: **5/5**, because the manuscript, code, split artifacts, raw result bundles, policy pages, and compiled outputs were inspectable.

## 6. Strengths And Weaknesses

### Strengths

- The 10,224-structure population, exclusions, reference-state convention, and relaxed-geometry boundary are now explicit and numerically audited.
- The transfer story is physically interpretable: pair-held-out MAE is 0.450 eV, host/impurity holdout is 0.938/0.844 eV, and pair-held-out interstitial/adsorbate MAE is 0.719/0.322 eV.
- The factorial preserves the negative E result and treats P as a composite intervention, avoiding causal overclaiming.
- Screening is reported as retrospective ranking with regret, not prospective discovery or a replacement for structural relaxation.
- The manuscript clearly separates internal marginal UQ from chemical-shift calibration.
- The evidence artifacts, seeds, split definitions, and executable numerical checks are unusually strong for a materials-ML paper.

### Weaknesses

Weakness: the angular-message set is selected by stored neighbor order after a 32-pair cap.
Evidence basis: `src/graph.py` enumerates ordered pairs and stops after 32; sampled IMP2D structures trigger the cap at every center.
Reviewer deduction: this can introduce an atom-order pseudofeature and contradicts the expected invariance of atomistic graph models.
Required fix: either implement an invariant construction and rebuild the evidence package, or provide a targeted, preregistered permutation-sensitivity analysis and narrow the method claim with author approval.

Weakness: the global radial bias does not use a true shortest-image distance in oblique cells.
Evidence basis: fractional differences are wrapped componentwise; static checks on hexagonal IMP2D cells found many pair distances different from exhaustive minimum-image distances.
Reviewer deduction: the “geometric” interpretation is weaker than the original Methods stated.
Required fix: retain the corrected description and decide whether a true shortest-image implementation requires a new model/evidence package.

Weakness: PRB significance remains narrower than the technical evidence volume.
Evidence basis: all results are internal to one existing neutral-defect database, without external target-matched DFT or a demonstrated new materials-physics mechanism.
Reviewer deduction: a skeptical PRB referee may view the work as protocol/model engineering more suited to PRM.
Required fix: keep the physics-first hierarchy and avoid claiming broader defect physics than the transfer/error evidence supports.

Weakness: submission metadata and figure provenance are incomplete.
Evidence basis: author/funder/release fields remain unresolved; manual figures lack tool/version/rights records; Fig. 1 retains two baked-in ambiguous phrases.
Reviewer deduction: the package cannot be submitted even if the scientific blocker is resolved.
Required fix: complete the author-controlled records and correct the source artwork through the approved editing workflow.

## 7. Potentially Missing Related Work

Work: permutation-invariant angular-message and geometric atomistic graph constructions.
Status: unverified as a complete search; neighboring model literature is cited, but this implementation risk is not positioned.
Why relevant: the 32-pair ordering issue directly affects the method's soundness.
Overlap: invariant angular aggregation addresses the exact failure mode found in `src/graph.py`.
Needed comparison: a conceptual implementation comparison is required; a new empirical baseline is not necessary unless the model is rebuilt.

Work: shortest-image treatment for nonorthogonal periodic cells.
Status: implementation-derived risk; no new paper citation is required to establish the geometric fact.
Why relevant: most retained 2D hosts use oblique/hexagonal cells.
Overlap: a correct periodic metric changes the global radial bias.
Needed comparison: document the present implementation and, if revised, verify old/new distances and resulting model evidence.

Work: external target-matched defect-formation-energy evaluation.
Status: acknowledged as missing and outside the frozen evidence package.
Why relevant: it would test transfer beyond IMP2D's reference convention.
Overlap: none of the current internal splits supplies an external reference convention.
Needed comparison: only a future, consistently relaxed and reference-matched set; it should not be invented for the current manuscript.

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Pair-held-out transfer is easier than host/impurity holdout | Abstract, Results, Discussion, Conclusion | Complete OOF predictions and aligned CIs; one of 10,224 pair-test rows lacks a training-seen host | Strong after qualification | Valid within IMP2D, not an absolute “all constituents seen” protocol | Keep “all but one/almost entirely” wording |
| Interstitial configurations are harder than adsorbates | Abstract, Results, Discussion | Pair-OOF class MAE 0.719 vs 0.322 eV; 316 vs 33 identity exclusions disclosed | Moderate–strong | Descriptive class asymmetry, not a proven physical mechanism | Retain selection-bias qualification |
| G and P provide small improvements; E is unresolved | Results and generated factorial table | Five paired repeats, orthogonal contrasts and bootstrap intervals | Strong for the evaluated factorial | P does not isolate normalization from gating | Keep composite-P language |
| DART's geometry is physically invariant | Original Methods claim; now withdrawn | Code audit shows order-sensitive triplet cap and non-shortest global wrapping | Weak/failing | Central method-soundness concern remains after prose repair | Resolve via invariant implementation/evidence or targeted sensitivity decision |
| SchNet additive readout contributes to large host-held-out error | Methods, Discussion, Supplemental Material | Controlled mean-readout sensitivity across five folds | Moderate | Supports this recipe only; does not prove formation energy is strictly local or addition universally wrong | Keep recipe-specific wording |
| Screening is useful for materials decisions | Results and Conclusion | 91.0% class preference accuracy and 0.139 eV mean global regret | Moderate | Applies only to retained, already relaxed IMP2D candidates | Keep retrospective/conditional boundary |
| UQ is calibrated | Results and Discussion | Split conformal coverage on an internal ensemble split | Moderate | Marginal internal calibration, not host/impurity-shift coverage | Do not elevate into the abstract headline |
| Data/software are reproducible and public | Data Availability Statement | Local artifacts and GitHub citation to an unpushed commit | Weak at submission level | Current citation is not publicly resolvable and repository license is unresolved | Publish an author-approved archival release and persistent identifier |

## 9. Experiment / Benchmark / Reproducibility Audit

- **Baselines:** validation-selected descriptor models and additive-readout periodic SchNet are aligned to the same splits. The comparison is correctly described as full recipes, not isolated backbones.
- **Ablations:** the paired (2^3) factorial is a strong design. It isolates G/E/P contrasts at the intervention level, but P remains composite.
- **Datasets/benchmarks:** IMP2D provenance and exclusions are clear. No external benchmark is claimed. Pair grouping has one singleton-host exception, now explicitly audited.
- **Metrics:** MAE, paired absolute-error contrasts, macro errors, low-energy recall, calibration/coverage, AURC, preference accuracy, and regret cover both prediction and ranking.
- **Statistical rigor:** repeated paired runs and cluster bootstraps are appropriate descriptive summaries; the paper avoids formal large-sample significance claims.
- **Robustness/failure cases:** host/impurity transfer, geometry-class errors, low-energy behavior, internal UQ, and readout sensitivity are covered. Atom-permutation sensitivity is the decisive missing robustness check.
- **Implementation details:** architecture, seeds, schedules, pooling, UQ calibration, triplet cap, and componentwise wrapping are now stated. Fig. 1 still contains ambiguous baked-in labels.
- **Artifacts and reproducibility:** the independent audit passes 1,346/1,346 checks and targeted paper tests pass 22/22. The final public release and license are absent.
- **Limitations:** the revised text accurately narrows relaxed-geometry screening, internal UQ, pair-split composition, readout inference, and geometric implementation. Disclosure cannot by itself repair the triplet-order risk.

## 10. Multi-Reviewer Panel

Reviewer: Method / Soundness Reviewer
Expertise: atomistic graph models and periodic geometry
Likely score: 2/10
Confidence: 5/5
Main positive signal: compact equations and code-to-text traceability are substantially improved.
Main negative signal: first-32 ordered triplet selection can make predictions depend on stored atom order; componentwise wrapping is not a true oblique-cell minimum image.
Evidence basis: `src/graph.py`, Methods equations, and static structure checks.
Fatal concern if any: unresolved permutation robustness.
Score-change condition: invariant triplet handling with rebuilt evidence, or a convincing preregistered sensitivity package and a deliberately narrower claim.

Reviewer: Evidence / Experiment Reviewer
Expertise: materials-ML evaluation and statistics
Likely score: 6/10
Confidence: 5/5
Main positive signal: aligned multi-axis splits, paired factorial repeats, negative ablation result, cluster bootstrap, UQ, and screening regret form a rich evidence package.
Main negative signal: no permutation-sensitivity evidence addresses the method risk revealed by code audit.
Evidence basis: archived predictions, result tables, split files, and 1,346-check audit.
Fatal concern if any: none in the reported numerical aggregation itself.
Score-change condition: add only the targeted evidence needed for the order-sensitivity decision; do not reopen unrelated experiments.

Reviewer: Evidence / Ablation Reviewer
Expertise: controlled ablations and causal interpretation
Likely score: 6/10
Confidence: 5/5
Main positive signal: G/E/P is fully crossed and paired; the manuscript preserves E as inconclusive.
Main negative signal: P bundles pre-normalization and distance gating, and E9 is post hoc.
Evidence basis: factorial table/narrative and SchNet readout supplement.
Fatal concern if any: none.
Score-change condition: maintain composite-P and recipe-specific E9 claims; no additional generic ablation is required.

Reviewer: Novelty / Positioning Reviewer
Expertise: defect ML and PRB editorial fit
Likely score: 4/10
Confidence: 4/5
Main positive signal: the joint transfer hierarchy and geometry-class asymmetry are more distinctive than a simple DART leaderboard.
Main negative signal: the work remains an internal study of an existing database, and the new physical insight may be judged incremental for PRB.
Evidence basis: Introduction, literature-positioning audit, and PRB scope.
Fatal concern if any: none independent of method soundness.
Score-change condition: keep claims physics-first and show that the transfer hierarchy, not model branding, is the durable contribution.

Reviewer: Writing / Clarity Reviewer
Expertise: PRB manuscript narrative and figures
Likely score: 7/10
Confidence: 5/5
Main positive signal: the title, abstract, Introduction, Discussion, and Conclusion now tell one chemical-transfer story and remove defensive workflow prose.
Main negative signal: Fig. 1 contains ambiguous baked-in wording, and some generated assets retain internal terms such as “Locked test.”
Evidence basis: current TeX, figure manifests, and rendered PDF.
Fatal concern if any: none.
Score-change condition: correct source artwork and complete final-size accessibility QA.

Reviewer: Ethics / Reproducibility Reviewer
Expertise: research integrity, AI disclosure, and artifact release
Likely score: 4/10
Confidence: 5/5
Main positive signal: substantive Codex use is disclosed in Methods and Acknowledgments; the paper has executable evidence checks.
Main negative signal: the cited commit is not public, the license is unresolved, and manual-figure creation rights/AI provenance are unconfirmed.
Evidence basis: Data Availability Statement, repository state, AI policy, and figure manifests.
Fatal concern if any: package-level no-go, not evidence fabrication.
Score-change condition: author-approved archival release, license, creators, and complete figure provenance.

Reviewer: Domain Application Reviewer
Expertise: defect energetics and two-dimensional materials
Likely score: 5/10
Confidence: 4/5
Main positive signal: class preference and regret are useful summaries once relaxed candidates exist.
Main negative signal: no structure generation, relaxation, thermodynamics, kinetics, or prospective validation is addressed.
Evidence basis: Methods target definition and screening Discussion.
Fatal concern if any: none if the retrospective boundary is maintained.
Score-change condition: retain the current conditional wording; do not imply full discovery acceleration.

Reviewer: Reproducibility Reviewer
Expertise: code, data, and computational provenance
Likely score: 6/10 locally; 3/10 as a public package
Confidence: 5/5
Main positive signal: local artifacts, seeds, hashes, tests, and numerical regeneration are comprehensive.
Main negative signal: the final commit, DOI, creators, license, and source archive are absent; the tracked zip is stale PRM content.
Evidence basis: Git state, DAS, checklists, and old source archive.
Fatal concern if any: submission package cannot yet support its availability claim.
Score-change condition: regenerate the source package only after the author-controlled release milestone.

Reviewer: Novice Advocate Reviewer
Expertise: accessibility for a new PRB reader
Likely score: 6/10
Confidence: 4/5
Main positive signal: the abstract now communicates the main numerical hierarchy without requiring knowledge of internal milestones.
Main negative signal: eight main-figure numbers and dense Methods details may still obscure the one-sentence physical result; Fig. 2 relies partly on color.
Evidence basis: rendered manuscript and captions.
Fatal concern if any: none.
Score-change condition: fix the figure wording/noncolor encoding while preserving the current narrative hierarchy.

Reviewer: AC / Meta-Reviewer
Expertise: decision synthesis
Likely score: 3/10
Confidence: 5/5
Main positive signal: the paper is unusually transparent and the PRB rewrite materially improves claim discipline.
Main negative signal: the permutation-risk concern cannot be averaged away by strong reporting or prose.
Evidence basis: all reviewer lenses and code-level red-team audit.
Fatal concern if any: unresolved atom-order sensitivity in the trained method.
Score-change condition: a user-approved scientific resolution of this single blocker, followed by a clean submission-package gate.

Agreement: the transfer evidence is well organized, the scope is now honest, and the local reproducibility package is strong.
Disagreement: reviewers differ on PRB significance and on whether targeted sensitivity evidence could suffice without rebuilding the model.
Decisive positive axis: axis-resolved chemical transfer plus geometry-specific error and retrospective ranking.
Decisive negative axis: order-sensitive triplet selection in an atomistic graph model.
Unresolved evidence: prediction sensitivity to atom reindexing and the impact of correct shortest-image global distances.
AC stance: reject/current NO-GO; invite a scientifically focused repair rather than another broad experiment campaign.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | fatal | First-32 ordered triplet cap introduces atom-order sensitivity | `src/graph.py`; cap triggers broadly in sampled structures | Soundness | method/soundness | User decides invariant rebuild versus targeted permutation-sensitivity package | `analysis-campaign` + `decision` | Resolving it could move overall 3→5 |
| C2 | major | Componentwise wrapping is not a shortest-image metric in oblique cells | `src/graph.py`; hex-cell static checks | Soundness | method/soundness | Keep corrected disclosure; decide whether code/evidence repair is necessary | `decision` | A verified resolution improves soundness by about one point |
| C3 | moderate | PRB physics significance may be judged incremental | One database; no external physical mechanism/validation | Significance | venue-mismatch | Preserve transfer/geometry physics story and bounded claims | `ccf-paper-writer` | Better positioning may move significance 2→3, not cure C1 |
| C4 | major | DAS cites an unpushed commit and no persistent licensed release | Git state and bibliography | Reproducibility | reproducibility | Publish author-approved tagged archive/DOI/license/creators | `finalize` | Required for public reproducibility 3→4 |
| C5 | major | Full author, funder, declaration, and cover-letter data are missing | `main.tex`, checklist, cover letter | Compliance | ethics/limitations | Authors supply verified metadata | human author | Removes package NO-GO only |
| C6 | major | Manual-figure creation rights and AI history are unknown | `manual_figure_manifest.json`; visible marks | Ethics / presentation | ethics/limitations | Authors verify tool/version/rights; add caption AI disclosure if applicable | human author + `ccf-visual-composer` | Required for ethical submission |
| C7 | moderate | Fig. 1 baked-in labels misstate embedding dimensionality and 12-Å semantics | raster artwork vs code | Clarity / soundness | writing | Correct source artwork through approved AutoFigure/editor path | `ccf-visual-composer` | Improves clarity; prevents method confusion |
| C8 | moderate | Fig. 2 has weak grayscale redundancy | grayscale inspection | Accessibility | writing | Add shape/pattern redundancy and rerun visual QA | `ccf-visual-composer` | Improves clarity 3→4 |
| C9 | minor | Old `main_latex729.zip` contains superseded PRM sources | archive inspection | Reproducibility | reproducibility | Exclude it; regenerate only from final PRB snapshot | `ccf-submission-checker` | Prevents an avoidable upload error |

## 12. AC / Meta-Review

Reviewer consensus: the revised manuscript is clearer, more honest, and better aligned to PRB than the previous method-led version. The numerical evidence itself is extensive and internally consistent.
Reviewer disagreement: a sympathetic evidence reviewer sees a borderline PRB paper after narrow repair, whereas the soundness reviewer regards invariant reconstruction as necessary.
Decisive acceptance axis: a reproducible physical hierarchy of transfer difficulty across pair, host, impurity, and incorporation geometry.
Decisive rejection axis: the method can select different angular neighborhoods after atom reindexing, which challenges a foundational assumption of atomistic graph modeling.
AC stance: reject/current NO-GO. Do not reopen every experiment. Resolve C1 and decide C2, then repeat the same focused review and submission gate.
Discussion risks: even after a technical resolution, editors may still prefer PRM unless the paper consistently presents the chemical-transfer hierarchy as the principal physics contribution.

## 13. Quantitative Scores

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty | 3 | 4 | Introduction and closest-work audit | Combination is useful but components and dataset are known; clearer invariant-method delta could raise it |
| Soundness | 1 | 5 | `src/graph.py`, Methods, red-team static checks | Order-sensitive cap is near-fatal; resolving it is mandatory for 3+ |
| Evidence | 4 | 5 | factorial, transfer, UQ, screening, 1,346 checks | Broad and aligned; missing permutation sensitivity prevents 5 |
| Significance | 2 | 4 | PRB scope and Discussion | Internal database study with limited new physics; a defensible technical repair plus sharper physics framing could reach 3 |
| Clarity | 4 | 5 | rewritten paper and rendered PDF | Strong narrative; source-artwork wording/accessibility prevents 5 |
| Reproducibility | 3 | 5 | local artifacts versus public release state | Excellent locally, incomplete publicly; archival DOI/license could raise it to 4 |
| Ethics / Limitations | 4 | 5 | AI disclosures and limitation sections | Honest disclosure; figure provenance and author declarations remain open |

Quality: 3/5
Clarity: 4/5
Significance: 2/5
Originality: 3/5
Soundness: 1/5
Evidence: 4/5
Reproducibility: 3/5
Ethics / Limitations: 4/5
Overall: **3/10**
Confidence: **5/5**
Score-change conditions: resolving C1 can move the paper into the 5/10 borderline band; resolving C2 and submission/release gates could support a further one-point movement, but prose alone cannot do so.

## 14. Questions For Authors

1. Do the authors require strict atom-permutation robustness as part of the DART claim, or will they authorize a targeted permutation-sensitivity gate to determine whether the frozen model is usable?
2. If the triplet construction is corrected, are the authors willing to treat this as a new evidence milestone rather than a continuation of the current frozen J-stage manuscript?
3. Do the authors consider the componentwise-wrapped global bias scientifically acceptable for the present claim, or should true shortest-image geometry be a co-required repair?
4. How were `figure1_manual.png`, `figure2_manual.png`, and `figure3_manual.png` created, under what tool versions and rights, and did generative AI contribute?
5. What are the verified full author list, affiliations, corresponding author/email/ORCID, funder award, conflict declaration, and submission history?
6. What author-approved license and archival repository will support the final code/data citation?

## 15. Score Revision Criteria

Raising the score would require: a documented resolution of atom-order sensitivity; a decision and evidence boundary for oblique-cell distances; corrected figure text/accessibility; and a public, licensed, persistent release with complete author declarations.
Lowering the score would be triggered by: measurable large prediction changes under benign atom reindexing, discovery that manual figures lack usable rights, or inability to reproduce the frozen artifacts from the released code.
Concerns unlikely to change before submission: the absence of an external target-matched DFT set and the inherent editorial risk that PRB may view the work as more methodological than physics-discovery oriented.

## 16. Action Plan And CCFA Handoffs

Priority: P0
Action: stop at the J-stage scientific gate and obtain the user's decision on triplet-order sensitivity.
Owner skill: `decision`
Input needed: this report and the code-level red-team evidence.
Expected output: explicit branch choice—targeted permutation audit, invariant rebuild, or stop/retarget.
Handoff required: yes

Priority: P0/P1
Action: if authorized, design only the minimal permutation and oblique-cell evidence package; do not repeat unrelated E1–E9 experiments.
Owner skill: `analysis-campaign`
Input needed: frozen model checkpoints, canonical structures, and preregistered thresholds.
Expected output: decision-grade sensitivity artifacts and a new milestone record.
Handoff required: yes

Priority: P1
Action: correct Fig. 1 wording, add noncolor encoding to Fig. 2, and verify manual-figure provenance.
Owner skill: `ccf-visual-composer`
Input needed: editable source or approved AutoFigure credentials/workflow, plus author rights/AI history.
Expected output: new versioned figures, captions, provenance manifest, and final-size QA.
Handoff required: yes

Priority: P1
Action: complete author metadata, declarations, archival release, DOI, creators, and license after scientific GO.
Owner skill: `finalize` + `ccf-submission-checker`
Input needed: verified author-controlled information.
Expected output: public citation, clean source package, completed checklist, and upload-ready PDFs.
Handoff required: yes

Priority: P2
Action: perform a final PRB prose and page-render pass only after the scientific decision.
Owner skill: `ccf-paper-writer` + `pdf`
Input needed: approved final evidence boundary and corrected figures.
Expected output: final manuscript/SM, line-clean build, and page-by-page visual QA.
Handoff required: no

Checks run: full source/code/artifact review; official PRB and APS policy verification; DOI/reference audit; 1,346/1,346 numerical checks; 22/22 targeted tests; LaTeX compilation; citation/reference/font diagnostics; post-edit humanization review; scientific red-team review.
Checks skipped: no new training, DFT, GPU computation, model inference, atom-permutation sensitivity run, or shortest-image retraining; these are outside the user-authorized J-stage pass.
Unresolved risks: triplet-order sensitivity, oblique-cell radial metric, PRB significance, manual-figure rights/AI history, public release/license, and author-controlled metadata.
