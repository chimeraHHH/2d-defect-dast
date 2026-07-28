# Pre-result skeptical review

## Review mode

- Review follow-up policy: auto-execute required non-DFT follow-ups.
- Manuscript edit mode: LaTeX required.
- Manuscript source status: reviewable, result-gated draft; not submission-ready.

## Summary

The manuscript has a defensible PRM route if it is presented as an audited
evaluation of impurity-incorporation transfer and applicability domains rather
than as the first defect graph network or a universal architecture advance.
The dataset audit, paired factorial, immutable split contract, and
held-out-calibration design are the strongest elements. The numerical
comparison, uncertainty, and screening claims remain fail-closed until both
downstream campaigns and all collectors finish.

The three highest risks are:

1. the principal transfer, UQ, and screening claims currently lack complete
   result bundles;
2. the neural baseline set is limited to periodic SchNet, so claims must stay
   within split-aligned comparison rather than broad state-of-the-art
   superiority;
3. the scientific domain is neutral IMP2D adsorbates/interstitials, without
   prospective DFT validation, charged defects, or arbitrary unseen
   chemistries.

## Strengths

- The target formula, raw components, impurity identity, duplicate groups, and
  split membership are explicitly audited.
- The full paired factorial separates module evidence from sequential
  test-driven ablation.
- Architecture, descriptor family, and checkpoints are selected without test
  labels.
- Calibration and test partitions are distinct, with marginal-coverage limits
  stated explicitly.
- Screening metrics are tied to out-of-fold predictions and do not claim new
  material discovery.

## Key issues

### R1: Canonical downstream evidence is incomplete

- Risk: critical until both 48-run queues complete and the collector verifies
  every manifest, split, prediction, metric, and hash.
- Route: finish E4--E7 in `paper_experiment_matrix.md`; never narrate partial
  queue metrics.
- Acceptance criterion: 48 DART and 48 SchNet runs complete with zero failures,
  followed by complete comparison, UQ, materials, and paper-asset manifests.

### R2: Baseline breadth limits the architecture claim

- Risk: major if the paper claims general neural state of the art; moderate
  under the current scoped claim.
- Evidence: the aligned neural comparator is periodic SchNet; descriptor
  families and the complete G/E/P factorial provide additional controls. DART
  also uses fixed ct-UAE vectors and JARVIS initialization that SchNet does not.
- Route: report paired split-aligned differences and explicitly call SchNet and
  descriptors strong comparators rather than an exhaustive architecture
  benchmark. Treat DART--SchNet as an end-to-end recipe comparison rather than
  a pure backbone effect. Do not compare against literature point estimates as
  if they used the same audited rows and partitions.

### R3: Novelty can be overstated easily

- Risk: major if framed as the first defect-aware GNN or first IMP2D machine
  learning study.
- Evidence: Kesorn et al. and El Alouani et al. already model IMP2D; Kazeev et
  al. model a distinct 2D defect database; several graph models already predict
  bulk defect energetics.
- Route: lead with the combined audit, paired factorial, transfer partitions,
  held-out UQ, and decision-level screening analysis. The verified boundary is
  recorded in `literature_positioning.md`.

### R4: Submission metadata remains incomplete

- Risk: blocking only at final packaging.
- Route: obtain the full author list and affiliations; replace the provisional
  repository sentence with a reference to the final tagged release and any
  author-supplied archival DOI; add the contact-author email required by APS.
  Do not invent any of these fields.

### R5: Grouped transfer is not disjoint from source-task chemistry

- Risk: major if host- or impurity-held-out results are described as completely
  unseen chemistry.
- Evidence: the clean overlap audit at commit `3a98513` finds reduced-formula
  matches for 26 of 44 IMP2D host labels (25 of 42 unique reduced formulas)
  among 69 JARVIS records; all 65 IMP2D impurity elements occur in the source
  corpus.
- Route: define grouped holdout as absence from IMP2D target supervision,
  distinguish formula overlap from structural identity, and retain the
  end-to-end DART--SchNet comparison boundary.

## Priority revision plan

1. Complete and collect the two downstream campaigns without exposing partial
   test metrics.
2. Run comparison, UQ, and materials analyses; generate the fail-closed result
   assets.
3. Rewrite the abstract last, using only canonical effect intervals,
   transfer/comparator results, calibration, and screening metrics.
4. Cross-check every manuscript number against machine-readable files and
   inspect every rendered page.
5. Run a final independent review, resolve author/release metadata, and create
   the tagged GitHub release.

## Novelty and related-work matrix

| Topic | This paper | Closest prior work | Residual value |
|---|---|---|---|
| IMP2D modeling | audited graph and descriptor evaluation | Kesorn 2024; El Alouani 2026 | duplicate/identity control plus aligned chemical-transfer partitions |
| 2D defect representation | explicit impurity node and periodic graph | Kazeev 2023 on 2DMD | distinct database, target, and split question |
| Defect graph prediction | neutral incorporation formation energy | Witman 2023; Rahman 2024; Fang 2025; Kiyohara 2025 | IMP2D-specific audit, factorial, transfer, UQ, and screening protocol |
| Dataset redundancy | coordinate and invariant-distance duplicate groups | MD-HIT 2024 | concrete integration into the IMP2D split contract |

## Current judgment

Continue. There is no publishability stop-loss at the protocol stage, but the
paper cannot be judged submission-ready until the pending numerical evidence
passes collection and the final claims are audited against it.
