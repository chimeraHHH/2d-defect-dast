# PRB visual contract: G0/G3 reconstruction

Status: G0 restructuring implemented; G3 figure contract frozen but not
rendered. This document does not authorize or report a new scientific result.

## Evidence boundary

- No G3 outcome, effect size, confidence interval, family ordering, or case was
  available when this contract was written.
- The current pair-OOF archive may be used only for explicitly exploratory G3
  development until the repaired-model OOF bundle is complete and audited.
- A G3 figure may enter the manuscript only after the preregistered analysis,
  numeric audit, and figure-sidecar checks pass on the GPU server.
- AutoFigure-Edit is restricted to editor-only presentation adjustments after
  the quantitative figure is locked. Generative editing must not move data,
  intervals, axes, thresholds, labels, or selected cases.

## Main-text figure sequence

| Order now | Artifact | Evidence role | Source | Action |
| ---: | --- | --- | --- | --- |
| 1 | DART architecture | Method overview | `figures/figure1_manual.png` | Retain in Introduction; pixels unchanged. |
| 2 | Incorporation classes | Physical task definition | `figures/figure2_manual.png` | Retain in Dataset subsection; pixels unchanged. |
| 3 | Environment enrichment | Method mechanism | `figures/figure3_manual.png` | Retain in DART subsection; pixels unchanged. |
| 4 | Chemical transfer | Primary applicability evidence | `figures/fig_transfer.pdf` | Retain in Results. |
| 5 | Retrospective reranking | Bounded database analysis | `figures/fig_screening.pdf` | Retain with a retrospective caption; rename at the manuscript layer only. |

The protocol overview, factorial visualization, and internal-uncertainty figure
move to Supplemental Figs. S1--S3. Existing descriptive error heterogeneity
becomes Supplemental Fig. S4. The current manuscript therefore contains five
main figures. If G3 passes, its physical-error map is inserted after the
transfer figure and the reranking figure becomes Fig. 6. The resulting target
contains six main figures, below the approved maximum of seven.

## Reserved G3 physical-error map

Artifact: four-panel physical error map

Target venue / format: Physical Review B, two-column `figure*`

Core claim: identify stable, noncausal physical associations between local
geometry, chemical mismatch, database reconstruction, chemical family, and
absolute pair-held-out error.

Reviewer question: why are interstitial predictions harder, and what physical
or chemical regimes contribute to heterogeneous error?

Evidence layer: mechanism/boundary analysis; not a new deployment claim

Source data: the audited join of the 10,224-row canonical sample table,
repaired pair-OOF predictions, exact local descriptors, raw IMP2D `depth`,
`extension_factor`, and `conv2`, plus the frozen host and impurity taxonomies.
Every row must remain traceable by `sample_index`, raw database row ID, fold,
host, impurity, and incorporation class.

Statistics / uncertainty: robust additive or mixed analysis of
`log(abs_error + 0.05)` with crossed host and impurity effects and fold
adjustment; 1,730-pair clustered bootstrap intervals; geometry and chemistry
block tests before BH-FDR-controlled single-feature effects; five-fold
directional stability; sensitivity excluding `extension_factor > 2`.

Figure prototype: asymmetric two-by-two evidence map with direct labels and a
shared legend strip. Use Okabe--Ito blue/orange for adsorbate/interstitial,
with marker shape or line style carrying the same distinction in grayscale.
Use neutral gray for reference lines and secondary categories. Do not use
red/green alone or a rainbow color map.

Panel map:

1. `(a) Local geometry`: adjusted absolute-error contrast over coordination or
   post-relaxation clearance, stratified by incorporation class. Plot only
   preregistered quantities that pass coverage and evidence gates.
2. `(b) Chemical mismatch`: adjusted 10th-to-90th percentile contrasts for
   covalent-radius, electronegativity, and valence mismatch with
   pair-clustered 95% intervals. Show null or unstable estimates without
   selective omission.
3. `(c) Reconstruction proxy`: error association with database-defined
   `extension_factor`; mark `XF = 2`, show the prespecified sensitivity, and
   describe XF as an out-of-plane expansion/reconstruction proxy rather than a
   relaxation displacement.
4. `(d) Chemical families`: host-family and impurity-series macro-MAE forest
   plot. Inferential emphasis requires at least five identities and 100
   samples; smaller groups remain descriptive or move to the supplement.

Exact label inventory: `(a) Local geometry`; `(b) Chemical mismatch`; `(c)
Reconstruction proxy`; `(d) Chemical families`; `Adsorbate`; `Interstitial`;
`Adjusted error contrast (eV)`; `Absolute OOF error (eV)`; `Coordination` or
`Clearance proxy (Å)` as selected before rendering; `Extension factor, XF`;
`XF = 2`; `Host family`; `Impurity series`; `95% pair-clustered interval`.

Caption role: state that panels use repaired pair-held-out OOF residuals,
identify adjustment and clustered uncertainty, distinguish descriptive from
inferential family estimates, state the XF sensitivity, and use
“associated with” rather than “explains” or causal language.

Manuscript placement: Results, immediately after the chemical-transfer figure
and before internal uncertainty and retrospective reranking. The Discussion
may interpret only effects that satisfy the preregistered gate.

Output formats: canonical editable SVG with live text, vector PDF for LaTeX,
300-dpi PNG preview, source-data CSV/JSON, and a numeric figure sidecar binding
every plotted value to the canonical G3 result bundle.

Traceability: G3 analysis manifest -> effect/family tables -> figure sidecar ->
SVG/PDF hashes -> manuscript caption and claim ledger. The figure must fail
closed if any source row, estimate, interval, sample count, label, or output
hash is missing.

## Manual-figure provenance and accessibility boundary

The three manual PNGs are hash-bound in
`review/manual_figure_manifest.json`, but final submission still requires the
authors to attest their creator/source, reuse permission, and absence of
unlicensed third-party elements. Figure 2 uses red and green atoms; its spatial
placement and caption provide redundant identification, but the raster cannot
be made fully color-independent without an author-approved source edit. No
pixel edit is authorized in this reconstruction.
