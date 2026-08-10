# J-stage figure redesign and manuscript contract

Status: **J-stage manuscript optimization**.  This pass does not authorize new
training, DFT, data selection, or numerical analysis.  Every scientific number
continues to come from the frozen E1--E9 evidence chain and the hash-bound paper
assets.

## Journal figure survey

The survey used recent APS papers whose scientific role is close to this work:
materials machine learning, graph/Transformer models, chemical transfer,
uncertainty, or screening.  Figure counts refer to main-text figures.

| Journal | Sample summary | Stable visual sequence |
|---|---|---|
| Physical Review Materials | Six 2024--2026 papers, 58 figures total; range 6--14, median 9 | Scientific object/workflow or architecture -> data/protocol -> benchmark -> representation/error diagnosis -> transfer/UQ/failure mode -> materials-facing validation |
| Physical Review B | Five 2024--2025 papers; 4--7 figures | Physics/method overview -> architecture or protocol -> controlled validation/ablation -> transfer or robustness -> physical application |
| Physical Review Letters | Five 2024--2025 papers; typically 3--5 figures | Mechanism/workflow -> representative example -> decisive quantitative result -> physical interpretation or screening consequence |

Representative PRM examples:

- [SHCTransformer](https://journals.aps.org/prmaterials/abstract/10.1103/jn3l-hp93):
  method overview, detailed architecture, prediction validation, embeddings,
  independent validation, and material cases.
- [Dispersion-corrected ML potentials for 2D van der Waals materials](https://journals.aps.org/prmaterials/abstract/10.1103/cl8c-8f1f):
  database workflow, unified error diagnostics, reference uncertainty, and
  representative structure/band cases.
- [Active learning for strongly anharmonic materials](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.9.063801):
  workflow, uncertainty definition, failure modes, learning curves, and
  downstream physical consequences.
- [OmniXAS](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.9.043803):
  data workflow, representation, global benchmark, paired transfer gains, and
  environment-resolved transfer.
- [ANN descriptors for lattice defects in Si](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.8.103805):
  architecture, residual distributions, learned descriptors, scaling, and
  architecture comparisons.
- [Defect exploration in amorphous silica](https://journals.aps.org/prmaterials/abstract/10.1103/597x-tzfd):
  structures, energy validation, successful/failing cases, and a correction
  workflow.

Representative PRB examples:

- [SCALINN Transformer for DMFT](https://journals.aps.org/prb/abstract/10.1103/bk5q-pfb2):
  physical loop, operating modes, architecture, attention mechanism,
  truncation/transfer tests, and a self-consistent physics benchmark.
- [High-throughput Fe intermetallics and ML](https://journals.aps.org/prb/abstract/10.1103/brpz-w2tk):
  data distribution, electronic structure, physical trend, heterogeneity,
  sampling repair, and the final ML benchmark.
- [Spin-dependent graph neural network potential](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.144426):
  framework, implementations, component validation, and a phase-transition
  application.
- [Learning the Kondo entanglement cloud](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.195125):
  physical task, target space, benchmark, size transfer, noise robustness, and
  neighborhood/sample sensitivity.
- [Predicting exciton binding energies](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.075204):
  proxy audit, relation testing, joint descriptors, outliers, and a screening
  boundary.

Representative PRL examples:

- [Charged-defect formation energies from crystal structures](https://journals.aps.org/prl/abstract/10.1103/h66h-y5k6):
  target protocol, physical-state schematic, parity validation, and a
  screening/case-study closure.
- [Machine-learning-assisted transition-state searches](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.096201):
  workflow, representative processes, and one decisive physical trend.
- [Diffusion in complex materials quantified with ML](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.186301):
  physical setup, method comparison, mechanism distribution, parametric trend,
  and experiment-facing comparison.
- [Discovering high-entropy oxides with an ML potential](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.216101):
  trust benchmark, composition landscape, decision boundary, and prospective
  validation.
- [Minimal training sets for learned potentials](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.167301):
  workflow, scaling, physical-property validation, and structural validation.

The cross-journal conclusion is not to add decorative figures.  A Transformer
paper needs an auditable architecture visual, the main transfer claim needs one
uncompressed hero figure, and purely descriptive heterogeneity belongs in the
Supplemental Material unless it establishes a new mechanism.

## Current eight-figure program

| Main figure | Scientific job | Action |
|---|---|---|
| Fig. 1 | Explain the verified `g111` DART data path and bounded retrospective screening | Use the user-supplied architecture schematic with a scientifically corrective caption |
| Fig. 2 | Distinguish representative adsorbate and interstitial structures | Add the user-supplied overlaid single-impurity schematic in Methods A |
| Fig. 3 | Establish provenance, target range, chemistry coverage, and frozen partitions | Retain the protocol overview |
| Fig. 4 | Explain the four-feature E branch at the impurity node | Add the user-supplied environment-enrichment schematic in Methods C |
| Fig. 5 | Show paired G/E/P evidence and interactions | Retain the AutoFigure editor-only factorial derivative |
| Fig. 6 | Establish axis-dependent transfer and regime-matched comparator gains | Retain the axis-resolved hero figure and explicit broken axis |
| Fig. 7 | Bound internal calibration and selective prediction | Retain coverage, risk--coverage, and uncertainty--error discrimination |
| Fig. 8 | Convert pair-held-out regression into retrospective screening decisions | Retain preference margins, regret, and decision accuracy |

The former main-text error-heterogeneity figure becomes Supplemental Fig. S1.
The main text keeps one boundary sentence and no longer interrupts the
factorial -> transfer -> UQ -> screening argument.

## Figure contracts

### Fig. 1 -- DART architecture

- **Conclusion:** the selected recipe joins a uniquely identified impurity,
  5-angstrom local interactions, all-atom geometric self-attention, and a
  non-extensive graph readout.
- **Archetype:** three-panel method schematic.
- **Target/output:** user-supplied 300-dpi raster PNG, placed at two-column
  width with non-destructive LaTeX whitespace trimming.
- **Panel map:** (a) feature initialization; (b) E, local P blocks, and
  geometric Transformer backbone; (c) G readout, per-structure energy, and
  retrospective class-wise screening.
- **Evidence hierarchy:** promoted `g111` config first; model and graph source
  code second; historical development comments are not evidence.
- **Statistics:** none; this is a code- and config-bound schematic.
- **Source evidence:** `configs/prm/promoted/g111/transfer/id_cv5_f0_seed242.yaml`,
  `src/models/crystal_v2.py`, `src/models/baseline.py`, and `src/graph.py`.
  Asset and pixel hashes are recorded in
  `paper_Q1/review/manual_figure_manifest.json`.
- **Image integrity:** the raster is a schematic with baked-in text; the
  caption, rather than the artwork, carries the final scientific qualifiers.
- **Reviewer risks:** 12 angstrom is a radial-bias grid endpoint, not an
  attention cutoff; G does not prove that attention selects the impurity; E is
  not independently supported; P is a composite intervention.

### Fig. 2 -- representative incorporation classes

- **Conclusion:** the adsorbate and interstitial views are alternative neutral
  single-impurity configurations, not simultaneous defects.
- **Archetype:** overlaid top/side structural schematic in a representative
  WSe$_2$-like host.
- **Target/output:** user-supplied raster PNG at two-column width.
- **Reviewer risk:** Top-W and hollow are examples rather than exhaustive site
  definitions; color is supplemented by position and labels but remains less
  distinct in grayscale.

### Fig. 4 -- defect-environment enrichment

- **Conclusion:** E forms four impurity-centered quantities on the 5-\AA{}
  graph, normalizes and projects them to 128 channels, and adds the result only
  to the impurity-node representation.
- **Archetype:** single-column local-neighborhood schematic.
- **Target/output:** user-supplied raster PNG at column width.
- **Reviewer risk:** the displayed $N=3$ neighborhood is schematic and the
  implementation adds a projected descriptor rather than concatenating raw
  quantities.

### Fig. 5 -- paired factorial

- **Conclusion:** G and P have supported validation effects, E is
  inconclusive, and the selected all-enabled recipe follows a validation-only
  ordering rule.
- **Archetype:** paired point-range plot plus orthogonal-effect forest plot.
- **Target/output:** two-column vector PDF plus PNG.
- **Panel map:** (a) eight readable module combinations; (b) main and
  interaction effects with paired-repeat bootstrap intervals.
- **Statistics/source data:** unchanged canonical factorial bundle.
- **Reviewer risk:** `G+E+P` selection must not be described as proof that all
  three modules are independently beneficial.

### Fig. 6 -- axis-resolved transfer

- **Conclusion:** pair-held-out error remains near random out-of-fold error,
  while complete host or impurity holdout is harder; DART has lower paired
  absolute error than both evaluated comparator recipes in every regime.
- **Archetype:** regime profile plus paired forest plot with a broken x axis.
- **Target/output:** two-column vector PDF plus PNG.
- **Panel map:** (a) DART MAE across five frozen regimes; (b) SchNet-add-minus-
  DART and descriptor-minus-DART paired differences with 95% intervals.
- **Statistics/source data:** canonical pooled metrics and regime-matched
  bootstrap intervals; no recomputation or model rerun.
- **Reviewer risk:** the broken axis must be explicit; the random-MAE guide is
  a visual reference, not an equivalence threshold; chemistry block is not a
  monotonic extrapolation level.

### Supplemental Fig. S1 -- descriptive heterogeneity

- **Conclusion:** pair-held-out error varies across host, impurity,
  incorporation class, and site labels.
- **Archetype:** unchanged descriptive scatter/lollipop figure.
- **Statistics/source data:** unchanged pair-held-out out-of-fold predictions.
- **Reviewer risk:** no subgroup was used for selection and no causal mechanism
  is inferred.

## Explicit exclusions

- No attention map, t-SNE/UMAP, learned-mechanism claim, or cherry-picked
  material case is added because E1--E9 do not support those claims.
- No generated atomic structure is presented as data.  The canonical paper
  bundle does not contain a geometry asset suitable for a traceable material
  rendering, so Fig. 1 uses a labelled schematic.
- No new DFT, GPU training, ablation, baseline, or uncertainty run is required.

## Acceptance gates

1. Every modified scientific number still passes the 1,331-check numeric audit.
2. Both PDFs compile without undefined references, clipping, or overfull boxes.
3. Every figure is inspected at final 183-mm width and remains distinguishable
   in grayscale through labels, marker shape, line style, or hatch.
4. The manual-figure manifest binds Figures 1, 2, and 4 to their paper assets,
   pixel hashes, captions, labels, and source comparisons; the protocol and
   result sidecars continue to bind quantitative figures.
5. Main-text figure order is architecture -> incorporation classes -> protocol
   -> environment enrichment -> factorial -> transfer -> UQ -> screening;
   Supplemental Fig. S1 contains descriptive heterogeneity.

## Execution record

- Canonical result assets were regenerated from clean commit `da099c0`; the
  result manifest records `dirty: false`.
- The protocol sidecar uses schema v3 and hashes the selected config, model,
  global block, graph construction, figure generator, protocol inputs, and all
  PDF/PNG outputs.
- Twenty targeted protocol/result-asset tests passed.
- The independent numeric audit passed 1,331 of 1,331 checks.
- The rebuilt 13-page main PDF and 3-page Supplemental Material have no
  undefined references, overfull boxes, clipping, or unembedded fonts.
- All nine paper-facing figures passed direct render inspection and grayscale
  inspection, with a documented grayscale advisory for Main Figure 2.  No
  scientific calculation or model run was performed.
