# Reviewer comment matrix: deployment loop and PRB physics depth

Status date: 2026-08-10
AutoSci state: J-R0 review-response planning
Execution state: **no scientific run authorized**

This matrix atomizes the two reviewer comments into evidence-bearing revision
items. It is a planning artifact, not an author response and not evidence that
any proposed experiment has been completed.

## Trust-ranked evidence inventory

| Asset | Trust rank for this revision | Permitted use | Prohibited use |
|---|---|---|---|
| `artifacts/prm_protocol_v2/`, canonical IMP2D pickle and raw `imp2d.db` | Primary | Final DFT-relaxed structures, audited targets, raw workflow scalars and split identity | Initial geometries, ionic trajectories, or wall-time savings |
| `artifacts/prm_results/materials/sample_predictions.csv` and archived host/dopant OOF predictions | Primary | Canonical sample-level residuals and paired transfer analysis | Prospective deployment or causal mechanism claims |
| `results/phase_a_*`, `results/phase_b_*` and their scripts | Implementation reference only | Reuse of descriptor code after rewriting it against protocol v2 and canonical OOF predictions | Paper numbers; these assets use an old random split, old baseline and 1,065 samples |
| Historical C17 candidate, QE and “prospective DFT” assets | Rejected as paper evidence | Negative lessons and code archaeology only | Blind/prospective validation, deployment acceleration, or target-matched external validation |
| `results/mace_mp_validation.json` | Reference only | Shows that an old MACE energy proxy was not an adequate label surrogate | Evidence for MACE-relaxed geometries; no such relaxation was tested |

The raw IMP2D release contains final relaxed structures and scalar workflow
fields, including `en1`, `en2`, `depth`, and `extension_factor`. It does not
contain the initial or intermediate atomic coordinates required for an ionic
snapshot analysis. `en1` is the total energy of the first workflow stage and
must not be relabeled as an initial-geometry target.

## Comment-to-action mapping

| ID | Reviewer issue | Current evidence answer | Required action | Acceptance evidence | Status |
|---|---|---|---|---|---|
| D1 | What produces the deployed “relaxed geometry”? | In the reported benchmark it is the final IMP2D DFT-relaxed structure. No alternative geometry source was evaluated. | Add an explicit geometry-provenance paragraph to Methods and Discussion. | The abstract, Methods, Discussion, Conclusion, captions and cover letter give the same answer. | Proposed |
| D2 | If DFT produces the geometry, what cost is saved? | No saving of the dominant DFT relaxation cost has been established; the final defect energy is normally produced by that workflow. | Delete general cost-reduction and screening-acceleration language. | Repository-wide search finds no unqualified acceleration claim. | Proposed |
| D3 | Is database completion a valid use case? | Only as a potential missing-label or consistency-check use case when a geometry exists independently of its energy label. It has not been deployed here. | Reframe the evaluated task as “retrospective energy estimation or reranking of available relaxed geometries.” | The phrase and its limitations are used consistently; potential use is not presented as validated deployment. | Proposed |
| D4 | Does DART tolerate a low-cost geometry distribution? | Unknown. Current formal configurations used final relaxed geometries and no online augmentation. | After the graph-correctness gate, run a preregistered low-cost-geometry experiment only if the geometry provenance is defensible. | Geometry source, failures, convergence, MAE change, ranking change and the low-cost-potential comparator are archived. | Blocked by G1 |
| D5 | Can unrelaxed or early DFT snapshots establish savings? | No matching coordinates or trajectories exist in the released database. | Treat an early-snapshot campaign as a separate, optional DFT milestone; do not infer it from `en1`. | New candidates are frozen before labels, target convention is reproduced, all ionic snapshots and wall times are saved, and contemporaneous DFT energy is a comparator. | Deferred; no DFT authority |
| P1 | Error versus coordination and local distances | The canonical structures and audited 5-Angstrom graph support this analysis. | Compute canonical pair-OOF associations using the exact E-module descriptors plus independent packing proxies. | Full-population result, clustered confidence intervals and fold stability are archived. | Proposed |
| P2 | Error versus local void size | `d_max` is not a void size. | Define a post-relaxation impurity-centered clearance/free-radius proxy before viewing errors. | Definition, units, periodic construction and sensitivity variants are frozen. | Proposed |
| P3 | Radius, electronegativity and valence mismatch | Fixed element tables and unique impurity identity support the analysis. | Use signed and absolute mismatch features with multiplicity control and covariate adjustment. | Geometry and chemistry blocks are jointly tested; individual effects use BH-FDR. | Proposed |
| P4 | Host-family decomposition | Forty-four retained hosts can be partitioned, but the retained set contains no nitride host. | Freeze an exhaustive taxonomy: chalcogenide; carbide/MXene-like; elemental/hydrogenated; halide/halochalcogenide; oxide. | Every host maps once; identity counts and sample counts are reported; no invented nitride panel. | Proposed |
| P5 | Impurity-series decomposition | The 65 impurities support 3d, 4d, 5d, main-group and f-block/other labels. | Report sample-weighted and identity-macro errors with identity-clustered intervals. | Inference only for groups with at least five identities and 100 samples. | Proposed |
| P6 | Why host holdout is harder than impurity holdout | The nominal pooled gap is only 0.094 eV and has not been shown significant. | First compare sample-matched OOF absolute errors; then test structural-novelty versus chemical-novelty differences. | A paired clustered interval establishes the gap, and novelty-gap association is stable in at least four of five folds; otherwise retain “not isolated.” | Proposed |
| P7 | Concrete high- and low-error structures | Canonical pair-OOF residuals map back to all structures. | Freeze a deterministic, matched case-selection algorithm before rendering structures. | No repeated host/impurity among high-error cases; controls are matched; all selected IDs and rejected cases are logged. | Proposed |
| P8 | Relaxation amplitude or local distortion | Atomic displacements are unavailable. The database-defined `extension_factor` can support an out-of-plane expansion/reconstruction proxy; `depth` and `conv2` are sensitivity variables. | Analyze `abs(log(XF))` and the `XF>2` exclusion sensitivity, while explicitly withholding any atomic-displacement claim. | Full and `XF<=2` results agree in direction or the claim is withdrawn. | Proposed |
| P9 | Independent external DFT validation | Current historical QE results are not target matched and are no longer blind. | Do not reuse them. A new external set requires its own preregistered DFT milestone. | Identical formation-energy convention, consistent relaxation settings, frozen candidates and untouched labels. | Deferred |

## Cross-cutting correctness blockers

The reviewer-requested analyses do not supersede two pre-existing model risks:

1. The local angle branch retains the first 32 ordered neighbor pairs and is
   not mathematically permutation invariant when the cap is reached.
2. The global radial bias uses componentwise fractional wrapping, which is not
   a guaranteed shortest periodic image for oblique cells.

A permutation/MIC inference audit is useful for sizing the problem, but even a
small empirical prediction change does not turn the current implementation
into a strictly invariant one. The PRB repair route therefore requires an
invariant triplet construction, a verified exact minimum-image distance and a
new evidence milestone before the new physical analysis is called canonical.

## Shortcuts explicitly rejected

- Do not call random coordinate noise an experimental unrelaxed distribution.
- Do not construct a “pristine” structure by removing the impurity from the
  final relaxed defect and call it the original geometry.
- Do not use `en1-en2` as an atomic relaxation displacement.
- Do not use the old 1,065-row random-split descriptor analysis as DART OOF
  evidence.
- Do not revive the 37 historical QE single points: they use an inconsistent
  isolated-atom reference, a post-hoc same-set dopant offset, a selected
  interstitial-only subset and an obsolete model.
- Do not claim that correlations with E-module inputs establish why the E
  module works; its independent factorial effect was unresolved.
- Do not add a nitride host panel when no retained nitride host exists.
