# AutoSci J-R revision plan for the new reviewer comments

Status date: 2026-08-10
Target: *Physical Review B* Regular Article
Current decision: **G1 and G3 started; PRB submission remains NO-GO**
Execution control: G1 diagnostics/formal repair/one-fold pilot, G3 pipeline and
old-OOF exploratory analysis, and claim-bounded figure/prose restructuring were
approved on 2026-08-10. G2 full repaired OOF, new DFT, release, and submission
still require separate explicit approval.

## Outcome

The two comments are valid and change the revision from a prose-polish task to
a gated scientific revision. The recommended PRB route is:

> truthful task boundary -> graph correctness -> repaired core OOF evidence ->
> materials-physics error analysis -> optional low-cost-geometry evidence ->
> physics-first manuscript rebuild.

Claim contraction is mandatory but is not, by itself, enough for a defensible
PRB submission. If the team does not authorize the graph repair and the
necessary core rerun, the honest route is a retrospective-only paper and a
venue retarget, not a PRB submission with stronger wording.

The current authorization permits G1 diagnostic inference, G1 implementation
tests and one repaired-fold pilot, G3 server-side analysis, and scoped milestone
commits/pushes. It does not permit the G2 full queue, new DFT, a prospective
geometry campaign, release, or submission.

## Route decision

| Route | Scope | Expected outcome | Recommendation |
|---|---|---|---|
| A. PRB repair | G0-G3 required; G4 high-value optional | Corrected model plus a materials-physics mainline; PRB becomes arguable if at least one physical result is stable | **Recommended** |
| B. Retrospective-only | G0 only; preserve current numerical evidence with explicit limitations | Honest database-internal benchmark, but the application and PRB physics case remain weak | Retarget rather than submit to PRB |
| C. Strong deployment | Route A plus G4 and later G5 | Direct answer to geometry-source and cost-saving objections | Defer until suitable initial structures or DFT capability exists |

## G0 — Task-boundary repair

Type: manuscript-only; no scientific computation.

Required edits:

1. Replace the abstract opening with an explicit retrospective definition,
   for example:

   > We study retrospective estimation of impurity formation energies from
   > available DFT-relaxed structures, a setting relevant to missing-label
   > analysis and reranking but not a replacement for structure generation or
   > first-principles relaxation.

2. Remove “can reduce the cost,” general “screening acceleration,” and any
   implication that the current workflow bypasses DFT relaxation.
3. State in Methods that IMP2D's final relaxed geometry and defect total energy
   are products of the same source DFT workflow.
4. Rename the current screening result as “retrospective reranking” or
   “ranking analysis over retained IMP2D candidates.”
5. State in Discussion that no dominant first-principles cost saving has been
   demonstrated when the input geometry itself comes from DFT.
6. Apply the same boundary to the Conclusion, Fig. 1 decision box/caption,
   the current ranking figure/table and the cover letter.

Acceptance gate G0:

- one repository-wide claim audit finds no unqualified cost-saving,
  prospective, discovery or deployment statement;
- all numerical ranking metrics remain explicitly conditional on the retained
  DFT-relaxed IMP2D candidates;
- database completion is described as a possible context, not a validated use.

Milestone: `J-R0-claim-boundary`; commit/push only after user approval and
independent prose audit.

## G1 — Graph correctness and representation gate

This gate precedes every new geometry or physics result. Geometry changes can
alter both the capped triplet set and the global radial distances, so adding
more analyses before this decision would deepen evidence around a potentially
order-sensitive model.

### G1A — Bounded diagnostic

Run on the GPU server from a pushed clean commit:

- regenerate each of the five pair-fold test graphs under 16 deterministic
  random/adversarial atom permutations while preserving impurity identity;
- report per-sample prediction range, 95th/99th percentiles, maximum change,
  pooled MAE change, incorporation-class flips, site-ranking flips and regret
  change;
- compare the stored global radial distances with an independent exact
  shortest-image implementation, stratified by cell obliquity, triplet-cap
  activation and incorporation class.

Suggested practical diagnostic thresholds are 99th-percentile prediction
range below 0.01 eV, maximum change below 0.05 eV, pooled MAE change below
0.01 eV and zero class/site flips. Passing these thresholds establishes only
empirical stability for the tested transformations; it does not establish
mathematical permutation invariance.

Estimated budget: 1-4 GPU-hours; no DFT and no training.

### G1B — Formal repair required for Route A

- replace first-32 ordered-pair retention with a strictly permutation-
  invariant triplet construction/aggregation;
- replace componentwise fractional wrapping with a verified shortest-periodic-
  image implementation for oblique cells;
- add unit/property tests for atom permutation, atom/lattice translation,
  cell obliquity and independent minimum-image agreement;
- run one bounded repaired-model fold as a numerical-stability pilot.

Acceptance gate G1:

- graph and prediction invariance tests pass at numerical tolerance;
- exact-MIC tests pass against an independent reference on orthogonal,
  hexagonal and adversarial oblique cells;
- the pilot completes without numerical instability or catastrophic error;
- the implementation and pilot are reviewed before any full queue is opened.

Stop rule: if formal repair/retraining is not authorized, Route A stops and the
paper returns to Route B. A favorable diagnostic alone is not a PRB correctness
waiver.

Milestone: `J-R1-geometry-correctness`.

## G2 — Repaired core evidence

The repaired graph changes the scientific model; old DART predictions cannot
be relabeled as repaired-model results.

Recommended lean PRB evidence package:

- freeze one repaired DART recipe before test evaluation;
- rerun the random, pair-held-out, host-held-out and impurity-held-out OOF core
  protocols with their existing immutable splits and declared seeds;
- reuse unchanged descriptor and SchNet evidence only after an identity audit;
- regenerate pair-OOF class errors and retrospective ranking from the repaired
  predictions;
- demote the complete factorial, internal UQ and SchNet readout sensitivity to
  historical/supporting evidence unless they are rerun under the repaired
  implementation.

If the manuscript retains “validation-selected g111” or causal G/E/P factorial
claims, the complete 40-run factorial must also be rerun. If it retains DART
UQ, all five repaired ensemble members and their calibration analysis must be
rerun. The lean route avoids those claims and gives the physical analysis the
main-text space.

Acceptance gate G2:

- every run originates from a pushed clean commit on the GPU server with GPU 2
  excluded and an immutable config/data/split manifest;
- sample coverage is exact and every paper-facing number is regenerated from
  predictions, not copied from old tables;
- the pair-versus-new-constituent hierarchy is either reproduced with
  clustered intervals or withdrawn;
- new pair-OOF predictions cover all 10,224 canonical rows before G3.

Milestone: `J-R2-repaired-core-oof`.

## G3 — Canonical materials-physics analysis

All analyses use repaired-model OOF predictions under Route A. Under Route B
they may be run on the frozen current predictions only as exploratory evidence
and must not erase the graph-correctness caveat.

### P0: provenance and coverage

Join, by audited `sample_index` and raw row `id`:

- protocol-v2 sample table and structures;
- pair-, host- and impurity-OOF predictions;
- raw `depth`, `extension_factor`, `conv2`, `en1` and `en2` fields;
- a frozen 44-host taxonomy and 65-impurity taxonomy.

Any main variable below 90% finite coverage is excluded from the paper; 90-95%
coverage is SM-only. `en1-en2` is not a displacement proxy.

### P1: local geometry and chemistry versus pair-OOF error

Pre-register before viewing outcomes:

- E-module-aligned `CN_5A`, mean/max neighbor distance and local
  electronegativity contrast;
- smooth coordination, local distance dispersion and a clearly named
  post-relaxation clearance/free-radius proxy;
- signed and absolute covalent-radius, electronegativity and valence mismatch;
- `depth`, distance from its class boundary and `abs(log(XF))`, where XF is the
  database-defined out-of-plane expansion/reconstruction proxy.

Primary outcome is absolute OOF error; signed residual is a bias diagnostic.
Fit a robust additive/mixed model to `log(abs_error + 0.05)` with crossed host
and impurity effects and fold adjustment. Stratify by adsorbate/interstitial,
test feature-by-class interactions, and control for atom count, cell/supercell,
training support and target magnitude in sensitivity analyses.

Test the geometry and chemistry blocks jointly before single-feature effects;
apply BH-FDR to the latter. Bootstrap by the 1,730 host-impurity pairs and
report fold-wise stability.

Suggested main-text evidence gate for a physical association:

- adjusted 10th-to-90th percentile error contrast at least 0.10 eV or 20%;
- 95% clustered interval excludes zero;
- direction agrees in at least four of five pair folds;
- direction survives exclusion of `XF>2` pathological reconstructions.

Otherwise report a null/unstable association and do not use “explains” or
causal language.

### P2: chemical families

Freeze one exhaustive host map before analysis:

- chalcogenide: 27 hosts;
- carbide/MXene-like: 5;
- elemental/hydrogenated: 7;
- halide/halochalcogenide: 4;
- oxide: 1;
- no retained nitride host exists.

Freeze impurity series as 3d (10), 4d (10), 5d (9), main group (35), and
f-block/other (Lu). Report sample-weighted and identity-macro MAE. Make an
inferential comparison only when a group contains at least five identities and
100 samples; otherwise describe it in the SM.

### P3: test, do not assume, the host-versus-impurity explanation

Because both OOF systems cover the same 10,224 rows, compare sample-matched
absolute errors. Estimate the paired host-minus-impurity error gap with
host/impurity-clustered uncertainty before asking why it exists.

Within each fold, fit all scaling/PCA/kNN operations only on the training
partition and construct:

- host structural novelty from composition, area/cell, slab thickness and
  local RDF/coordination;
- impurity chemical novelty from radius, electronegativity, valence,
  period/group.

Only write “consistent with structural-motif shift” when the host novelty is
larger, novelty-gap and error-gap are associated with a nonzero interval, and
the direction is stable in at least four folds. Otherwise retain “the cause is
not isolated.”

### P4: deterministic structural cases

- choose four high-error interstitials by a frozen ranking rule with no
  repeated host or impurity;
- match four low-error adsorbates by host family, target-energy decile, atom
  count and impurity series;
- keep `XF>2` or high-`conv2` pathology cases in a separate panel;
- archive every selected ID and the full eligible pool before rendering.

Cases are illustrative, not proof. They enter the main text only if they agree
with a population-level P1 result and cover at least three host families.

### P5: physics outputs and figure plan

Canonical outputs:

- an exact joined descriptor/residual table;
- a preregistration JSON and analysis manifest;
- effect estimates, clustered intervals, FDR ledger and five-fold stability;
- family tables and a deterministic case-selection ledger;
- fail-closed figure sidecars and a numeric audit.

Proposed main figure replacing the current factorial main figure:

1. adjusted error versus coordination/clearance by incorporation class;
2. radius/electronegativity/valence mismatch effects;
3. error versus `extension_factor`, with `XF=2` marked;
4. host-family and impurity-series macro-MAE forest plot.

If P3 passes, add a compact novelty-gap panel. Move full factorial, UQ, all 44
hosts/65 impurities, cutoff sensitivities, null results and the complete case
atlas to the Supplemental Material.

Acceptance gate G3:

- P1 or P3 yields a stable, interpretable association, and P2 supplies an
  auditable chemical context; or the paper explicitly reports that mechanism
  was not isolated;
- no association is called causal;
- every main-text effect passes the canonical numeric and figure audit.

Stop rule: when both geometry and chemistry blocks are unstable and P3 fails,
stop further slicing and reassess PRB rather than searching for a favorable
subgroup.

All P0-P5 jobs are formal computations and therefore run on the GPU server,
although they are CPU statistical workloads and need not allocate a CUDA
device.

Milestone: `J-R3-physics-error-map`.

## G4 — Optional GPU-only low-cost-geometry evidence

This is high value only after G1 and must be labeled according to the actual
geometry provenance.

### G4A: provenance gate

Preferred source order:

1. genuine saved early ionic snapshots;
2. versioned initial structures reproduced from the original IMP2D
   DefectBuilder workflow and source host structures;
3. structures relaxed by one predeclared universal ML potential from those
   authentic/reproduced initial structures.

The current database contains none of the first item. If the second cannot be
reconstructed without using the final geometry, a perturbation around the
final DFT basin may be run only as a sensitivity study, not as deployment
evidence.

### G4B: bounded pilot and formal test

- freeze one universal potential before viewing DART outcomes; audit its
  element coverage and 2D-PBC behavior;
- stratify approximately 128 host-impurity pairs by host, class and atom count
  and retain all candidates (roughly 256-512 structures);
- first run a 64-structure pilot;
- evaluate the unrelaxed/cheap-relaxed/final-DFT geometries with the matching
  pair-fold DART and the same final IMP2D target;
- retain every relaxation failure, basin change and coordination change;
- compare DART with the low-cost potential's own energy/ranking output.

Suggested pass conditions:

- relaxation success at least 90%;
- clustered 95% upper bound on DART MAE degradation below 0.10-0.15 eV;
- incorporation-class accuracy loss at most 5 percentage points;
- regret increase at most 0.05 eV;
- neither class incurs more than 0.15 eV extra MAE;
- DART adds value over direct low-cost-potential ranking.

Budget estimate: 1-3 GPU-hours for the pilot and 5-20 GPU-hours for the formal
set, subject to a server-side measured budget freeze.

Even after a pass, the strongest permissible statement for a near-basin proxy
is “robust to one low-cost relaxation proxy initialized near the DFT basin.”
It does not establish general prospective acceleration.

Milestone: `J-R4-cheap-geometry-shift`.

## G5 — Optional target-matched early-DFT campaign

This gate is deferred because the team does not currently have a target-
matched DFT trajectory workflow. Existing QE single points cannot be reused.

If later authorized, first reproduce the IMP2D formation-energy convention and
relaxation setup on anchors, then freeze genuinely new candidates before
labels. Save step 0, 1, 2, 5, 10, 25 and final geometries, energies, forces and
wall time. Compare DART at each step with the contemporaneous DFT ionic-step
energy; otherwise a cost-saving conclusion is impossible.

Run six full-relaxation pilots before any 48-candidate campaign. A deployment
claim would require reliable ranking by no more than 25% of full-relaxation
time, at least 50% median measured saving, no more than 0.10 eV MAE degradation,
no more than 5 percentage points class-accuracy loss, and an improvement over
the current-step DFT-energy comparator with a paired interval.

Indicative, not authorized, budget: 10-50 GPU-node-hours for the pilot and
100-500 GPU-node-hours for 48 candidates, to be replaced by measured pilot
costs.

Milestone: `J-R5-early-dft-snapshots`.

## Manuscript rebuild after the evidence gates

Recommended physics-first Results order:

1. chemical-transfer hierarchy;
2. physical correlates of error heterogeneity;
3. host-versus-impurity novelty and deterministic structural cases, if passed;
4. retrospective reranking over available relaxed IMP2D candidates;
5. low-cost-geometry robustness only if G4 passes.

Recommended seven-or-fewer main-figure structure:

- retain the DART architecture and the two manual physical schematics;
- move the protocol overview or full factorial to the SM;
- keep one transfer/applicability figure;
- add one four-panel physical error map;
- add one novelty/case figure only if its population-level gate passes;
- retain a compact retrospective-ranking figure, or replace it with G4 if the
  latter becomes the stronger conclusion;
- move internal UQ and complete family/case atlases to the SM.

The abstract and conclusion may reintroduce “acceleration,” “early stopping,”
or “prospective screening” only after G5, not after G3 and not merely after a
near-basin G4 sensitivity pass.

## Manual-control checkpoints

| User command | Authorized scope | Explicitly not authorized |
|---|---|---|
| `批准 G0` | Claim-boundary manuscript edits and prose audit | Inference, training, DFT, commit/push |
| `批准 G1A` | Server-side permutation/MIC diagnostic only | Model repair or retraining |
| `批准 G1B` | Invariant graph/MIC implementation, tests and one-fold pilot | Full queue |
| `批准 G2` | Frozen repaired core OOF queue and collection | G3/G4/G5 |
| `批准 G3` | Preregistered physics analysis and figures | Geometry relaxation or DFT |
| `批准 G4 pilot` | Sixty-four-structure low-cost-geometry pilot | Formal set or acceleration claim |
| `批准 G4 full` | Frozen 256-512 structure geometry-shift test | DFT |
| `批准 G5 pilot` | Six target-matched DFT relaxations with snapshots | Forty-eight-candidate campaign |

At each milestone: code and contracts are committed and pushed before formal
server execution; results are collected into a separate reviewed commit; no
later stage starts until the user accepts the preceding gate.
