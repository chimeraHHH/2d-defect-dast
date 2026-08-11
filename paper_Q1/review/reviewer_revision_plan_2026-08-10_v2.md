# AutoSci J-R revision plan v2: physics-depth response for PRB

Status date: 2026-08-10
Supersedes: `reviewer_revision_plan_2026-08-10.md` (v1) as the controlling
revision plan. The v1 gate definitions G0, G1A/G1B, G2, G4 and G5 remain valid
and are referenced here rather than restated. v1 is retained as history.
Reviewer input: the full PRB physics-depth comment, atomized in
`reviewer_comment_matrix_2026-08-10.md` items P1--P9, plus two elements that
the matrix does not yet carry and that this plan adds as W-items and the EV
gate.
Execution state: G1 repair, G1C pretraining-lineage repair, and G3
implementation plus server-side exploratory analysis are authorized
(2026-08-10). **G2 and EV-A were approved by the user on 2026-08-10** (see the
updated checkpoint table); G2 remains sequenced behind G1/G1C server
acceptance, and EV-A behind its frozen preregistration. EV-B/C/D, canonical G3
promotion, release, and submission are not authorized. This plan authorizes
nothing by itself.

## 1. What the new comment changes relative to v1

The reviewer accepts three physical conclusions as the paper's genuine value:

- **C1** — missing-host and missing-impurity supervision is harder than new
  host--impurity combinations of seen constituents;
- **C2** — interstitial incorporation is harder to predict than adsorption;
- **C3** — defect formation energy is not suited to simple atomwise additive
  pooling.

The criticism is that the manuscript spends most of its space on ML protocol
and benchmarking and does not answer *why* any of C1--C3 holds. This changes
the revision target from "add a physics section" to "restructure the paper so
that each headline conclusion is paired with a mechanism-level analysis."

Two elements are new relative to the v1 plan:

1. The comment explicitly requests external validation on a small,
   independently generated, consistently computed DFT set. v1 treated
   target-matched DFT as a deferred cost-saving gate (G5). The reviewer's
   request is different and smaller: it is about *accuracy transfer*, not
   acceleration. This plan therefore defines a separate **EV gate** and
   recommends it, while keeping it under explicit manual authorization
   because it requires new DFT.
2. The comment asks for host-family decomposition "chalcogenide, carbide,
   nitride, ...". The retained 44-host set contains **no nitride host**. The
   response must state this fact and use the frozen five-family taxonomy; a
   nitride panel cannot and must not be invented.

## 2. Conclusion-to-mechanism map (the "why" program)

Each accepted conclusion gets exactly one preregistered why-program. These are
work packages W1--W3; they reuse matrix items P1--P8 where possible.

| ID | Conclusion | Why-program | Evidence base | Canonical gate |
|---|---|---|---|---|
| C3 | Additive pooling unsuitable | **W1 extensivity diagnostic** | Archived SchNet add/mean predictions; DART panel from repaired OOF | SchNet panels: one-time pipeline-independence audit. DART panel: G2 |
| C1 | Host holdout hardest | **W2 novelty-and-support decomposition** (extends P6/P3) | Sample-matched host- and impurity-OOF errors on the same 10,224 rows | G2 for canonical; legacy OOF exploratory now |
| C2 | Interstitial harder than adsorbate | **W3 local-environment attenuation test** (stratifies P1/P2/P8) | Pair-OOF residuals joined with frozen geometry/chemistry descriptors | G2 for canonical; legacy OOF exploratory now |

### W1 — extensivity diagnostic for the readout conclusion

Physical framing to be tested, not asserted: the impurity formation energy is
a localized, non-extensive quantity (a difference of extensive totals), so a
sum readout forces the prediction to scale with system size and a mean
readout dilutes the defect signal as 1/N, while a defect-centered readout
matches the locality of the target.

Preregistered analysis:

- regress the signed OOF residual of the additive-readout SchNet on total
  atom count and host-only atom count, with host-clustered bootstrap
  intervals; report the slope, not a correlation anecdote;
- repeat for the 15-run mean-readout host-CV sensitivity arm; the extensivity
  hypothesis predicts a near-zero size slope but an error contribution that
  grows with dilution; report both;
- repeat for DART; the locality hypothesis predicts a near-zero slope;
- report the three slopes side by side with the archived pooled MAEs
  (6.703 eV add, 2.552 eV mean, 0.938 eV DART on host-CV).

Canonicality condition: the SchNet panels use archived, hash-verified
predictions from an implementation that is separate from the DART graph
builder. Before calling any SchNet-only panel canonical, run one bounded audit
confirming that the SchNet input pipeline shares neither the 32-pair ordered
triplet cap nor the componentwise-wrapped radial distance. If the audit finds
shared code, W1 drops to exploratory until G2. The DART panel is
legacy-exploratory until G2 regardless.

Main-text gate: the add-readout size slope is nonzero with a clustered 95%
interval excluding zero and a sign consistent across at least four of five
host folds; otherwise the extensivity explanation is reported as not
established and C3 remains a purely empirical recipe result.

### W2 — why host holdout is hardest

The nominal pooled host-minus-impurity gap is 0.094 eV and has not been shown
significant. The why-question is only admissible after the gap itself is
established. Ordered protocol:

1. estimate the paired, sample-matched absolute-error gap on the shared
   10,224 rows with host- and impurity-clustered uncertainty;
2. if the interval includes zero, the manuscript reports "comparable
   difficulty; the cause is not isolated" and W2 stops;
3. if the gap is established, adjudicate **three** preregistered candidate
   explanations jointly, not just the two from v1:
   - host structural novelty (train-fitted composition, cell/area, slab
     thickness, RDF/coordination distances);
   - impurity chemical novelty (radius, electronegativity, valence,
     period/group distances);
   - **training-support asymmetry**: five host folds each remove roughly
     9 of 44 hosts while impurity folds remove roughly 13 of 65 impurities,
     so per-test-sample training support differs by construction; include
     same-constituent support counts as a covariate before crediting any
     novelty mechanism.

Write "consistent with structural-motif shift" only when host novelty
dominates after support adjustment, the novelty-gap association has a nonzero
clustered interval, and the direction is stable in at least four of five
folds.

### W3 — why interstitials are harder

The class gap is answered with an attenuation test rather than a new
correlation atlas:

1. fit the P1 robust model with an incorporation-class indicator and no
   geometry block; record the class coefficient;
2. add the frozen geometry/chemistry block (coordination number, mean/max
   neighbor distance, clearance/free-radius proxy, `depth`,
   `abs(log(XF))`, mismatch features);
3. report the attenuation of the class coefficient with clustered intervals.

Interpretation ladder, frozen in the preregistration: full attenuation means
the interstitial penalty is accounted for by measurable local crowding and
distortion; partial attenuation is reported as partial; no attenuation means
the penalty is not explained by the available descriptors. The 14
zero-neighbor rows (13 adsorbates, 1 interstitial) follow frozen amendment A1
of the G3 preregistration and are never silently dropped.

## 3. Feasibility verdict on the eight requested analyses

| Reviewer request | Verdict | Basis and boundary |
|---|---|---|
| Error vs coordination number, local void size, mean/max neighbor distance | Feasible | Canonical structures and the audited 5 A graph (P1). "Void size" requires the frozen post-relaxation clearance/free-radius proxy of P2; the name "void size" is not used for `d_max`. |
| Error vs radius mismatch, electronegativity difference, valence difference | Feasible | Fixed element tables and the permutation-invariant impurity identity (P3); signed and absolute variants, BH-FDR over single-feature tests. |
| Host-family decomposition incl. nitride | Feasible minus nitride | Frozen taxonomy: chalcogenide 27, carbide/MXene-like 5, elemental/hydrogenated 7, halide/halochalcogenide 4, oxide 1. **Zero nitride hosts are retained; the response states this explicitly** (P4). |
| Impurity-series decomposition 3d/4d/5d/main group | Feasible | 3d 10, 4d 10, 5d 9, main group 35, f-block/other 1 (Lu). Inference only for groups with at least 5 identities and 100 samples (P5). |
| Why host holdout is worse | Feasible as W2 | Gap must first be shown significant; three-way adjudication including training-support asymmetry (P6). |
| Concrete high-error interstitial / low-error adsorbate structures | Feasible | Deterministic frozen selection rule, matched controls, no repeated host/impurity, full eligible-pool ledger (P7). Illustrative only; main text only if population-level W3/P1 agrees. |
| Error vs relaxation amplitude / local distortion | Proxy only | Atomic displacements and trajectories are absent from the released database. Only the database-defined `extension_factor` supports an out-of-plane expansion/reconstruction proxy: `abs(log(XF))` plus the `XF>2` exclusion sensitivity. No atomic-displacement claim is permitted, and `en1-en2` is not a displacement (P8). |
| External validation on a small, independently generated, consistently computed DFT set | Requires new DFT | The 37 historical QE points are rejected (inconsistent isolated-atom reference, post-hoc offset, selected subset, obsolete model). Defined below as the EV gate; recommended, but blocked on explicit authorization (P9). |

## 4. EV gate — external DFT validation (new; responds to the explicit request)

Purpose: test whether the repaired model's formation-energy accuracy transfers
to structures generated and computed *outside* the IMP2D pipeline under one
consistent convention. This gate supports a transferability claim only; it
does not support any acceleration or cost-saving claim (those remain G5).

Design, to be frozen in its own preregistration before any label exists:

1. **Convention anchors (EV-A).** Reproduce the IMP2D formation-energy
   convention (reference states, chemical-potential terms, relaxation
   settings, k-mesh, vacuum, functional) on 6 anchor structures drawn from
   the retained set. Acceptance: anchor formation energies agree with the
   database values within a frozen tolerance decided before the runs;
   disagreement stops EV and is reported as a convention mismatch, not
   hidden.
2. **Candidate freeze (EV-B).** Generate 40--60 new defect structures that
   are absent from IMP2D, stratified in advance across: at least three host
   families, at least three impurity series, both incorporation classes, and
   an intentional out-of-support slice (unseen host--impurity pairs; if
   possible, one to two unseen hosts or impurities). Candidate IDs, initial
   geometries, and the stratification ledger are committed before any DFT
   output exists. Model predictions on the frozen candidates are also
   committed before labels (blind protocol).
3. **Consistent computation (EV-C).** One DFT code, one settings file, all
   inputs/outputs/wall times archived; failed relaxations retained and
   reported, never resampled silently.
4. **Evaluation (EV-D).** Compare blind predictions against the new labels:
   MAE, bias, Spearman, class-resolved errors, and coverage of the calibrated
   intervals from the repaired UQ arm if available. Report per-stratum
   results with the small-sample caveat; no post-hoc candidate exclusion.

Ordering constraint: EV predictions must come from the repaired model, so EV
runs after G2. EV-A convention work may be authorized earlier since it
touches only retained anchor structures.

Indicative budget (to be replaced by measured pilot costs): 6 anchor
relaxations first; then 40--60 defect supercell relaxations. This is the
single most expensive item in the physics-depth response and is why it stays
a separately authorized gate.

Stop rule: if EV is not authorized, the manuscript states plainly that all
evidence is database-internal and that external validation is future work.
The paper does not simulate external validity with historical QE numbers.

## 5. Gated execution order

Nothing below reorders the correctness chain: analyses of current OOF
predictions are exploratory (legacy graph) and are prohibited from canonical
manuscript claims until G2 exists.

| Stage | Content | Authorization state |
|---|---|---|
| S0 (now) | Exploratory G3 run on legacy OOF per the frozen preregistration; implement W1--W3 as preregistered extensions of the G3 CLI; one-time SchNet pipeline-independence audit; claim-bounded prose and figure restructuring | Already authorized (G1/G3 scope + audit is diagnostic) |
| S1 | Complete G1 acceptance (invariant triplets, exact MIC, property tests, one-fold pilot) and G1C corrected pretraining lineage (19,902-row rebuild, versioned checkpoint, receipts) | Authorized, pending server execution |
| S2 | G2 repaired core OOF: pair/host/impurity/random protocols on immutable splits, seeds per the acceptance contract; produce `prm_g2_acceptance_v1` | **Approved 2026-08-10**; launches after S1 per `g2_execution_runbook.md` |
| S3 | Canonical G3 = P0--P5 plus W1--W3 on repaired OOF; regenerate every physics number and figure sidecar | Blocked by S2; promotion needs `批准 canonical G3` |
| S4 | EV-A anchors, then EV-B/C/D blind external validation | **EV-A approved 2026-08-10** under `ev_a_preregistration.md`; EV-B/C/D requires separate approval; EV-D after S2 |
| S5 | Physics-first manuscript rebuild, numeric audit, release, submission decision | Blocked by S3 (and S4 if EV is run) |

W1--W3 additions to the G3 pipeline must be appended to the frozen G3
preregistration as amendments **before** any of their outcomes are viewed on
legacy predictions, in the same style as amendment A1. Outcomes already
inspected under the exploratory tier can never be re-frozen as confirmatory.

## 6. Manuscript restructuring (replaces v1's rebuild section)

Target shape: the ML protocol becomes the supporting skeleton; C1--C3 with
their mechanisms become the paper.

Results order:

1. chemical-transfer hierarchy (C1) with the W2 decomposition;
2. incorporation-geometry asymmetry (C2) with the W3 attenuation result and
   the deterministic structure cases (P7) if the population gate passes;
3. locality/extensivity of the formation-energy readout (C3) with W1,
   presenting the DART--SchNet comparison as a complete-recipe contrast and
   the readout sensitivity as its diagnostic;
4. physical error map (P1/P2 associations, host-family and impurity-series
   forest plots);
5. retrospective reranking over retained IMP2D candidates, in the truthful
   task boundary of G0;
6. external validation (EV) if and only if it ran.

Main figures (seven or fewer):

- keep the DART architecture schematic and the two manual physical
  schematics (with the two pending source-artwork corrections);
- one transfer/applicability figure (C1);
- one four-panel physical error map: error vs coordination/clearance by
  class; mismatch effects; error vs `extension_factor` with `XF=2` marked;
  family/series macro-MAE forest plot;
- one extensivity/readout figure (W1): residual-vs-size slopes for
  add/mean/DART plus the pooled MAE bars;
- one novelty/case figure only if W2 and P7 gates pass.

Move to Supplemental Material: the full factorial, internal UQ detail, the
complete 44-host/65-impurity atlases, cutoff and `XF` sensitivities, all null
results, and the complete case atlas.

Prose obligations fixed by this plan:

- state once, plainly, that the retained host set contains no nitride;
- state that atomic displacements are unavailable and that `extension_factor`
  is a database-defined out-of-plane proxy, so no atomic-relaxation-amplitude
  claim is made;
- keep every current number labeled legacy-graph evidence until G2 replaces
  it; the physics sections are written against repaired numbers only;
- abstract and conclusion are rebuilt around C1--C3 plus mechanisms, inside
  the G0 retrospective task boundary; no acceleration language.

## 7. Acceptance gates and stop rules

- **W-gates:** each W-item has its main-text evidence gate defined above;
  a failed gate produces a reported null ("not established" / "not
  isolated"), never silent omission and never causal wording.
- **No favorable-subgroup search:** when both the geometry and chemistry
  blocks are unstable and W2 fails, further slicing stops and the PRB route
  is reassessed (unchanged from v1).
- **Nitride and displacement honesty:** inventing a nitride panel or a
  displacement metric is a hard fail of the response audit.
- **EV blindness:** any label computed before the corresponding prediction
  is committed voids that candidate.
- **Submission decision:** PRB submission remains NO-GO until G1, G1C, G2,
  canonical G3 (with W1--W3), the response letter mapping every reviewer
  bullet to evidence or an honest limitation, and the v1 G0/administrative
  blockers are all closed. EV strengthens but is not a formal precondition;
  its absence must be disclosed as a limitation.

## 8. Manual-control checkpoints (updated)

| User command | Authorized scope | Explicitly not authorized |
|---|---|---|
| (already given) | G1 diagnostics/repair, G1C rebuild, G3 exploratory + W1--W3 preregistration amendments, SchNet pipeline audit, prose/figure restructuring | Everything below |
| `批准 G2` — **granted 2026-08-10** | Frozen repaired core OOF queue, collection, `prm_g2_acceptance_v1` | Canonical G3, EV, release |
| `批准 canonical G3` — **granted 2026-08-11** | P0--P5 + W1--W3 on repaired OOF, canonical figures | EV, release, submission |
| `批准 EV-A` — **granted 2026-08-10** | Six convention-anchor DFT runs | Candidate campaign |
| `批准 EV` | Frozen 40--60 candidate blind external validation | Any acceleration claim |
| `批准 release` | Archival release, persistent identifier | Journal upload |
| `批准 submission` | PRB upload with response letter | — |

At each stage: contracts and code are committed and pushed before formal
server execution; results land in a separate reviewed commit; no later stage
starts before the user accepts the preceding gate.
