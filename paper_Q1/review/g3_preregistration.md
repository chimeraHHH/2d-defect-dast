# G3 preregistration: materials-physics error map

Frozen: 2026-08-10

Milestone: `J-R3-physics-error-map`
State: **authorized for implementation and server analysis; canonical evidence blocked by G2**

## Evidence boundary

G3 asks which local geometric and chemical environments are associated with
out-of-fold formation-energy error. It does not test a causal mechanism and it
does not establish that the defect-enrichment module is responsible for an
association. The current DART predictions were produced with the graph
implementation now under G1 review. Any analysis of those predictions is
therefore hard-labeled:

> EXPLORATORY — LEGACY GRAPH; NOT FOR MANUSCRIPT

Canonical output is impossible unless the CLI receives an accepted
`prm_g2_acceptance_v1` manifest that enumerates repaired pair-, host-, and
impurity-OOF predictions and their hashes. This is enforced in code, not left
to author discretion. The acceptance must additionally bind the actual
`prm_g1c_pretraining_receipt_v1`, its corrected initialization asset, and the
`corrected_pretrain` arm used by every G2 run.

## Frozen inputs and execution

All extraction and statistics run on `WHUServer-L40S` from a clean pushed
commit with:

```bash
CUDA_VISIBLE_DEVICES='' "$PRM_PYTHON" \
  scripts/prm_g3_physics_analysis.py all \
  --evidence-tier exploratory \
  --model-status legacy_order_sensitive_graph \
  --data-path "$PRM_DATASET_PATH" \
  --raw-db "$PRM_RAW_DB_PATH" \
  --prediction-root artifacts/prm_results/comparison/runs/selected/g111/transfer \
  --output-dir "$PRM_RUN_ROOT/legacy-exploratory-<COMMIT>" \
  --bootstrap-draws 2000 --bootstrap-seed 20260810
```

The cleaned pickle must hash to
`1d59cc818d81252d49da525c6d77e2da8549ceb604d54af953d1caa4fb974a9b`.
The raw database must hash to
`3a71db999b477112da248dcf762c4384e455689953679d58b3d71a91e7148fc4`.
The model atom-feature table `data/atom_features_ref.pth` must hash to
`5fa97d7788ef9b1e10be6d874aa9c6c7d55c532d055c187bb9452659983c5bb4`.
The uncommitted run manifest may record supplied and resolved server paths;
committed manifests retain aliases and hashes only. G3 is a CPU statistical
job; the empty CUDA mask prevents it from reserving a GPU.
The count and seed above are hard requirements, not defaults that may be
changed. Analysis outputs are write-once; a partial or repeated run uses a new
versioned directory rather than overwriting prior evidence.

## P0: join and coverage

The exact join is `sample_index -> protocol row -> raw row id`. The canonical
population is all 10,224 protocol-v2 retained rows. Pair predictions use the
single frozen OOF member at seed 242. Sample-matched host and impurity OOF
errors use the mean of seeds 242, 243, and 244. Every prediction archive is
checked against its immutable test split and target vector.

Raw `depth`, `extension_factor`, `conv2`, `en1`, and `en2` values are copied
without reinterpretation. In particular, `en1` is a first-workflow-stage total
energy, not an ionic snapshot or displacement. From the IMP2D depth definition,
the signed class margin is `abs(depth)-1` and distance to the class boundary is
`abs(abs(depth)-1)`. `abs(log(XF))` is an out-of-plane
expansion/reconstruction proxy; `XF>2` is the prespecified pathology threshold.

Variables below 90% finite coverage are excluded. Variables from 90% to below
95% coverage are Supplemental-only. Main-text eligibility starts at 95%.
For a canonical `analyze` command, the immutable protocol, all accepted OOF
archives, the raw database, and every descriptor are loaded again and the
complete joined table is reconstructed and compared field by field. A CSV and
its external manifest cannot jointly self-certify canonical evidence.

## P1: local geometry and chemistry

The four E-aligned quantities are read from the actual graph sample consumed
by the corresponding prediction: the frozen legacy cleaned pickle for the
exploratory tier and the receipt-bound repaired G1 pickle for the canonical
tier. They use every directed impurity-centred graph entry, including nonzero
periodic impurity self-images retained when `i == j`; only the zero-displacement
self interaction is absent. Coordination, mean distance, and maximum distance
use that exact edge multiset. The electronegativity
contrast uses column 2 of the hash-frozen `atom_features_ref.pth`, the same
encoded scale consumed by `DefectEnvironmentEnrichment`, rather than claiming
raw Pauling units. The extraction archives the count of periodic impurity
self-images and independently recomputes the raw-database neighbor multiset.
Raw-minus-graph edge count, topology agreement, and maximum distance delta
when the topology agrees are audit columns; raw edges are never substituted
for canonical graph-native E quantities.

The independent physical geometry and chemistry descriptors instead use the
explicit host-only subset of those directed entries. This block adds
cosine-weighted smooth coordination, distance dispersion, and the minimum
signed covalent-sphere clearance around the impurity. The latter is
deliberately called a **post-relaxation minimum clearance**, not a pristine
void radius. Smooth coordination is `sum[0.5(cos(pi d/5)+1)]` and clearance is
`min[d-r_cov(impurity)-r_cov(neighbor)]` over host-only entries. Covalent radii
come from the run-time ASE table; raw Pauling electronegativity and group
tables come from `src/features.py` at the pinned commit. For physical
mismatches, zero-placeholder raw electronegativities and nonfinite radii are
missing, never imputed. Valence follows group for groups 1--12 and `group-10`
for groups 13--18.

The chemistry block contains signed and absolute local mismatches in covalent
radius, electronegativity, and valence count. The outcome is
`log(|pair-OOF residual| + 0.05 eV)`. The primary estimator is a Huber
M-estimator with weighted alternating projection of crossed host, impurity,
and pair-fold fixed effects repeated inside every IRLS iteration. It is a robust fixed-effect model, not a mixed-effects
model. Incorporation class and all feature-by-class interactions are fixed
before outcomes are viewed.

Geometry and chemistry blocks receive joint cluster-robust Wald tests. Each
feature is summarized by a 10th-to-90th percentile adjusted contrast in both
classes. Uncertainty comes from 2,000 host-impurity-pair bootstrap draws;
individual contrasts use Benjamini-Hochberg correction. Five fold-specific
refits and an `XF<=2` sensitivity are mandatory.

A paper-facing association requires a 0.10 eV contrast or 20% contrast on the
back-transformed predicted absolute-error scale. For each class, non-focal
continuous terms are fixed at class medians; the focal term is changed from
its class q10 to q90; host/impurity/fold effects are marginalized under the
final Huber-IRLS weighted zero-sum constraint;
`exp(eta)-0.05` is clipped at zero. A 95% clustered interval must exclude zero, the
same direction in at least four folds, and the same direction after excluding
`XF>2` and after adding the frozen nuisance controls plus space-group and
supercell fixed effects. Otherwise the result is reported as null or unstable.

## P2: chemical families

The exhaustive maps are in `g3_preregistration.json`: 27 chalcogenides, five
carbide/MXene-like hosts, seven elemental/hydrogenated hosts, four
halide/halochalcogenides, and one oxide. There is no nitride panel. Impurities
are frozen as ten 3d, ten 4d, nine 5d, 35 main-group, and Lu as f-block/other.
Both sample-weighted and identity-macro MAE are reported. Inferential language
requires at least five identities and 100 rows.

## P3: host-versus-impurity transfer

The first test is the sample-matched absolute-error difference between host
OOF and impurity OOF predictions. Structural novelty is fit only on each host
fold's training rows using host-only composition, area per host atom, slab
thickness, RDF, and coordination descriptors. Chemical novelty is fit only on
each impurity fold's training rows using radius, electronegativity, valence,
period, and group. Within each fold, descriptors are first averaged over rows
of each unique identity. Missing novelty descriptors alone are imputed with
the training-identity median and accompanied by an explicit missingness
indicator; this P3-only rule does not alter P1's no-imputation rule. Augmented
columns with zero training IQR are removed, retained columns use training
median/IQR scaling, and PCA plus nearest-neighbor references never see their
corresponding test partition.

After fold-local distances are complete, host and impurity novelty are each
median/IQR calibrated over unique OOF identities without labels. This declared
post-OOF operation only makes the distance scales comparable and never refits
descriptors, PCA, or neighbors. A Huber regression relates the raw-eV
host-minus-impurity absolute-error gap to the calibrated novelty gap plus
incorporation class, with 2,000 host-impurity-pair bootstrap draws.
“Consistent with structural-motif shift” is permitted only when the lower 95%
bootstrap bounds for both the mean host-minus-impurity error gap and the
novelty-gap slope are positive, and each is positive in at least four of five
strata assigned by the frozen pair split. Negative or mixed direction is
non-support or contradiction, never support. Host and impurity fold numbers
are never assumed to align. Host RDF uses
12 bins over 0--6 Å counted as directed host neighbor-list entries per host
atom; PCA retains the minimum training-only dimension
for 95% variance, capped at 10; novelty is mean Euclidean distance to five
training **identity** neighbors (unique hosts or unique impurity elements,
never repeated structure rows). Otherwise the
cause remains unisolated.

## P4 and figure data

Four high-error non-pathological interstitials are selected greedily without
repeated hosts or impurities. Four bottom-quartile adsorbates are matched by
host family, target decile from the full canonical target distribution,
impurity series, and atom count. Ties and rank keys end in `sample_index`.
Non-pathological requires finite `XF<=2`, finite `|conv2|`, and exclusion from
the exactly 103 finite rows in the full-population top 1% of `|conv2|`, ranked
by decreasing value then `sample_index`; controls pass the same rule. There is
no relaxed matching fallback: fewer than four pairs fails the case panel
rather than inviting manual substitution. Pathological `XF>2`, the 103-row
`|conv2|` set, and missing pathology provenance are ranked separately. The
full 10,224-row pool and every rank key are archived before rendering.

The pipeline writes data for a four-panel physical error map: adjusted
coordination/clearance trends, chemistry mismatch contrasts, extension-factor
behavior with `XF=2`, and host/impurity family errors. Panel a candidates are
`cn_5A` and post-relaxation minimum clearance. For each candidate that passes
the frozen class-stratified coverage gate, the analysis itself writes a
41-point class-specific q10--q90 adjusted profile from the same primary fit,
with pointwise (not simultaneous-band) 95% intervals from the same 2,000 pair
bootstrap draws. At least one candidate must pass. Panels a and c also receive
row-level `sample_observation` records containing class, absolute error, and
the plotted coordinate. Panel c carries a literal `reference_x=2` marker and
keeps raw XF observations; `abs(log XF)` is not inverted to a fictitious
single raw-XF profile. The pipeline does not itself admit a panel to the
manuscript. Legacy output carries an explicit watermark and must remain outside
`paper_Q1/sections` and paper-facing figure directories.
