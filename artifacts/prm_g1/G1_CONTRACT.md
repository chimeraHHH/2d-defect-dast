# G1 geometry-correctness contract

Status: **implementation hardened; formal GPU acceptance pending**. G2 full
OOF retraining is explicitly out of scope. A favorable G1A diagnostic is not
a correctness waiver, and no scientific GO exists until the final collector
finishes on the server and its receipt survives independent review.

## Frozen inputs

| Alias | SHA-256 | Role |
|---|---|---|
| `PRM_DATASET_PATH` | `1d59cc818d81252d49da525c6d77e2da8549ceb604d54af953d1caa4fb974a9b` | frozen 10,641-row IMP2D container |
| `PRM_RAW_DB_PATH` | `3a71db999b477112da248dcf762c4384e455689953679d58b3d71a91e7148fc4` | authoritative row identity and PBC |
| `PRM_CT_UAE_PATH` | `ac77b2720b7bb8a3b290d6bfbae482857962d55daa7fc3486f2c7f44c41c60dc` | elemental initialization |
| `PRM_LEGACY_PRETRAINED_PATH` | `5dd085b1393acee4db83422241102595a5d011c18a880a44c09947b29274a3bc` | legacy JARVIS initialization; diagnostic only |
| `PRM_JARVIS_DATA_PATH` | `41284603e6eeb6920fe132975570e657e69601a414257bb48d2196d7389369fa` | frozen 19,902-row source-task container |

Every formal command runs from a clean pushed commit on the
`WHUServer-L40S` profile with `PRM_PYTHON`. GPU 2 is excluded. Public receipts
contain aliases, hashes, commit, and GPU UUID only; personal paths and SSH
coordinates are not versioned.

## G1A — legacy sensitivity diagnostic

The old checkpoint is evaluated through an embedded frozen legacy builder:
first 32 ordered neighbour pairs per centre and component-wise fractional
wrapping. Sixteen deterministic transformations comprise eight seeded random
permutations and eight adversarial orderings. Separately, the dense radial
matrix alone is replaced by exact MIC while the old local graph and checkpoint
remain fixed.

Report over all five pair folds and exactly 10,224 canonical rows:

- per-sample prediction range over archived baseline plus 16 permutations;
- 95th/99th percentile and maximum range;
- maximum absolute pooled-MAE change;
- incorporation-class, global-site, and within-class site flips;
- absolute regret changes;
- old-to-exact-MIC prediction/MAE/decision changes;
- distance differences stratified by cell obliquity, legacy-cap activation,
  and incorporation class.

The diagnostic stability gate is: 99th-percentile range below 0.01 eV,
maximum range below 0.05 eV, MAE change below 0.01 eV, and zero class/site
flips. The archived checkpoint round trip must reproduce predictions within
`5e-5` eV or the command fails. That bound is set by measured GPU
floating-point nondeterminism, not pipeline fidelity: repeated inference of
the identical stored graphs on the assigned L40S differs by up to ~3e-6 eV
with single-sample spikes to ~1.1e-5 eV across the 51,120 canonical
inferences (measured 2026-08-10, torch 2.11.0+cu126); the gate sits five
times above the observed spike and two hundred times below the smallest
0.01 eV decision gate.

The archived dataset was built with a pre-3.28 ASE whose neighbour-list bin
traversal enumerates the identical edge set in a different arbitrary order
than the pinned ASE 3.28.0. Because the legacy first-32 triplet rule is
enumeration-order dependent, the identity arm imposes the archived
`(i, j, shift)` edge sequence — a fail-closed bijection on unique edge-image
keys — so the archived model inputs are reproduced bit-for-bit; permuted
variants keep the natural current-ASE enumeration, which is one more
arbitrary ordering of the same edge set. The diagnostic summary records how
many samples' natural enumeration differs from the archive. Archived float32
noise is compared with 1e-5 tolerances, and angles are compared in cosine
space because `acos` amplifies sub-1e-6 noise without bound near collinear
triplets.

G1A also freezes the historical E-module zero-neighbour behaviour as
`legacy_batch_dependent_v0`. This compatibility mode is diagnostic-only and is
set explicitly when the archived g111 checkpoints are loaded; it adds no
parameters and therefore preserves the legacy state-dict contract. Because
that historical rule is itself batch-composition dependent, the diagnostic
also fixes inference batch size at 64 and records it in the summary.

## Corrected E-module zero-neighbour semantics

At least one IMP2D impurity can have no outgoing defect-centred edge within the
formal 5 Å cutoff. Historically, an isolated impurity was unchanged when its
whole batch had no defect edge, but received an artificial
`|EN_impurity - 0|` feature (and projection bias) when another sample in the
same batch did have such an edge. This made predictions depend on batch
composition.

The default and all correctness-clean G1 pilots now freeze
`env_zero_neighbor_mode: zero_residual_v1`. The E-module computes whether each
defect atom has at least one outgoing local edge and masks the residual *after*
the projection, so all four environment features and the projection bias are
a strict no-op for a zero-neighbour impurity. The 11-feature experimental E
module obeys the same rule. The property suite checks mixed-batch versus
single-sample equality and separately checks the frozen legacy formula without
increasing the ten-test acceptance inventory.

## G1B — repaired IMP2D graph and bounded pilot

Graph-builder version: `exact_mic_invariant_triplets_v1`.

For a centre of degree `d`, construct the complete ordered-angle multiset of
size `n=d(d-1)` by duplicating every unordered angle. If `n<=32`, retain all
angles. If `n>32`, sort the complete multiset and retain ranks

`r_q = floor((2q+1)n/(2*32)), q=0,...,31`.

The ranks are unique and sample midpoint strata, avoiding endpoint bias.
Selection never uses atom index. Equal-angle ties feed identical operator
inputs. The current local block consumes only centre plus angle;
`triplet_index` first/third identities are provenance fields and do not enter
the message. Peak temporary complexity is `O(d^2)` for one centre and stored
complexity remains `O(32)` per centre.

Dense distances use `ase.geometry.find_mic` with explicit PBC recovered from
the hash-verified raw row after ID/numbers/positions/cell identity checks
(`atol=2e-5`, `rtol=0`). The rebuild preserves all 10,641 source rows, order,
targets, metadata, non-graph fields, and the 10,224/417 canonical/excluded
index map. It writes a new versioned pickle and receipt; overwriting the frozen
pickle is forbidden. Formal cutoff is exactly 5.0 Å.

Immediate one-fold pilot arms on the immutable pair-fold-0 membership are:

1. `legacy_init`: repaired graph plus old JARVIS asset. This is only a
   legacy-initialization compatibility sensitivity.
2. `no_pretrain`: repaired graph without the old source-task asset. This is
   correctness-clean for IMP2D but does not validate transfer initialization.

Both use a predeclared 30-epoch stability budget. Neither test metric is a
paper result or may select a later recipe. Required pilot evidence is finite
training/validation history, readable checkpoint, no numerical failure, and
best validation MAE below the preregistered catastrophic threshold of 3.0 eV.
All three generated pilot configurations, the freeze manifest, and each pilot
receipt bind `zero_residual_v1` plus the exact `src/models/crystal_v2.py` hash.

## G1C — corrected source-task lineage

The legacy source-task builder has three correctness defects: triplet centre
stored in the wrong column, order-dependent six-neighbour truncation, and a
nonperiodic dense `cdist`. Therefore the old pretrained asset cannot support
an end-to-end corrected claim.

G1C must preserve all 19,902 frozen JARVIS members/order/targets while
rebuilding only graph fields with correct centre semantics, the same bounded
invariant angular operator, and exact 3D MIC/PBC. It produces a new dataset,
new 30-epoch pretraining asset, and hash-bound receipts without overwriting the
legacy files. A third `corrected_pretrain` G1B pilot arm then checks asset/model
compatibility under the same fold and 30-epoch budget. End-to-end G1 GO is
blocked until this arm and its lineage receipts pass. G1C is not G2.

The corrected source-task recipe is fully frozen: baseline
`CrystalTransformer`, seed 42, Python-random 80/10/10 split, batch size 64,
30 epochs, MSE loss, AdamW (`lr=3e-4`, `weight_decay=1e-4`), plateau scheduler,
gradient clip 5.0, hidden size 128, three local and two global layers, four
heads, 5/12 Å local/global radii, no defect embedding, and dropout 0.1. Its
receipt records source validation/test metrics and asset compatibility hashes;
downstream usefulness is accepted only through the fixed third pilot arm.

## Copy-ready server sequence

```bash
test "$(hostname)" = "WHUServer-L40S"
test -n "$PRM_PYTHON"
test -n "$PRM_DATASET_PATH"
test -n "$PRM_RAW_DB_PATH"
test -n "$PRM_RUN_ROOT"
test -n "$PRM_G1_ROOT"
test -n "$PRM_REPAIRED_DATASET_PATH"
test -n "$PRM_JARVIS_DATA_PATH"
test -n "$PRM_CORRECTED_JARVIS_DATA_PATH"
test -n "$PRM_G1C_PRETRAIN_ROOT"
test -n "$PRM_G1_PILOT_ROOT"
test -n "$PRM_CT_UAE_PATH"
test -n "$PRM_LEGACY_PRETRAINED_PATH"
test -n "$PRM_G1_GPU"
test "$PRM_G1_GPU" != "GPU-33963073-e04e-1698-5d02-1a16d098c931"
test -z "$(git status --porcelain)"

"$PRM_PYTHON" scripts/prm_g1_rebuild_dataset.py \
  --source-data "$PRM_DATASET_PATH" \
  --raw-db "$PRM_RAW_DB_PATH" \
  --protocol-samples artifacts/prm_protocol_v2/samples.csv \
  --output-data "$PRM_REPAIRED_DATASET_PATH" \
  --output-receipt "$PRM_G1_ROOT/receipts/imp2d_rebuild.json"

CUDA_VISIBLE_DEVICES="$PRM_G1_GPU" "$PRM_PYTHON" scripts/prm_g1_diagnose_geometry.py \
  --data "$PRM_DATASET_PATH" \
  --raw-db "$PRM_RAW_DB_PATH" \
  --run-root "$PRM_RUN_ROOT" \
  --protocol-dir artifacts/prm_protocol_v2 \
  --ct-uae "$PRM_CT_UAE_PATH" \
  --out-dir "$PRM_G1_ROOT/g1a" --device cuda

"$PRM_PYTHON" scripts/prm_g1c_rebuild_jarvis.py \
  --source-data "$PRM_JARVIS_DATA_PATH" \
  --output-data "$PRM_CORRECTED_JARVIS_DATA_PATH" \
  --output-receipt "$PRM_G1_ROOT/receipts/jarvis_rebuild.json"

CUDA_VISIBLE_DEVICES="$PRM_G1_GPU" "$PRM_PYTHON" scripts/prm_g1c_pretrain.py \
  --corrected-data "$PRM_CORRECTED_JARVIS_DATA_PATH" \
  --dataset-receipt "$PRM_G1_ROOT/receipts/jarvis_rebuild.json" \
  --ct-uae "$PRM_CT_UAE_PATH" \
  --out-dir "$PRM_G1C_PRETRAIN_ROOT" --device cuda

"$PRM_PYTHON" scripts/prm_g1_freeze_pilots.py \
  --repaired-dataset-receipt "$PRM_G1_ROOT/receipts/imp2d_rebuild.json" \
  --corrected-pretraining-receipt "$PRM_G1C_PRETRAIN_ROOT/pretraining_receipt.json" \
  --split-output artifacts/prm_g1/splits/pair_cv5_f0_repaired_v1.json \
  --config-output-dir configs/prm/g1/generated \
  --freeze-manifest artifacts/prm_g1/pilot_freeze.json

# Commit and push the frozen split/configs before the property gate and pilots.
CUDA_VISIBLE_DEVICES="$PRM_G1_GPU" "$PRM_PYTHON" scripts/prm_g1_run_property_tests.py \
  --output-receipt "$PRM_G1_ROOT/receipts/property_tests.json"

CUDA_VISIBLE_DEVICES="$PRM_G1_GPU" "$PRM_PYTHON" scripts/prm_g1_run_pilot.py \
  --config configs/prm/g1/generated/legacy_init_pair_cv5_f0_seed242.yaml \
  --data "$PRM_REPAIRED_DATASET_PATH" \
  --dataset-receipt "$PRM_G1_ROOT/receipts/imp2d_rebuild.json" \
  --ct-uae "$PRM_CT_UAE_PATH" --pretrained "$PRM_LEGACY_PRETRAINED_PATH" \
  --result-root "$PRM_G1_PILOT_ROOT"

CUDA_VISIBLE_DEVICES="$PRM_G1_GPU" "$PRM_PYTHON" scripts/prm_g1_run_pilot.py \
  --config configs/prm/g1/generated/no_pretrain_pair_cv5_f0_seed242.yaml \
  --data "$PRM_REPAIRED_DATASET_PATH" \
  --dataset-receipt "$PRM_G1_ROOT/receipts/imp2d_rebuild.json" \
  --ct-uae "$PRM_CT_UAE_PATH" --result-root "$PRM_G1_PILOT_ROOT"

CUDA_VISIBLE_DEVICES="$PRM_G1_GPU" "$PRM_PYTHON" scripts/prm_g1_run_pilot.py \
  --config configs/prm/g1/generated/corrected_pretrain_pair_cv5_f0_seed242.yaml \
  --data "$PRM_REPAIRED_DATASET_PATH" \
  --dataset-receipt "$PRM_G1_ROOT/receipts/imp2d_rebuild.json" \
  --ct-uae "$PRM_CT_UAE_PATH" \
  --pretrained "$PRM_G1C_PRETRAIN_ROOT/pretrained_embed_corrected.pt" \
  --pretraining-receipt "$PRM_G1C_PRETRAIN_ROOT/pretraining_receipt.json" \
  --result-root "$PRM_G1_PILOT_ROOT"

CUDA_VISIBLE_DEVICES="$PRM_G1_GPU" "$PRM_PYTHON" scripts/prm_g1_collect_acceptance.py \
  --source-data "$PRM_DATASET_PATH" \
  --raw-db "$PRM_RAW_DB_PATH" \
  --protocol-samples artifacts/prm_protocol_v2/samples.csv \
  --repaired-data "$PRM_REPAIRED_DATASET_PATH" \
  --repaired-dataset-receipt "$PRM_G1_ROOT/receipts/imp2d_rebuild.json" \
  --property-tests-receipt "$PRM_G1_ROOT/receipts/property_tests.json" \
  --g1a-summary "$PRM_G1_ROOT/g1a/summary.json" \
  --g1a-predictions "$PRM_G1_ROOT/g1a/predictions.npz" \
  --g1a-sample-diagnostics "$PRM_G1_ROOT/g1a/sample_diagnostics.csv" \
  --legacy-run-root "$PRM_RUN_ROOT" \
  --jarvis-source-data "$PRM_JARVIS_DATA_PATH" \
  --g1c-dataset "$PRM_CORRECTED_JARVIS_DATA_PATH" \
  --g1c-dataset-receipt "$PRM_G1_ROOT/receipts/jarvis_rebuild.json" \
  --g1c-pretraining-receipt "$PRM_G1C_PRETRAIN_ROOT/pretraining_receipt.json" \
  --g1c-corrected-asset "$PRM_G1C_PRETRAIN_ROOT/pretrained_embed_corrected.pt" \
  --ct-uae "$PRM_CT_UAE_PATH" \
  --legacy-pretrained "$PRM_LEGACY_PRETRAINED_PATH" \
  --legacy-pilot-receipt "$PRM_G1_PILOT_ROOT/g1/pilots/legacy_init/pair_cv5_f0/seed242/g1_pilot_receipt.json" \
  --legacy-pilot-dir "$PRM_G1_PILOT_ROOT/g1/pilots/legacy_init/pair_cv5_f0/seed242" \
  --no-pretrain-pilot-receipt "$PRM_G1_PILOT_ROOT/g1/pilots/no_pretrain/pair_cv5_f0/seed242/g1_pilot_receipt.json" \
  --no-pretrain-pilot-dir "$PRM_G1_PILOT_ROOT/g1/pilots/no_pretrain/pair_cv5_f0/seed242" \
  --corrected-pilot-receipt "$PRM_G1_PILOT_ROOT/g1/pilots/corrected_pretrain/pair_cv5_f0/seed242/g1_pilot_receipt.json" \
  --corrected-pilot-dir "$PRM_G1_PILOT_ROOT/g1/pilots/corrected_pretrain/pair_cv5_f0/seed242" \
  --output artifacts/prm_g1/g1_acceptance.json
```

The final collector is intentionally a live revalidation, not a receipt
concatenator. It reopens both frozen source datasets, the raw IMP2D database,
the protocol table, all five G1A manifests/checkpoints/predictions, the
corrected source-task checkpoint/export/asset, and all three pilot run
directories. It reruns the current 15-test property/negative-contract suite,
rebuilds every graph with the current `build_graph`, strict-loads every model,
and re-infers every G1A baseline, identity, exact-MIC, and 16-permutation test
prediction, the corrected source-task validation/test metrics, plus all pilot
validation and test partitions. Budget 4--12 hours on the assigned GPU/host;
interruption produces no
acceptance because the output is write-once and created only after all gates.

If a fail-closed command leaves a partial output directory, archive that
directory and rerun to a new versioned path. Never reuse or overwrite it.

## Milestone gates

- `J-R1a-diagnostic`: tests, G1A summary, and repaired-dataset receipt are
  committed after independent hash verification.
- `J-R1b-pilot`: two-arm pilot receipts are committed; claims remain bounded.
- `J-R1c-source-lineage`: corrected JARVIS dataset/asset and third-arm receipt
  are committed. Only then can G1 be reviewed for GO.
- No G2 queue may open without explicit user authorization after G1 review.

Estimated bounded budget is 3--8 GPU-hours plus 1--3 server CPU-hours for the
two versioned graph rebuilds. No DFT calculation is involved.
