# G2 trust contract required by canonical G3

Canonical G3 is fail-closed until an accepted `prm_g2_acceptance_v1` JSON is
provided. This file documents the fields consumed by
`scripts/prm_g3_physics_analysis.py`; it does not claim that G2 exists or has
been authorized.

The acceptance must bind:

- source IMP2D hash and repaired graph-dataset hash;
- a hash-verified `prm_g1_repaired_dataset_v1` receipt with 10,641 rows,
  10,224 canonical rows, 417 excluded rows, preserved order/non-graph fields,
  and graph builder `exact_mic_invariant_triplets_v1`;
- a separate accepted `prm_g1_acceptance_v1` receipt that binds property
  tests, the bounded pilot, repaired-dataset receipt/output, and corrected
  pretraining lineage;
- the actual hash-verified `prm_g1c_pretraining_receipt_v1`, including its
  19,902-row corrected source dataset, graph-builder version, frozen 30-epoch
  seed-42 training contract, and corrected asset SHA-256;
- a 40-character G2 code commit descending from the G1 repair commit;
- one frozen model-recipe hash and one corrected-pretraining-lineage hash;
- initialization arm `corrected_pretrain`, the corrected asset SHA-256, and
  the hash-frozen `data/atom_features_ref.pth` SHA-256;
- pair, host, and impurity OOF sources for all five folds.
- exactly seed 242 for every pair fold and exactly seeds 242, 243, and 244
  once each for every host and impurity fold; extra, missing, or duplicate
  seeds are rejected;
- the G2 code commit must be contained by a fetched remote branch and be an
  ancestor of the current pushed G3 run commit.

Each OOF source entry contains `seed`, `split_sha256`, `prediction_path`,
`prediction_sha256`, `run_manifest_path`, and `run_manifest_sha256`. The run
manifest itself must be complete and clean and must reproduce the G2 code
commit, repaired-data hash, model-recipe hash, graph-builder version, and
pretraining-lineage hash. It must also identify `corrected_pretrain`, bind the
actual corrected asset SHA-256 and atom-feature table SHA-256, and bind its
prediction output hash. The manifest's `split_sha256` and `split_counts` must
match the actual commit-pinned repository split after bounds, uniqueness,
partition-disjointness, full 10,641-row coverage, 417 exclusions, and grouped
identity-disjointness checks. Its prediction is then rechecked against the
commit-pinned protocol split and target vector.

Minimal structural sketch (hash values intentionally omitted):

```json
{
  "schema_version": "prm_g2_acceptance_v1",
  "milestone": "J-R2-repaired-core-oof",
  "status": "accepted",
  "graph_status": "repaired_exact_mic_invariant_triplets",
  "data_sha256": "<source IMP2D SHA-256>",
  "repaired_dataset": {
    "source_data_sha256": "<source IMP2D SHA-256>",
    "graph_dataset_sha256": "<G1 repaired pickle SHA-256>",
    "path": "$PRM_REPAIRED_DATASET_PATH"
  },
  "g1": {
    "repaired_dataset_receipt": {"path": "<alias/path>", "sha256": "<SHA-256>"},
    "acceptance_receipt": {"path": "<alias/path>", "sha256": "<SHA-256>"},
    "corrected_pretraining_receipt": {"path": "<alias/path>", "sha256": "<SHA-256>"}
  },
  "code_commit": "<40 hex>",
  "model_recipe_sha256": "<SHA-256>",
  "pretrained_asset_lineage_sha256": "<SHA-256>",
  "initialization_arm": "corrected_pretrain",
  "pretrained_asset": {"path": "<alias/path>", "sha256": "<corrected asset SHA-256>"},
  "atom_feature_table_sha256": "5fa97d7788ef9b1e10be6d874aa9c6c7d55c532d055c187bb9452659983c5bb4",
  "prediction_sources": {
    "pair_cv": {"0": ["<source-entry>"], "1": [], "2": [], "3": [], "4": []},
    "host_cv": {"0": ["<source-entry>"], "1": [], "2": [], "3": [], "4": []},
    "dopant_cv": {"0": ["<source-entry>"], "1": [], "2": [], "3": [], "4": []}
  }
}
```

The illustrative empty arrays above are not valid acceptance data; every fold
must contain at least one source. G3 never infers acceptance from directory
names or status prose.

Every referenced run manifest has schema `prm_g2_run_manifest_v1` and contains
the exact `split_id`, `split_sha256`, `split_counts`, `seed`,
`initialization_arm`, `pretrained_asset_sha256`,
`pretrained_asset_lineage_sha256`, `atom_feature_table_sha256`,
`graph_builder_version`, `model_recipe_sha256`, repaired `data.data_sha256`,
and `outputs.prediction_sha256`. Canonical analysis reopens every archive and
then reconstructs the joined table from the raw database; an edited CSV plus
an edited external manifest cannot promote itself to canonical evidence.
