# G2 execution runbook

Status date: 2026-08-10
Authorization: **G2 approved by the user on 2026-08-10.** Approval authorizes
the repaired full core OOF queue and its collection. It does not reorder the
correctness chain: the queue may not launch until every G1/G1C precondition
below is closed on the server, and canonical G3 promotion remains a separate
checkpoint.

The binding field-level specification for what G2 must produce is
`g3_g2_acceptance_contract.md`. This runbook is orchestration only and does
not redefine any contract field.

## Preconditions, in order

1. **Commit and push the current G1 implementation.** The repaired graph
   builder (`exact_mic_invariant_triplets_v1` in `src/graph.py`), the
   corrected E-module mode `zero_residual_v1` in `src/models/crystal_v2.py`,
   the `scripts/prm_g1_*.py` tooling, and `configs/prm/g1/` are currently
   uncommitted local work. Formal runs require a pushed clean commit.
2. **G1 server acceptance.** Run, from the pushed clean commit on the
   accepted server profile with GPU 2 excluded: the G1A legacy
   permutation/MIC diagnostic, the invariance and exact-MIC property tests,
   the repaired-dataset rebuild (10,641 rows in, 10,224 canonical, 417
   excluded), the one-fold pilot, and the acceptance collector. Required
   receipts: `prm_g1_repaired_dataset_v1` and `prm_g1_acceptance_v1`.
3. **G1C corrected pretraining lineage.** `prm_g1c_rebuild_jarvis.py` must
   reconstruct the hash-frozen 19,902-record source subset;
   `prm_g1c_pretrain.py` must rerun the frozen 30-epoch seed-42 recipe on the
   repaired graph and version the corrected checkpoint. Required receipt:
   `prm_g1c_pretraining_receipt_v1` with the corrected asset SHA-256.

## Implementation gap that must close before the queue

As of this date the repository contains **no producer** of
`prm_g2_run_manifest_v1`; the only reference is the consumer
`scripts/prm_g3_physics_analysis.py`. Before the queue launches:

- the training entry point must emit a complete `prm_g2_run_manifest_v1` per
  run, with every field the acceptance contract enumerates (split identity
  and counts, seed, `corrected_pretrain` arm, corrected asset and
  atom-feature-table SHA-256, `env_zero_neighbor_mode=zero_residual_v1` at
  top level and inside `config.model_kwargs`,
  `model_recipe_scope=config.model_kwargs`, recomputable recipe hash,
  model-source SHA-256, `config.batch_size=64`, graph-builder version,
  repaired-data hash, prediction output hash);
- an acceptance builder (suggested name `scripts/prm_g2_build_acceptance.py`)
  must assemble `prm_g2_acceptance_v1` from the run manifests, recomputing
  the recipe hash from each run's embedded `model_kwargs` rather than
  trusting prose, and binding the three G1/G1C receipts;
- both land in a pushed commit that descends from the G1 repair commit.

## Queue definition

- Protocols: pair-, host-, and impurity-held-out OOF are required by the
  acceptance contract; the random-repeat protocol is additionally rerun for
  the paper's core comparison, per the lean evidence package of the v1 plan.
- Seed law (hard): every pair fold exactly seed 242; every host and impurity
  fold exactly seeds 242, 243, and 244 once each. Extra, missing, or
  duplicate seeds invalidate acceptance. Minimum queue: 5 pair + 15 host +
  15 impurity runs, plus the random-protocol repeats.
- Every run: pushed clean commit, immutable split manifests, initialization
  arm `corrected_pretrain` with the G1C asset, inference batch size 64,
  GPU 2 excluded, exclusive-create outputs, durable logs.
- The complete factorial, internal UQ, and SchNet readout sensitivity are
  **not** rerun under this authorization; per the lean package they are
  demoted to historical/supporting evidence, and the manuscript drops
  validation-selection and factorial-effect claims accordingly.

## Collection and acceptance

1. From a separate clean collector worktree, verify the exact expected run
   census; no partial-queue number may enter any paper artifact.
2. Verify pair-OOF coverage of all 10,224 canonical rows (G3 precondition).
3. Build `prm_g2_acceptance_v1`; run the G3 CLI's fail-closed acceptance
   validation against it as a dry check.
4. Commit the acceptance and archived evidence in a reviewed commit whose
   ancestry satisfies the contract (descends from the G1 repair commit, is
   contained by a fetched remote branch, and is an ancestor of the eventual
   G3 run commit).
5. Then request `批准 canonical G3`.

Milestone: `J-R2-repaired-core-oof`.
