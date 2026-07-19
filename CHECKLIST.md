# PRM revision checklist

## Protocol

- [x] Create clean `prm-revision` branch from GitHub `main`.
- [x] Record dataset SHA256, size, schema and target/metadata summary.
- [x] Generate and validate immutable development, host, dopant and block splits.
- [x] Recompute targets from raw components and exclude four source sentinels.
- [x] Exclude coordinate- or invariant-distance-equivalent duplicate rows.
- [x] Freeze the canonical 10,572-structure modeling set and dedicated UQ split.
- [x] Add split-file support and sample-index preservation to the trainer.
- [x] Add run manifest and canonical prediction schema.
- [x] Pin ct-UAE and DFT-3D-lite initialization assets by SHA-256 and record
      the exact copied versus seeded parameter sets in every DART run.
- [x] Pass unit tests and one bounded remote smoke run.

## Experiments

- [ ] Run 2^3 architecture factorial for seeds 42-46.
- [x] Run descriptor baselines with validation-only tuning (27/27 formal splits;
      canonical descriptor bundle frozen from a clean worktree).
- [ ] Run SchNet comparator under the same split contract. (Implementation and
      configurations complete; runs await GPU capacity.)
- [ ] Select the final DART configuration from validation/CV evidence only.
- [ ] Run random out-of-fold and host--dopant-pair cross-validation.
- [ ] Run host-group cross-validation.
- [ ] Run dopant-group cross-validation.
- [ ] Run predefined host-by-dopant chemistry block holdout.
- [ ] Rebuild UQ with a distinct held-out calibration partition.
- [ ] Complete mechanism-preference, error and applicability-domain analyses.

## Paper

- [x] Replace the Q1 title and provisional protocol-level abstract.
- [x] Rewrite Introduction, Methods, and Evaluation Protocol around the frozen
      protocol.
- [ ] Replace test-selected ensemble and confounded ablation tables.
- [ ] Replace old OOD and UQ sections with canonical results.
- [x] Remove prospective-DFT success claims from the rewritten manuscript
      framing and state their exclusion from the evidence contract.
- [ ] Generate all main figures and supplementary tables from canonical data.
- [x] Add provisional Data and Code Availability statements.
- [ ] Compile PDF and resolve LaTeX warnings that affect content.
- [ ] Cross-check all numerical claims against result files.
- [ ] Complete independent review and final revision.
