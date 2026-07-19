# PRM revision checklist

## Protocol

- [x] Create clean `prm-revision` branch from GitHub `main`.
- [x] Record dataset SHA256, size, schema and target/metadata summary.
- [x] Generate and validate immutable development, host, dopant and block splits.
- [x] Add split-file support and sample-index preservation to the trainer.
- [x] Add run manifest and canonical prediction schema.
- [x] Pass unit tests and one bounded remote smoke run.

## Experiments

- [ ] Run 2^3 architecture factorial for seeds 42-46.
- [ ] Run descriptor baselines with validation-only tuning.
- [ ] Run SchNet comparator under the same split contract.
- [ ] Select the final DART configuration from validation/CV evidence only.
- [ ] Run host-group cross-validation.
- [ ] Run dopant-group cross-validation.
- [ ] Run predefined host-by-dopant chemistry block holdout.
- [ ] Rebuild UQ from validation predictions only.
- [ ] Complete mechanism-preference, error and applicability-domain analyses.

## Paper

- [ ] Replace the Q1 title, abstract and contribution statement.
- [ ] Rewrite Methods around the frozen protocol.
- [ ] Replace test-selected ensemble and confounded ablation tables.
- [ ] Replace old OOD and UQ sections with canonical results.
- [ ] Remove prospective-DFT success claims and retain only a limitations note.
- [ ] Generate all main figures and supplementary tables from canonical data.
- [ ] Add Data Availability, Code Availability and reproducibility statements.
- [ ] Compile PDF and resolve LaTeX warnings that affect content.
- [ ] Cross-check all numerical claims against result files.
- [ ] Complete independent review and final revision.
