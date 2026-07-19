# PRM evidence intake audit

## Current state

- State class: `draft_ready`, `baseline_partial`, `analysis_partial`.
- Git anchor: GitHub `main` at `5ae045b5b717ac337b78461c62b73a251365afae`.
- Writing anchor: `paper_Q1/`; its tracked contents match the archived Q1 copy.
- Data anchor: `/home/huayiming/Workspace/yiminghua/project/data/processed/cleaned_dataset.pkl`.
- Compute anchor: `/home/huayiming/.conda/envs/yiminghua/bin/python` on WHU L40S.
- New result root: `/storage/ssd/metaiot_guest/yiminghua/prm_runs`.

## Trust ranking

| Asset | Trust | Manuscript role | Reason |
|---|---|---|---|
| Q1 LaTeX source and model code | usable with verification | revision source | tracked and matched to archive |
| IMP2D cleaned dataset | usable with verification | canonical dataset | local/remote hash previously matched; schema audit pending |
| V2 seeds 42-45 | usable with verification | development comparator | fixed targets and archived checkpoints, but old protocol was repeatedly inspected |
| V4 MoE single run | usable with verification | development comparator | only one seed; no significant advantage over V2 |
| Existing 2^3 component runs | reference only | none until rerun | incomplete seed coverage and historical baseline was confounded |
| Greedy 0.344 eV ensemble | stale/conflicting | internal only | ensemble membership selected against test MAE |
| 0.206 eV result | stale/conflicting | excluded | augmentation-before-split leakage |
| Existing constrained OOD table | stale/conflicting | internal only | trainer used the earlier model class rather than DART |
| Existing UQ table | stale/conflicting | internal only | temperature calibrated on a subset of the test set |
| Existing prospective DFT result | reference only | limitations only | incompatible reference protocol and same-set elemental correction |

## Reusable assets

- Model implementation and archived checkpoints can initialize diagnostics.
- The historical split is retained as a named development benchmark.
- Existing scripts are implementation references, not authoritative evaluators.
- Negative variants such as LDS, RnC and heteroscedastic regression can be
  summarized in the Supplement after provenance checks.

## Stale routes to ignore

- Architecture-SOTA framing based on the test-selected ensemble.
- “First defect-aware GNN” language.
- Causal interpretation of attention concentration.
- Prospective discovery hit-rate or corrected-DFT-accuracy claims.

## Next decision scope

First establish whether the three DART core components have reproducible main
effects or interactions. That evidence determines whether the final paper leads
with architecture or with applicability-domain analysis.
