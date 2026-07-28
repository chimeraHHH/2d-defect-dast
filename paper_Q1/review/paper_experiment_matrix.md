# Paper experiment matrix

Status date: 2026-07-29

| ID | Purpose | Durable status | Evidence anchor | Manuscript role |
|---|---|---|---|---|
| E1 | Audit IMP2D targets, identity, redundancy, and immutable partitions | Complete | `artifacts/prm_protocol_v2/`; `artifacts/prm_results/paper/protocol_figure.json` | Methods and protocol figure; supports the 10,224-structure evidence boundary |
| E2 | Test G/E/P modules in a paired repeated full factorial | Complete and written | `artifacts/prm_results/factorial/` | Architecture selection and orthogonal-effect claims |
| E3 | Select descriptor comparators using validation data on every formal regime | Complete and jointly collected | `artifacts/prm_results/descriptors/`; `artifacts/prm_results/comparison/descriptor_selection.json` | Classical baseline rows in the main comparison |
| E4 | Compare promoted DART with periodic SchNet under aligned transfer partitions | Complete and collector-validated | `artifacts/prm_results/comparison/` | Main transfer table, paired intervals, and applicability diagnostics |
| E5 | Evaluate five-member DART uncertainty on a separate calibration/test split | Complete and collector-validated | `artifacts/prm_results/uq/` | Calibration table, risk--coverage figure, and abstention claims |
| E6 | Convert pair-held-out predictions into preference/site-screening decisions and descriptive error-heterogeneity diagnostics | Complete and collector-validated | `artifacts/prm_results/materials/` | Screening table, regret figure, and host/impurity/incorporation error analysis |
| E7 | Generate fail-closed manuscript tables, figures, and macros | Complete and hash-validated as schema v2 with 23 outputs | `artifacts/prm_results/paper/result_assets.json`; `paper_Q1/generated/`; `paper_Q1/figures/` | Releases all numerical Results, Discussion, Conclusion, and Supplemental Material text |
| E8 | Audit chemistry overlap between JARVIS source-task pretraining and canonical IMP2D | Complete and written | `artifacts/prm_results/operations/pretraining_overlap_3a98513.json` | Bounds grouped transfer as holdout from IMP2D target supervision |
| E9 | Test whether SchNet's additive graph readout contributes to host-held-out failure | Complete and collector-validated; 15/15 runs | `artifacts/prm_results/sensitivity/schnet_readout/`; `configs/prm/sensitivity/schnet_mean_host/` | Post-hoc exploratory robustness result written in Discussion and Supplemental Material |

## Claim mapping

| Claim | Required rows | Current classification |
|---|---|---|
| C1: the reported modeling set is reconstructable and leakage controlled under the declared audits | E1 | completed and written |
| C2: selected architectural effects are supported or inconclusive under paired repeats | E2 | completed and written |
| C3: interpolation and chemical-transfer behavior differ across held-out axes and comparators | E3, E4, E8 | asset-backed and written; final independent review pending |
| C4: held-out ensemble uncertainty supports a bounded selective-prediction policy | E5 | asset-backed and written; final independent review pending |
| C5: pair-held-out predictions have measurable screening utility and regret within IMP2D | E4, E6 | asset-backed and written; final independent review pending |
| C6: the SchNet host-held-out result is interpreted at the complete-recipe rather than backbone-only level | E4, E9 | asset-backed and written; final independent review pending |

C3--C6 have the hash-validated E7 schema-v2 readiness marker and are present in
the compiled manuscript package. The independent numerical audit passes 1,265
checks with zero failures. All claims remain subject to the final skeptical
prose review gate; author and archival metadata are separate administrative
blockers.
