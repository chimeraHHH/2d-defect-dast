# Paper experiment matrix

Status date: 2026-07-28

| ID | Purpose | Durable status | Evidence anchor | Manuscript role |
|---|---|---|---|---|
| E1 | Audit IMP2D targets, identity, redundancy, and immutable partitions | Complete | `artifacts/prm_protocol_v2/`; `artifacts/prm_results/paper/protocol_figure.json` | Methods and protocol figure; supports the 10,224-structure evidence boundary |
| E2 | Test G/E/P modules in a paired repeated full factorial | Complete and written | `artifacts/prm_results/factorial/` | Architecture selection and orthogonal-effect claims |
| E3 | Select descriptor comparators using validation data on every formal regime | Complete, awaiting joint table | `artifacts/prm_results/descriptors/` | Classical baseline rows in the main comparison |
| E4 | Compare promoted DART with periodic SchNet under aligned transfer partitions | Running; no partial result admissible | remote queues `dart-g111-v1` and `schnet-v1` | Main transfer table, paired intervals, and applicability diagnostics |
| E5 | Evaluate five-member DART uncertainty on a separate calibration/test split | Running; no partial result admissible | five promoted UQ configurations in `dart-g111-v1` | Calibration table, risk--coverage figure, and abstention claims |
| E6 | Convert pair-held-out predictions into preference/site-screening decisions and descriptive error-heterogeneity diagnostics | Analysis implemented; predictions pending E4 | `scripts/prm_materials_analysis.py` | Screening table, regret figure, and host/impurity/incorporation error analysis |
| E7 | Generate fail-closed manuscript tables, figures, and macros | Pending E4--E6 | `scripts/prm_make_result_assets.py` | Releases all numerical Results, Discussion, and Conclusion text |

## Claim mapping

| Claim | Required rows | Current classification |
|---|---|---|
| C1: the reported modeling set is reconstructable and leakage controlled under the declared audits | E1 | completed and written |
| C2: selected architectural effects are supported or inconclusive under paired repeats | E2 | completed and written |
| C3: interpolation and chemical-transfer behavior differ across held-out axes and comparators | E3, E4 | written but not yet fully evidenced |
| C4: held-out ensemble uncertainty supports a bounded selective-prediction policy | E5 | written but not yet evidenced |
| C5: pair-held-out predictions have measurable screening utility and regret within IMP2D | E4, E6 | written but not yet evidenced |

No claim in C3--C5 becomes paper-ready until E7 creates the readiness marker
from complete, collector-validated bundles.
