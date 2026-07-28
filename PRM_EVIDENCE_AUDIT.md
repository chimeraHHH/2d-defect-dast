# PRM evidence intake audit

This file records the evidence-routing decision made at intake.  Live execution
status is maintained in `PLAN.md` and `CHECKLIST.md`.

## Current state

- State class: `protocol_frozen`, `factorial_complete`,
  `downstream_campaigns_active`, `paper_result_gated`.
- Git anchor: GitHub branch `prm-revision`; every formal campaign records its
  own clean commit rather than inheriting the moving branch tip.
- Writing anchor: `paper_Q1/`; the Q1 source has been rewritten as the active
  PRM manuscript and its numerical sections remain fail-closed.
- Data anchor: `/home/huayiming/Workspace/yiminghua/project/data/processed/cleaned_dataset.pkl`.
- Canonical dataset SHA-256:
  `1d59cc818d81252d49da525c6d77e2da8549ceb604d54af953d1caa4fb974a9b`.
- Compute anchor: `/storage/ssd/metaiot_guest/yiminghua/prm_env/bin/python`
  on WHU L40S.
- Formal result root: `/storage/ssd/metaiot_guest/yiminghua/prm_runs_v2`.
- Completed factorial anchor: clean training commit
  `7a6593a7eaaf991fd9c18a15653c3d888b060c7e`, queue
  `factorial-v4-indexed`, and clean collector commit
  `865e90a695e3641e21bc9f2c98fc05c80723abcf`.
- Active downstream anchor: clean commit
  `6c401374baa9ac22b5fa4353d467366ac7cc222f`, queues
  `dart-g111-v1` and `schnet-v1`, with disjoint GPU sets and GPU 2 excluded.

## Trust ranking

| Asset | Trust | Manuscript role | Reason |
|---|---|---|---|
| PRM LaTeX source and controlled model code | authoritative when clean | active manuscript and formal runs | result-gated, tested, and committed on `prm-revision` |
| IMP2D protocol v2 | authoritative | canonical dataset and splits | target, identity, duplicate, schema, and hash audits complete |
| Corrected ct-UAE and JARVIS initialization assets | authoritative when hash-matched | DART initialization | exact hashes and copied/seeded tensors are checked in every run |
| Protocol-v2 descriptor archive | authoritative | classical baselines | 27 formal splits collected with validation-only model selection |
| Corrected paired factorial | authoritative | architecture selection | all 40 runs passed collector validation; `g111` was selected by mean validation MAE only |
| Promoted DART and periodic SchNet campaigns | pending collection | transfer, comparison, and UQ | clean 48-run queues are active; partial metrics are inadmissible |
| V2 seeds 42-45 | usable with verification | development comparator | fixed targets and archived checkpoints, but old protocol was repeatedly inspected |
| V4 MoE single run | usable with verification | development comparator | only one seed; no significant advantage over V2 |
| Existing 2^3 component runs | reference only | none until rerun | incomplete seed coverage and historical baseline was confounded |
| Greedy 0.344 eV ensemble | stale/conflicting | internal only | ensemble membership selected against test MAE |
| 0.206 eV result | stale/conflicting | excluded | augmentation-before-split leakage |
| Existing constrained OOD table | stale/conflicting | internal only | trainer used the earlier model class rather than DART |
| Existing UQ table | stale/conflicting | internal only | temperature calibrated on a subset of the test set |
| Existing prospective DFT result | reference only | limitations only | incompatible reference protocol and same-set elemental correction |

## Reusable assets

- The immutable 10,224-structure protocol and its split hashes define all
  formal training and analysis.
- The corrected DART implementation, periodic SchNet comparator, collectors,
  and result-asset generator form the active executable chain.
- The historical split is retained only as a named development benchmark.
- Historical checkpoints can initialize diagnostics but cannot replace a
  protocol-v2 formal run.
- Negative variants such as LDS, RnC and heteroscedastic regression can be
  summarized in the Supplement after provenance checks.

## Stale routes to ignore

- Architecture-SOTA framing based on the test-selected ensemble.
- “First defect-aware GNN” language.
- Causal interpretation of attention concentration.
- Prospective discovery hit-rate or corrected-DFT-accuracy claims.

## Next decision scope

Allow both downstream queues to complete without failures, then collect all 96
runs from a separate clean checkout.  Generate comparison, uncertainty, and
materials analyses only after complete-run validation.  The final narrative
will be chosen from the canonical intervals and applicability-domain evidence,
not from partial queue metrics.
