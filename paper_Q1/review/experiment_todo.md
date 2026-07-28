# Required follow-up experiments and analyses

## EXP-001: Complete aligned neural campaigns

- Status: complete. The formal queues contain exactly 48 complete DART runs and
  48 complete SchNet runs, including all five promoted UQ members.
- Matrix rows: E4 and E5.
- Route: existing frozen DART and SchNet campaigns.
- Tier: main required.
- Minimum task: reach exactly 48 complete DART runs and 48 complete SchNet runs
  with zero failed, stopped, pending, or running jobs.
- Acceptance: collector verifies config identity, 150-epoch histories, split
  hashes, predictions, recomputed metrics, checkpoints, and declared hashes.
- Placement: main comparison and uncertainty sections.

## EXP-002: Generate comparison and applicability evidence

- Status: complete in `artifacts/prm_results/comparison/`.
- Matrix row: E4.
- Route: existing-result analysis after EXP-001.
- Tier: main required.
- Minimum task: collect pooled metrics, macro-group errors, low-energy
  diagnostics, and paired bootstrap intervals for every declared regime.
- Acceptance: no test-driven architecture or descriptor-family selection;
  every plotted value is reproduced from canonical prediction files.
- Placement: transfer table, applicability table, and transfer figure.

## EXP-003: Generate held-out uncertainty evidence

- Status: complete in `artifacts/prm_results/uq/`.
- Matrix row: E5.
- Route: existing-result analysis after EXP-001.
- Tier: main required.
- Minimum task: fit variance scale/floor and conformal quantiles on their
  assigned calibration halves, then evaluate untouched test predictions.
- Acceptance: coverage, width, NLL, CRPS, uncertainty--error rank correlation,
  AURC, and excess AURC all pass finite-value and provenance checks.
- Placement: uncertainty table and figure.

## EXP-004: Generate materials-facing decision analysis

- Status: complete in `artifacts/prm_results/materials/`.
- Matrix row: E6.
- Route: existing-result analysis after EXP-002.
- Tier: main required.
- Minimum task: compute incorporation-class preference, exact/top-two site
  recovery, and regret from pair-held-out out-of-fold predictions.
- Acceptance: eligibility counts, tie handling, cluster-bootstrap intervals,
  and prediction coverage agree with the frozen contract.
- Placement: screening table and figure.

## EXP-005: Bound the SchNet readout interpretation

- Status: complete. All 15 runs finished from clean commit `91cb8ac` on GPUs
  1, 4, and 5; GPU 2 and all other devices were explicitly excluded.
- Matrix row: E9.
- Route: post-hoc exploratory robustness campaign.
- Tier: supporting.
- Question: does replacing atomwise sum readout with mean readout reduce the
  repeated host-held-out SchNet error?
- Intervention: change only `model_kwargs.readout` from `add` to `mean`.
- Fixed conditions: all five host-CV folds, seeds 342--344, dataset, splits,
  SchNet interaction stack, optimizer, schedule, sampling, label noise, and
  150-epoch budget.
- Acceptance: exactly 15 complete runs, aligned out-of-fold predictions,
  host-cluster paired interval, size--error diagnostics, and archived
  provenance.
- Evidence: `artifacts/prm_results/sensitivity/schnet_readout/`; all 15
  run archives and their output hashes were independently verified after
  collection from clean commit `f1756e4`.
- Result: mean readout lowers pooled host-CV MAE from 6.703 to 2.552 eV. The
  paired mean-minus-add difference is -4.151 eV with a 95% host-cluster
  bootstrap interval of [-7.704, -1.256] eV, and all five folds improve.
- Interpretation: exploratory and mechanism-informative; it does not replace
  the prespecified additive-SchNet main comparator, and the residual gap cannot
  be assigned to the SchNet backbone alone.
- Placement: complete in Discussion and Supplemental Material.

No additional DFT calculation is part of this frontier.
