# Required follow-up experiments and analyses

## EXP-001: Complete aligned neural campaigns

- Status: in progress. The comparison inputs are complete; the five-member UQ
  campaign is still running.
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

- Matrix row: E5.
- Route: existing-result analysis after EXP-001.
- Tier: main required.
- Minimum task: fit variance scale/floor and conformal quantiles on their
  assigned calibration halves, then evaluate untouched test predictions.
- Acceptance: coverage, width, NLL, CRPS, uncertainty--error rank correlation,
  AURC, and excess AURC all pass finite-value and provenance checks.
- Placement: uncertainty table and figure.

## EXP-004: Generate materials-facing decision analysis

- Status: ready to execute from the complete pair-held-out OOF predictions.
- Matrix row: E6.
- Route: existing-result analysis after EXP-002.
- Tier: main required.
- Minimum task: compute incorporation-class preference, exact/top-two site
  recovery, and regret from pair-held-out out-of-fold predictions.
- Acceptance: eligibility counts, tie handling, cluster-bootstrap intervals,
  and prediction coverage agree with the frozen contract.
- Placement: screening table and figure.

No additional DFT calculation is part of this frontier.
