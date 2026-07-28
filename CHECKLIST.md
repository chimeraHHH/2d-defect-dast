# PRM revision checklist

## Protocol

- [x] Create clean `prm-revision` branch from GitHub `main`.
- [x] Record dataset SHA256, size, schema and target/metadata summary.
- [x] Generate and validate immutable development, host, dopant and block splits.
- [x] Recompute targets from raw components and exclude four source sentinels.
- [x] Exclude coordinate- or invariant-distance-equivalent duplicate rows.
- [x] Freeze protocol v2 with 10,224 uniquely labeled structures and a
      dedicated UQ split.
- [x] Add split-file support and sample-index preservation to the trainer.
- [x] Add run manifest and canonical prediction schema.
- [x] Pin ct-UAE and DFT-3D-lite initialization assets by SHA-256 and record
      the exact copied versus seeded parameter sets in every DART run.
- [x] Pass 133 remote unit tests from clean GitHub commit `866fcd4` in an
      isolated verification checkout.
- [x] Complete bounded protocol-v2 DART and periodic-SchNet CPU smoke runs
      from clean training commit `af6eeec`.
- [x] Reconstruct all 9,871,460 stored graph-edge distances from positions,
      periodic image shifts, and cells without topology or cutoff violations.
- [x] Correct ct-UAE atomic-number indexing with a zero-padding row; verify the
      published checkpoint derivation, exact Z=1/Z=100 endpoint mappings, and
      all 138 tests from a clean remote checkout.
- [x] Complete a two-epoch corrected-index GPU smoke at clean commit `7a6593a`;
      verify both asset hashes, all five declared output hashes, two history
      epochs, zero worker processes, and the 101-row checkpoint table.

## Experiments

- [x] Run 2^3 architecture factorial for seeds 42-46.
      (`factorial-v2` and `factorial-v3-fd0` remain scientifically excluded.
      The corrected 40-run `factorial-v4-indexed` queue completed from clean
      commit `7a6593a` with no failed jobs and GPU 2 excluded. Clean collector
      commit `865e90a` validated and archived all 40 runs; validation-only
      selection promoted `g111`.)
- [x] Rerun descriptor baselines on protocol v2 with validation-only tuning
      and select the learned family independently inside every split
      (27 formal splits; protocol-v1 outputs are superseded).
- [x] Hash the actual descriptor input file and all 27 metrics/prediction pairs,
      encode undefined constant-baseline correlations as strict-JSON `null`, and
      mirror the verified 55-file archive to the formal result store.
- [x] Run SchNet comparator under the same split contract. (All 48 formal
      `schnet-v1` runs completed from clean commit `6c40137` and passed the
      aligned comparison collector.)
- [x] Select the final DART configuration from validation/CV evidence only.
      (`g111`; test metrics were not used by the ordering rule.)
- [x] Run random out-of-fold and host--dopant-pair cross-validation.
- [x] Run host-group cross-validation.
- [x] Run dopant-group cross-validation.
- [x] Run predefined host-by-dopant chemistry block holdout.
- [x] Rebuild UQ with a distinct calibration partition and dedicated internal
      test partition. (All five members and the complete internal analysis are
      archived under `artifacts/prm_results/uq/`; the test is not described as
      untouched by earlier architecture development.)
- [x] Complete mechanism-preference, error and applicability-domain analyses.
      (The pair-OOF screening, trivial reference comparisons, and heterogeneity
      evidence are archived under `artifacts/prm_results/materials/`.)
- [x] Complete the bounded 15-run SchNet mean-readout host-CV sensitivity.
      (All 15 runs completed from clean commit `91cb8ac` on GPUs 1, 4, and 5.
      The hash-verified archive and paired host-cluster analysis are under
      `artifacts/prm_results/sensitivity/schnet_readout/`; mean readout reduces
      host-CV MAE from 6.703 to 2.552 eV, with a paired difference of
      -4.151 eV and 95% cluster-bootstrap interval [-7.704, -1.256] eV. This
      remains post-hoc supporting evidence and does not replace the additive
      main comparator.)

## Paper

- [x] Replace the Q1 title and provisional protocol-level abstract.
- [x] Rewrite Introduction, Methods, and Evaluation Protocol around the frozen
      protocol.
- [x] Replace test-selected ensemble and confounded ablation tables.
- [x] Replace old OOD and UQ sections with canonical results.
- [x] Remove prospective-DFT success claims from the rewritten manuscript
      framing and state their exclusion from the evidence contract.
- [x] Generate all main figures and supplementary tables from canonical data.
      (The 23 fail-closed outputs, including the SchNet readout sensitivity
      assets, and readiness marker are hash-bound in
      `artifacts/prm_results/paper/result_assets.json`.)
- [x] Add exact DART, SchNet, optimization, seed, and descriptor-search
      specifications to the Supplemental Material.
- [x] Add provisional Data and Code Availability statements.
- [x] Audit all DOI-bearing bibliography entries and remove the unresolvable
      `10.1021/acs.chemmater.4c02907` record and dependent claim.
- [x] Compile and visually inspect the provisional result-gated PDF.
- [x] Compile and visually inspect the populated main PDF and Supplemental
      Material; no undefined references, content-affecting warnings, overfull
      boxes, clipping, or overlap remain.
- [x] Independently cross-check all numerical claims against archived
      predictions and result files. (`scripts/prm_verify_paper_numbers.py`
      passes 1,331 checks with zero failures; the deterministic report is
      `artifacts/prm_results/paper/numeric_audit.json`.)
- [x] Complete independent review and final scientific-content revision.
