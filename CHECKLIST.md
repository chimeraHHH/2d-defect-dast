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
      (27 formal splits; protocol-v1 outputs are superseded).
- [x] Hash the actual descriptor input file and all 27 metrics/prediction pairs,
      encode undefined constant-baseline correlations as strict-JSON `null`, and
      mirror the verified 55-file archive to the formal result store.
- [ ] Run SchNet comparator under the same split contract. (Implementation,
      configurations, periodic-edge audit, and GPU smoke run complete; the
      formal 48-run `schnet-v1` queue is active from clean commit `6c40137`.)
- [x] Select the final DART configuration from validation/CV evidence only.
      (`g111`; test metrics were not used by the ordering rule.)
- [ ] Run random out-of-fold and host--dopant-pair cross-validation.
      (Formal `dart-g111-v1` queue active.)
- [ ] Run host-group cross-validation. (Formal queue active.)
- [ ] Run dopant-group cross-validation. (Formal queue active.)
- [ ] Run predefined host-by-dopant chemistry block holdout.
      (Formal queue active.)
- [ ] Rebuild UQ with a distinct held-out calibration partition. (The collector
      is fail-closed for non-finite inputs, undefined rank statistics, and
      non-standard JSON; five formal members are active in `dart-g111-v1`.)
- [ ] Complete mechanism-preference, error and applicability-domain analyses.
      (Eligibility and energy-tie contracts are implemented and tested; formal
      out-of-fold predictions await the selected architecture.)

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
- [x] Audit all DOI-bearing bibliography entries and remove the unresolvable
      `10.1021/acs.chemmater.4c02907` record and dependent claim.
- [x] Compile and visually inspect the provisional result-gated PDF.
- [ ] Compile the final populated PDF and resolve warnings that affect content.
- [ ] Cross-check all numerical claims against result files.
- [ ] Complete independent review and final revision.
