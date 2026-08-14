# AutoSci J-stage PRB review

Review date: 2026-08-10

## Stage decision

The project remains in AutoSci stage J: target-journal redirection, manuscript
optimization, independent review, and submission gating. No training, model
inference, DFT, or GPU computation was authorized or performed in this pass.
The PRB rewrite and local paper QA are complete, but the manuscript is
**NO-GO for submission** because the scientific red team found a model-level
order-sensitivity risk that cannot be removed by prose.

## PRB-facing contribution

1. Chemical transfer within IMP2D is axis dependent. Pair-held-out error is
   0.450 eV, while complete host and impurity holdout reach 0.938 and 0.844 eV.
   A split audit confirms that 10,223 of 10,224 pair-held-out test structures
   retain both constituent identities in the corresponding training fold.
2. Incorporation geometry is a second applicability axis: pair-held-out
   interstitial and adsorbate MAEs are 0.719 and 0.322 eV.
3. Gated readout and the composite pre-normalized, distance-gated local block
   show small paired improvements; the independent E effect remains unresolved.
4. Screening remains retrospective and conditional on available DFT-relaxed
   candidates. UQ remains an internal marginal-calibration result.

## Scientific red-team gate

- The angle branch retains the first 32 ordered neighbor pairs at each center.
  When more pairs exist, the selected triplets depend on stored neighbor order;
  exact atom-permutation robustness is not established.
- The global radial bias wraps fractional differences componentwise. In oblique
  cells this need not be the shortest periodic image.
- The manuscript now discloses both implementation details and withdraws the
  strict permutation-invariance and minimum-image claims. Disclosure does not
  resolve the first issue. The next stage requires a user decision: targeted
  permutation-sensitivity evidence, invariant reconstruction with a new
  evidence milestone, or stop/retarget.

## Validation result

- Independent numerical audit: 1,346 of 1,346 checks passed, including the
  10,223/10,224 pair-constituent statement.
- Targeted protocol/result/paper tests: 22 passed.
- Compiled package: 13-page main manuscript and 3-page Supplemental Material.
- Build diagnostics: no overfull boxes, missing citations/references, fatal
  errors, or unembedded fonts.
- Visual QA: all 16 pages inspected at final size; no clipping, overlap, broken
  glyphs, or unreadable equations/tables. Figure 1's baked-in wording and Figure
  2's grayscale redundancy remain explicit blockers.
- PDF SHA256: main
  `8924fe74619198ee6afab9e8b8fcc295932d4c612e3d91b3127c4f9a714c2261`;
  supplement
  `f7d430ab9372f81e5e26ec04fbde0a3d29e0b94210371b6f38b4fe16b4592c6b`.

## Submission blockers

- scientific decision on order-sensitive triplet construction and oblique-cell
  radial distance;
- correction of Figure 1 source wording and noncolor encoding for Figure 2;
- creation tool/version/rights/AI provenance for all user-supplied manual
  figures;
- complete author order, affiliations, corresponding-author contact/ORCID,
  contributions, funder award, conflicts, and submission history;
- author-approved license and a public tagged archival release with persistent
  identifier and creators;
- removal or regeneration of the stale PRM source archive.

The complete decision record is
`paper_Q1/ccfa-review-reports/2026-08-10-learning-impurity-incorporation-prb-review.md`.
No commit, push, release, or journal upload was performed.
