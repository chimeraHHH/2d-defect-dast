# Literature positioning audit

Audit date: 2026-07-29

PRB redirection delta audit: 2026-08-10

## DOI integrity

All 21 cited DOI-bearing records in `references.bib` were queried against
Crossref and the DOI resolver. Twenty resolved to metadata consistent with the local
title and authors. The record previously keyed as `definet2025` did not:

- Claimed DOI: `10.1021/acs.chemmater.4c02907`
- Claimed title: *DefiNet: Equivariant Graph Neural Network for Predicting
  Defect Formation Energies in Crystalline Materials*
- Claimed authors: Park et al.
- Audit result: no Crossref record, DOI resolver HTTP 404, and no matching
  publisher or title record.

That record and its manuscript citation were removed. A real work called
DefiNet is Yang et al., *Modeling crystal defects using defect informed neural
networks*, npj Computational Materials 11, 229 (2025),
`10.1038/s41524-025-01728-w`. It predicts relaxed defect structures rather
than defect formation energies, so it is not substituted into the
formation-energy literature sentence.

## Closest prior work

- Davidsson et al. introduced IMP2D and its neutral adsorbate/interstitial
  formation-energy data (`10.1038/s41699-023-00380-6`).
- Kesorn et al. applied tree-based models directly to IMP2D and reported
  random, grouped, and blind evaluations (`10.1088/2632-2153/ad66ae`).
- El Alouani et al. studied how IMP2D data engineering changes descriptor
  model selection (`10.1016/j.mtphys.2025.102006`).
- Kazeev et al. learned 2D defect properties from the distinct 2DMD database
  using sparse defect representations (`10.1038/s41524-023-01062-z`).
- Witman et al., Rahman et al., Fang and Yan, and Kiyohara et al. establish
  neighboring graph-learning routes for bulk or semiconductor defect
  energetics; none supplies the same audited IMP2D comparison protocol used
  here.

## Defensible contribution boundary

The paper must not claim the first graph model for defects, the first machine
learning study of IMP2D, or a new DFT dataset. Its defensible contribution is
the combination of:

1. replayed IMP2D provenance, reconstruction checks, impurity-node exclusions,
   and coordinate plus invariant-distance redundancy control;
2. a paired repeated full-factorial test of three defect-aware modules;
3. aligned graph and descriptor baselines across random, host, impurity,
   host--impurity-pair, and predefined block transfer;
4. internal dedicated-split uncertainty calibration, selective prediction,
   and materials-facing screening-regret analysis with trivial decision
   references.

Claims remain limited to the audited IMP2D evidence. No external DFT
calculation is represented as validation.

## PRB physics-context delta

The PRB revision adds two established physics references rather than new model
comparators:

- Freysoldt et al., *First-principles calculations for point defects in
  solids*, Reviews of Modern Physics 86, 253 (2014),
  `10.1103/RevModPhys.86.253`, for the defect-formation-energy reference-state
  framework.
- Komsa and Krasheninnikov, *Native defects in bulk and monolayer MoS2 from
  first principles*, Physical Review B 91, 125304 (2015),
  `10.1103/PhysRevB.91.125304`, for the structural and energetic distinction
  between bulk and monolayer defect environments.

Their titles, authors, venues, volumes, pages/article numbers, years, and DOIs
were verified against Crossref records on 2026-08-10. The IMP2D article and its
Supplementary Note 3 were also checked directly: the source convention uses
the energy of the dopant element's relaxed crystalline phase as the chemical
potential. This supports the expanded definition following Eq. (1); it does
not add a new computed result.
