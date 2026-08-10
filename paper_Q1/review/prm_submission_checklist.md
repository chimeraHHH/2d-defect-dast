# Physical Review Materials submission checklist (superseded)

This checklist is retained as revision history. The active target-journal gate
is `prb_submission_checklist.md`.

Verified against the APS author pages on 2026-07-28:

- [Physical Review Materials information for authors](https://journals.aps.org/prmaterials/authors)
- [APS data availability guidelines](https://journals.aps.org/authors/data-availability-statements)
- [PRM section selection](https://journals.aps.org/prmaterials/authors/guidelines-section-selection-physical-review-materials)

## Article route

- [x] Target a Physical Review Materials Regular Article. APS lists no fixed
      length limit for this article type.
- [x] Use REVTeX 4.2 with `aps,prmaterials,reprint`.
- [x] Keep figures inline with their captions for review.
- [ ] Select the standard non-open-access publication route during submission.
- [ ] Suggest section M3-A, Development of new methods for materials, in the
      cover letter.

## Scientific package

- [x] State the question, relation to prior work, evaluation protocol, and
      limitations for a broad materials audience.
- [x] Complete the J-stage narrative-convergence pass around the three durable
      claims: paired module evidence, axis-resolved transfer, and bounded
      retrospective screening utility.
- [x] Populate all result-gated sections from canonical bundles.
- [x] Inspect all figure labels for readability without relying on color alone.
- [x] Supply Supplemental Material as a separate file and cite it from the main
      manuscript using the APS publisher-link placeholder.
- [x] Include exact model, optimization, seed, and descriptor-search
      specifications in the Supplemental Material.
- [x] Complete the post-result skeptical review; no additional non-DFT
      scientific experiment is required under the current claim boundary.

## Data and software

- [x] Include a Data and Code Availability section in the manuscript.
- [ ] Create the final tagged release only after the paper-facing artifacts and
      manuscript source are committed.
- [ ] Add a reference-list entry for the released data/software. APS requires
      publicly shared data and software to be cited in the reference list.
- [ ] Replace the provisional availability sentence with the actual version,
      repository/archive, and persistent identifier. A DOI is preferred but
      must not be invented.
- [x] Confirm that every plotted value and table is present in a
      machine-readable repository file selected by the paper asset manifest.
- [ ] Decide and record the repository/data license with the rights holders.

## Author metadata

- [ ] Replace `% TODO: add other authors` with the verified author order,
      affiliations, and contributions.
- [ ] Resolve the candidate metadata present only in the older PeriDefT draft
      (`Yiming Hua, Leyan Wu, Jiaqi Bi, Sheng Chang`) with all authors before
      transferring any name or order into this manuscript.
- [ ] Add the post-publication contact author and verified email using an APS
      byline footnote.
- [ ] Verify funder name and award wording with the authors.
- [ ] Collect ORCID identifiers from authors who want them associated at
      submission.

## Submission files

- [x] Current scientific-content manuscript PDF with no undefined references or
      content-affecting warnings, overfull boxes, clipping, or overlap; final
      byline and release metadata remain administrative substitutions.
- [x] Complete LaTeX source, bibliography, Supplemental Material, and all
      figure files.
- [x] Draft cover letter summarizing context, key findings, and PRM fit.
- [ ] Replace the draft cover letter's author-supplied placeholders for
      submission history, conflicts, release metadata, and correspondence.
- [ ] Optional suggested/excluded referees, supplied by the authors rather than
      inferred.
- [ ] Copy the final Data Availability Statement into the APS submission form.
