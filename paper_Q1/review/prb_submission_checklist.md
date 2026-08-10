# Physical Review B submission checklist

Audit date: 2026-08-10

Official sources:

- [PRB Information for Authors](https://journals.aps.org/prb/authors)
- [PRB scope and editorial criteria](https://journals.aps.org/prb/about)
- [PRB section selection](https://journals.aps.org/prb/authors/guidelines-section-selection-physical-review-b)
- [APS Style Basics](https://journals.aps.org/authors/style-basics)
- [APS web-submission guidelines](https://journals.aps.org/authors/web-submission-guidelines-physical-review)
- [APS Data Availability Statements](https://journals.aps.org/authors/data-availability-statements)
- [APS Supplemental Material instructions](https://journals.aps.org/authors/supplemental-material-instructions)
- [APS Appropriate Use of AI Tools](https://journals.aps.org/authors/appropriate-use-ai-tools)

## Article type and journal fit

- [x] Prepare a PRB Regular Article; PRB sets no fixed length limit for this
      article type.
- [x] Use REVTeX 4.2 with `aps,prb,reprint` in the main manuscript and `prb`
      in the Supplemental Material.
- [x] Target condensed-matter/materials-physics readers through chemical-axis
      transfer, non-extensive formation-energy readout, and incorporation-
      geometry asymmetry rather than model leaderboard language.
- [x] Suggest B15(4), surface/nanoscale/low-dimensional systems, as the primary
      section and B1(1), crystal defects, as a secondary section.
- [ ] Select accurate PhySH concepts in the submission system.
- [ ] Choose the standard subscription route; do not select optional Gold OA.

## Manuscript content

- [x] Keep the abstract as one self-contained paragraph under 500 words with no
      citations.
- [x] Define the IMP2D formation-energy terms and crystalline impurity reference
      state after Eq. (1).
- [x] State that predictions are conditional on available DFT-relaxed candidate
      geometries and do not replace site generation or structural relaxation.
- [x] Treat DART--SchNet as a complete-recipe comparison and the readout
      sensitivity as a diagnostic, not a universal backbone claim.
- [x] Report the adsorbate/interstitial error asymmetry and the asymmetric
      impurity-identity exclusions in the main text.
- [x] Keep UQ claims at internal marginal calibration and risk ranking.
- [x] Disclose the order-sensitive 32-triplet cap and componentwise-wrapped
      global radial distance in Methods and Discussion.
- [ ] Complete the authorized G1 server acceptance: legacy permutation/MIC
      diagnostic, invariant triplets, exact shortest-image property tests, and
      one repaired-fold pilot. Text qualification alone does not pass this gate.
- [ ] Complete G1C source-task repair.  The historical JARVIS graph construction
      has a triplet-center column mismatch, stored-order-dependent bounded
      selection, and a nonperiodic dense distance matrix.  Rebuild the frozen
      19,902-record subset, retrain a versioned checkpoint, and bind both hashes
      in receipts.  Until then, a repaired IMP2D run using the old checkpoint is
      only a legacy-initialization sensitivity arm.
- [ ] Keep all current paper numbers labeled as legacy-graph evidence. G2 full
      repaired OOF is not authorized or available, so repaired-method wording
      and canonical G3 claims must not enter the manuscript.
- [x] Include titles in numbered references.
- [x] Place the Data Availability Statement after Acknowledgments and cite the
      public project repository as a numbered reference.
- [ ] Replace the current GitHub commit citation with the final versioned
      archival release, persistent identifier, creators, and approved license.

## Authors and declarations

- [ ] Confirm the complete author order, affiliations, and contributions.
- [ ] Designate exactly one corresponding author in the submission system.
- [ ] Add the corresponding-author email footnote to the manuscript.
- [ ] Supply the corresponding author's ORCID and every author's email.
- [ ] Verify the funder name and award number.
- [ ] Confirm originality, concurrent-submission status, Physical Review
      submission history, and conflicts of interest.
- [x] Disclose substantive OpenAI Codex assistance in the Acknowledgments,
      including the tool, its role, author direction, and author verification.
- [x] Include AI-assisted computational code and analysis verification in the
      consolidated Acknowledgments disclosure; confirm before submission that
      this description covers the complete project history.
- [ ] Resolve all bracketed cover-letter fields and optional referee requests.

## Figures and Supplemental Material

- [x] Upload the main manuscript and Supplemental Material as separate PDFs.
- [x] Keep Supplemental references represented in the main bibliography and
      cite Supplemental Material once as a numbered reference.
- [x] Provide `README.TXT` for the Supplemental Material upload.
- [ ] Correct the baked-in Figure 1 phrases “1 binary impurity embedding” and
      “Radial bias up to 12 Å” in the source artwork through the approved
      AutoFigure editing workflow.
- [ ] Add fully redundant non-color encoding to Figure 2 and recheck grayscale
      legibility.
- [ ] Confirm the creation tools, versions, rights, and author verification for
      all three user-supplied manual figures.  If any figure was AI-generated,
      add the tool and use disclosure to that figure's caption before submission.
- [ ] Repeat final-size visual QA after the two artwork corrections.

## Build and release gate

- [x] Rebuild `main.pdf` and `supplement.pdf` after the PRB prose revision.
- [x] Confirm no missing citations/references, overfull boxes, clipped figures,
      or unembedded fonts.
- [x] Rerun the independent numerical audit on the pushed J-R1 source after
      server-side figure-label regeneration: 1346 checks, 0 failures.
- [ ] Run the targeted paper test modules in an accepted server environment;
      the frozen server Python lacks `pytest`, so these tests were not run.
- [x] Record the verified rebuild: 12-page main PDF
      (`4aa3d11...35c275`) and 7-page Supplemental PDF
      (`ac1e5dba...8ba82`).
- [x] Author approval was given for scoped G1/G3 and figure/prose milestone
      commits and pushes; release and journal upload remain unapproved.
- [ ] Exclude or regenerate the stale `main_latex729.zip`; it contains the
      superseded PRM manuscript and is not a PRB submission source package.

## Current decision

**NO-GO for submission.** The decisive scientific blockers are unfinished G1
server acceptance (including corrected JARVIS pretraining lineage), the absence
of an authorized repaired full-OOF G2 bundle, and the resulting block on
canonical G3 evidence. The current manuscript
numbers remain legacy-graph evidence. Author metadata, archival release,
cover-letter declarations, and two figure-source corrections are additional
submission blockers.
