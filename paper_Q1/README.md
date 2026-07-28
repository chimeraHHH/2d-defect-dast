# PRM manuscript source

`main.tex` is the active manuscript and `main.pdf` is its compiled preview.
The paper uses the APS `prmaterials` REVTeX profile.

Numerical Results, Discussion, and Conclusion content is exposed only when
`generated/results_ready.tex` exists. That gate is created by the canonical
result pipeline after all required non-DFT runs and provenance checks pass.
The current gate is present and the populated scientific content has passed
the independent numerical and final skeptical reviews. If the gate is absent,
placeholder text is intentional and historical values must not be inserted
manually.

Canonical inputs:

- `../artifacts/prm_protocol_v2/` for dataset and split provenance.
- `../artifacts/prm_assets/manifest.json` for initialization provenance.
- `../artifacts/prm_results/` for validated metrics and operation records.
- `../configs/prm/` for the controlled training contract.

Only `fig_protocol_overview.pdf` is present before result release. The
remaining manuscript figures are created by
`../scripts/prm_make_result_assets.py` and must be declared in the canonical
result manifest. Historical exploratory figures and their hard-coded
generators remain available through Git history, not in the active submission
source.

Verify the paper-facing numbers and asset hashes from the repository root:

```bash
python3 scripts/prm_verify_paper_numbers.py
```

The deterministic report is
`../artifacts/prm_results/paper/numeric_audit.json`.

Build from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

Tectonic is an equivalent fallback when `latexmk` is unavailable:

```bash
tectonic --keep-logs --keep-intermediates main.tex
tectonic --keep-logs --keep-intermediates supplement.tex
```

The remaining placeholders in `main.tex` are administrative: verified author
order/affiliations, corresponding-author contact, funder wording, and the final
tagged archival release identifier.
