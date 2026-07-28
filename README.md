# IMP2D defect learning: PRM revision

This branch contains the controlled revision for a Physical Review Materials
submission on defect-aware graph learning of neutral impurity incorporation
energies in two-dimensional materials.

## Evidence status

`paper_Q1/` is the only active manuscript. Its numerical Results, Discussion,
and Conclusion are fail-closed until the complete canonical result pipeline
creates `paper_Q1/generated/results_ready.tex`.

The paper is limited to the provenance-audited IMP2D evidence. Historical
exploratory scores, unmatched first-principles calculations, and prospective
DFT outputs are not admissible evidence for the active manuscript. They remain
available through Git history and legacy scripts but are not summarized here.

## Authoritative surfaces

- `PLAN.md` and `CHECKLIST.md`: research scope, execution gates, and status.
- `artifacts/prm_protocol_v2/`: the immutable 10,224-structure protocol.
- `artifacts/prm_assets/manifest.json`: pinned ct-UAE and JARVIS pretraining
  provenance, hashes, tensor shapes, and atomic-number indexing.
- `configs/prm/`: factorial, promoted DART, and SchNet training contracts.
- `artifacts/prm_results/`: validated metrics, archived run evidence, and
  operation records.
- `paper_Q1/`: REVTeX manuscript source and compiled preview.

## External assets

Large data and model files are intentionally excluded from Git. Reproduction
must verify them against `artifacts/prm_assets/manifest.json`.

The canonical protocol tables are modified derivatives of version 2 of the
IMP2D Interstitial and Adsorbate Structure Database,
[`10.11583/DTU.19692238.v2`](https://doi.org/10.11583/DTU.19692238.v2),
described by Davidsson *et al.*,
[`10.1038/s41699-023-00380-6`](https://doi.org/10.1038/s41699-023-00380-6).
The upstream dataset is CC BY 4.0; its attribution, license link, and the
changes made by this study are recorded in `THIRD_PARTY_NOTICES.md`.

The ct-UAE source checkpoint is available from
[`fduabinitio/ct-UAE`](https://github.com/fduabinitio/ct-UAE) at commit
`0141ff9e09277d2229c9d7a24c1bcc5eac9de78e`. The derived 100 by 128 elemental
table is generated and checked as follows:

```bash
mkdir -p data/external/ct-uae
curl -L --fail \
  -o data/external/ct-uae/model_best_mt3_256.pth.tar \
  https://raw.githubusercontent.com/fduabinitio/ct-UAE/0141ff9e09277d2229c9d7a24c1bcc5eac9de78e/embeddings/model_best_mt3_256.pth.tar
python scripts/prm_verify_ct_uae_asset.py \
  --checkpoint data/external/ct-uae/model_best_mt3_256.pth.tar \
  --asset data/ct_uae_mt3_embeddings.pt \
  --derive \
  --expected-checkpoint-sha256 6f5305770a64a91f683e27c1d9b2c574f5b748b381598590e9b675ee9b43e26b \
  --expected-asset-sha256 ac77b2720b7bb8a3b290d6bfbae482857962d55daa7fc3486f2c7f44c41c60dc \
  --expected-tensor-sha256 acf66488770f2e82d81cf396c0c28a3a3c244e9825f7e0ae443d2ff0e7a32edc
```

The pinned ct-UAE source is MIT licensed; its notice is retained in
`THIRD_PARTY_NOTICES.md`. This notice does not assign a license to this
repository.

The JARVIS-DFT source-task subset is reconstructed with:

```bash
python scripts/fetch_dft_3d.py
python -m scripts.pretrain_element_embeddings \
  --config configs/pretrain_dft3d.yaml
```

This source corpus is not chemistry-disjoint from IMP2D. The recorded overlap
audit is
`artifacts/prm_results/operations/pretraining_overlap_3a98513.json`.

## Verification

Run the repository test suite in the pinned scientific Python environment:

```bash
python -m pytest -q
```

Build the provisional manuscript from `paper_Q1/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Do not create the result readiness marker manually. It is written last by
`scripts/prm_make_result_assets.py` only after the comparison, uncertainty,
materials, provenance, and output-hash contracts all pass.

## Scope boundary

The formal protocol covers neutral adsorbate and interstitial impurities under
the IMP2D target convention. Charged defects, vacancies, substitutions,
transition levels, migration barriers, finite-temperature free energies,
competing phases, and prospective DFT validation are outside the current
paper.
