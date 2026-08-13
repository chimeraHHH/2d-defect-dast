# PRM v19 visual contract

Target: PRB Regular Article, two-column main text.

| Main figure | Core claim | Source | v19 decision |
| --- | --- | --- | --- |
| Fig. 1 | DART combines impurity-local enrichment, local geometry, global attention, and gated non-extensive readout. | Manuscript methods and implementation | Replace the user schematic with the author-directed GPT Image 2 edit; no quantitative content. |
| Fig. 2 | Adsorbate and interstitial structures are alternative single-impurity configurations. | IMP2D task definition | Replace the overlaid schematic with two author-directed GPT Image 2 panels distinguished by color and shape. |
| Fig. 3 | Pair-held-out predictions retain useful accuracy, whereas missing constituent supervision causes a sharp error increase. | Repaired pair OOF predictions and repaired transfer metrics | Merge parity and transfer hierarchy in one deterministic figure. |
| Fig. 4 | Adsorbate and interstitial errors have different geometry/chemistry associations. | Canonical G3 physical analysis | Retain unchanged. |
| Fig. 5 | Available DFT-relaxed candidates can be retrospectively reranked. | Repaired pair OOF reranking outputs | Retain unchanged; do not imply prospective screening. |
| Fig. 6 | Atom-order defects can move individual predictions while leaving pooled error nearly unchanged. | Frozen permutation diagnostic | Retain unchanged. |

The protocol overview, benchmark table, periodic-table map, parity-only plot,
family/XF summaries, and illustrative structure cases are retained in the
Supplemental Material. The case gallery remains illustrative because its
main-text case gate is not satisfied.

Image generation/editing: only Figs. 1 and 2 use OpenAI GPT Image 2. The
authors supplied the source images, specified every scientific label and
connection, and inspected the outputs. All quantitative figures remain
deterministically rendered from archived source data. No manuscript text,
result table, prediction archive, or author identity was sent to the image
model.
