# W1 precondition: SchNet pipeline-independence audit

Audit date: 2026-08-12
Auditor scope: the frozen precondition in
`paper_Q1/review/w1_extensivity_preregistration.md` — before any canonical
SchNet panel, confirm the SchNet input pipeline shares neither of the two
DART graph-builder defects (the first-32 ordered-neighbour-pair triplet cap
and the componentwise-fractional-wrap dense radial matrix).

## Findings

1. `src/models/schnet_pbc.py` consumes only `edge_index_list` and
   `edge_dist_list` — the stored periodic neighbour-list edges. A repository
   search finds **zero references** to `dist_matrix`, `triplet_index`, or
   `angles` in `src/models/schnet_pbc.py` and `src/train_schnet.py`.
2. The stored edges themselves are ASE `neighbor_list` output whose topology
   was independently reconstructed for all 9,871,460 edges during protocol
   v2 and re-verified during the G1 rebuild (topology exact, distances
   within float32 storage noise). Neither DART defect lives in the edge
   list: the triplet cap acts on DART's angular branch, and the wrapped
   distances feed DART's global radial bias, both absent from the SchNet
   forward pass.
3. The shared `collate_fn` packs graph fields for both models, but packing
   is not consumption: the SchNet module's forward path reads only the edge
   fields listed above.

## Verdict

**Independent.** The archived SchNet predictions (the 48-run campaign and
the 15-run mean-readout sensitivity arm) are unaffected by the two DART
implementation defects, so SchNet panels of the W1 extensivity diagnostic
are eligible for the canonical tier. The DART panel uses the repaired G2
OOF predictions under the accepted `prm_g2_acceptance_v1` chain.

This audit is committed before any W1 outcome is computed, as the frozen
preregistration requires.
