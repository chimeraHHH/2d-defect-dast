# Cover letter draft — Physical Review B

> Author action required before submission: replace every bracketed field,
> verify the final title and journal section, and remove this note.

[DATE]

Dear Editors,

We submit the manuscript “Learning impurity incorporation energetics across
host and impurity chemistry in two-dimensional materials with a defect-aware
graph Transformer” for consideration as a Regular Article in *Physical Review
B*. We suggest B15(4), “Surface physics, nanoscale physics, low-dimensional
systems,” as the primary section and B1(1), “Structure, structural phase
transitions, mechanical properties, defects,” as a secondary section.

Formation-energy models are often assessed with random structure splits, even
though retrospective database analysis spans chemically distinct forms of transfer. Our
study resolves those forms for 10,224 neutral impurity structures from IMP2D.
Except for one singleton-host structure, its pair-held-out folds recombine
hosts and impurities represented elsewhere in training and are much easier
than transfer to a host or impurity absent from target supervision. Under the
same pair-held-out evaluation, interstitial incorporation energies are more
difficult to predict than adsorbate energies. These observations define a chemically and geometrically
resolved applicability domain for energy estimation in two-dimensional
materials.

The Defect-Aware Radial Transformer (DART) supplies the modeling framework. A
paired factorial isolates small, repeat-consistent contributions from gated
graph readout and a composite local interaction block. A controlled SchNet
readout analysis further connects model design to the target physics: atomwise
addition, natural for extensive total energies, contributes to the large
host-held-out error for this evaluated recipe. Finally,
pair-held-out predictions recover the lower-energy incorporation
class with 91.0% accuracy and quantify site-selection regret.

The claims are deliberately tied to the available evidence. Energy estimation
and reranking are retrospective and conditional on the DFT-relaxed candidate
geometries released by IMP2D. Because those geometries and their reference
energies arise from the same completed DFT workflow, the model does not replace
structure generation or relaxation or claim to reduce their cost.
Uncertainty results are reported as internal calibration rather than a
guarantee under chemical shift. No new external first-principles validation or
prospective materials discovery is claimed.

[AUTHOR CONFIRM: This manuscript is original, is not under consideration
elsewhere, and its submission history with Physical Review is as follows: ...]

[AUTHOR INSERT: final public data/software release, persistent identifier,
license, and repository citation.]

[AUTHOR CONFIRM: conflict-of-interest statement.]

[OPTIONAL AUTHOR INPUT: suggested or excluded referees and reasons.]

Thank you for considering this manuscript.

Sincerely,

[CORRESPONDING AUTHOR NAME]

[AFFILIATION]

[EMAIL]
