# Cover letter draft — Physical Review Materials

> Author action required before submission: replace every bracketed field,
> verify the final title, and remove this note.

[DATE]

Dear Editors,

We submit the manuscript “Defect-aware graph transformer for impurity
formation energies in two-dimensional materials: Chemical transferability and
error analysis” for consideration in *Physical Review Materials*. The work
sits at the journal's materials-informatics core and is a natural fit for the
scope represented by the Machine Learning for Materials Discovery and
Understanding collection.

Formation-energy models are often assessed with random structure splits, even
though retrospective database analysis spans chemically distinct forms of
transfer. Our study resolves those forms for 10,224 neutral impurity
structures from IMP2D with the Defect-Aware Radial Transformer (DART), an
impurity-centered graph model whose permutation invariance and exact
minimum-image distances passed an independent acceptance audit. Error grows
only mildly from random interpolation (0.408 eV) through recombination of
familiar host--impurity pairs (0.442 eV) and a predefined chemistry block
(0.483 eV), but roughly doubles for a withheld impurity (0.845 eV) or host
(0.891 eV); a preregistered analysis attributes the residual host--impurity
asymmetry to the structural novelty of the held-out host after controlling
for training-support asymmetry.

The error analysis then separates the physics of the two incorporation
classes --- interstitial error tracks local crowding while adsorbate error
tracks chemical mismatch --- and a readout analysis connects model design to
the non-extensive, defect-local target: atomwise addition, natural for
extensive total energies, degrades withheld-host error to 6.7 eV, a mean
readout alone recovers most of that gap, and the defect-centered readout
shows no size-dependent residual drift. Pair-held-out predictions recover the
lower-energy incorporation class with 91.4% accuracy against a 66.2%
class-prior reference and quantify site-selection regret. A methodological
finding accompanies the physics: a permutation diagnostic of a superseded,
order-sensitive implementation moved single predictions by up to 8.8 eV while
pooled errors moved by under 1e-3 eV, showing that aggregate benchmarks can
hide representation defects that dominate individual predictions.

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
