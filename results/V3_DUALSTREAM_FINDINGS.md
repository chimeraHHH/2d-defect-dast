# v3 dual-stream cross-attention findings (2026-05-04 / 2026-05-05)

## Architecture

`DualStreamPeriodicTransformer` (`src/models/dualstream.py`):
- Shared PFA encoder runs over both defect and pristine supercells (same weights).
- 2 cross-attention blocks: defect tokens (Q) attend to pristine tokens (K, V),
  with a distance bias on the defect-pristine pair distance.
- Graph representation: Δz = pool(h_def_xattn) − pool(h_pri).
- Bias-free linear readout (1.142 M params total). Soft invariance loss
  λ * MSE(model(x_pri, x_pri), 0) with λ=0.05.

Pristine reference is constructed by removing the dopant atom from the defect
supercell (exact for IMP2D's adsorbate + interstitial defect types). 5/5
spot-check samples have cell_diff = pos_diff = 0.

## Results so far

### Phase 1: dualstream WITHOUT augmentation (baseline check)

- Run: `dualstream_h128_imp2d` (h=128, 60 ep, no aug, seed=42)
- **Test MAE: 0.5830 eV** (best_val 0.5684, RMSE 1.1343)
- Invariance loss: 0.138 → 0.002 over training (70× decay) — model learns
  the f(pristine, pristine) = 0 identity strongly.
- Wall: ~25 min on RTX 5090.

Apples-to-apples comparison (same training-set size, no aug):

| Model | Train | Params | Test MAE | Δ vs baseline |
|---|---|---|---|---|
| baseline_h128_long (no aug, h128, 60ep) | 8512 | 0.747 M | 0.622 | — |
| **dualstream_h128_imp2d (no aug, h128, 60ep)** | 8512 | 1.142 M | **0.5830** | **−6.3%** |

So the physical inductive bias of seeing both defect and pristine **does**
deliver a real (~6%) gain at the same data scale.

For reference, against the leak-free augmented (3×) v1 baseline:

| Model | Train | Params | Test MAE |
|---|---|---|---|
| baseline_h128_aug_long_safe (3× aug, h128, 50ep) | 25536 | 0.747 M | 0.516 |
| v2 single-source full (3× aug, h128, 50ep) | 25536 | 0.744 M | 0.519 |
| **dualstream_h128_imp2d (no aug, h128, 60ep)** | 8512 | 1.142 M | 0.583 |

The −6.3% gain from dual-stream is not yet enough to overtake the gain from
a 3× augmented training set with the v1 backbone. The natural next step is
to combine both: train dual-stream on a leak-free augmented (defect, pristine)
paired dataset.

### Phase 2: dualstream WITH leak-free joint augmentation (in progress)

`scripts/build_pristine_pairs_aug.py` constructs the joint-augmented
dataset:
- Each train sample produces three pairs: original, rotated, perturbed.
- Rotation R ∈ SO(2) is applied jointly to (defect_pos, defect_cell,
  pristine_pos, pristine_cell) — the same R for both streams.
- Gaussian perturbation σ=0.02 Å is applied per-atom; the noise tensor for
  the host atoms shared between defect and pristine is the SAME (so the
  defect's Δr_host = pristine's Δr_host exactly), while the dopant atom
  in defect gets its own independent noise.
- Output: `data/processed/aug_pristine_dataset_safe.pkl`,
  meta version `leak_free_aug_pristine_v1`,
  ordered split: 25536 train aug | 1064 val | 1065 test.

Training run: `configs/dualstream_h128_aug.yaml` (h=128, 50 ep, seed=42).
Result will land here once the run completes.

Expected outcome:
- Lower bound: 0.50 eV (matches dualstream gain on top of v2 PFA-only-aug).
- Target: ≤ 0.45 eV (below baseline_h128_aug_long_safe 0.516, achieving
  the publishable result we set out for in the differential-physics
  formulation).
- Stretch: ≤ 0.40 eV (would compete with v2 multi-source ID 0.4929).

## Open questions / next experiments

1. **4-seed verification** if Phase 2 result is competitive.
2. **Multi-source dual-stream**: extend the recipe to JARVIS-2D /
   JARVIS-3D defect sources (need pristine reconstruction for their
   vacancy structures), with a single-stream readout for DFT-3D
   pristine (no defect counterpart). Cross-attention layer would only
   receive gradient from genuine defect pairs.
3. **Interpretability**: the cross-attention weights provide a per-atom
   measure of "how much the defect prediction depends on each host
   atom". This can be visualised as a defect-locality kernel and
   compared to the v1.2 occlusion-attribution map.
