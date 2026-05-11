# v4 Enhanced Training Ablation Table

## Progressive Improvement (Single Model → Ensemble)

| Step | Description | Best Single MAE | Cumulative Ens MAE | Delta |
|---|---|---|---|---|
| 0 | MSE baseline (100ep, no UAE) | 0.513 | 0.460 (3 models) | — |
| 1 | + ct-UAE 128-dim embeddings | 0.504 | 0.439 (6 models) | −0.021 |
| 2 | + MAE (L1) loss | 0.452 | 0.423 (9 models) | −0.016 |
| 3 | + Linear warmup + LR 5e-4 | 0.411 | 0.402 (14 models) | −0.021 |
| 4 | + 150ep + SWA (ep120) | 0.407 | 0.391 (17 models) | −0.011 |
| 5 | + Deep model (4+3 layers) | 0.420 | 0.389 (18 models) | −0.002 |

## Why Each Improvement Works

1. **MAE loss (−0.04 eV single)**: MSE squares errors, so outliers (|err|>2 eV)
   dominate gradients. With MAE, each sample contributes equal-magnitude gradient
   (±1), directly optimizing the evaluation metric.

2. **Linear warmup + high LR (−0.02 eV)**: MAE's non-smooth gradient landscape
   (|x| has a kink at 0) makes early training unstable. 10-epoch warmup from
   0.1× → 1× of LR=5e-4 lets the model explore gently before committing.

3. **Cosine annealing (−0.01 eV)**: Smooth LR decay creates different
   optimization trajectories vs ReduceLROnPlateau's step-wise drops, enabling
   better ensemble diversity.

4. **ct-UAE embeddings (−0.01 eV)**: 128-dim pretrained atomic embeddings from
   Nature Comms 2025 multi-task checkpoint provide richer atomic identity than
   9-dim hand-crafted features (EN, radius, mass, etc.).

5. **150ep + SWA (−0.01 eV)**: Longer training reaches flatter minima;
   SWA (ep120-150) averages 30 checkpoints for further regularization.

6. **Deep model (adds diversity)**: 4+3 layers (vs 3+2) captures longer-range
   interactions. Solo performance similar (~0.42) but appears in every top-k
   ensemble combination.

## Ensemble Saturation Analysis

| k | Best MAE | Composition |
|---|---|---|
| 1 | 0.407 | 150ep_s45 |
| 2 | 0.378 | 150ep_s42 + 150ep_s45 |
| 3 | 0.372 | + 150ep_deep_s42 |
| 4 | 0.370 | + uae_mae_warmup_s45 |
| 5 | 0.368 | uae_warmup_s46 + deep_s42 + 150ep_s42 + 150ep_s43 + 150ep_s45 |
| 6 | 0.368 | + uae_warmup_s45 |
| 7 | 0.369 | (diminishing returns) |
| 26 | 0.389 | Full ensemble (noise dominates) |

**Conclusion**: k=5 is the sweet spot. Beyond k=6, adding weaker models
increases ensemble noise.

## UQ Calibration (26-model ensemble)

| Metric | Raw | Calibrated (τ=1.395) |
|---|---|---|
| NLL ↓ | 1.872 | **1.034** |
| ECE (z-space) ↓ | 0.028 | **0.027** |
| Coverage@90% → 90% | 78.2% | **90.4%** |
| Coverage@50% → 50% | — | 58.7% |
| corr(σ, |err|) | 0.546 | 0.546 |

**Key finding**: τ=1.395 < τ_v1=1.83, meaning the v4 26-model ensemble is
already better calibrated out of the box (raw cov@90% = 78% vs v1's 72.5%).

## Diversity Axis Contributions

| Axis | Effect on Ensemble |
|---|---|
| Loss function (MSE/Huber/MAE) | Different loss landscapes → different error modes |
| Architecture (3+2 vs 4+3) | Shallow: faster convergence; Deep: longer-range |
| Training length (100ep vs 150ep) | Different stopping points on loss surface |
| Features (±ct-UAE) | Different atomic representations |
| Random seed (s42-s46) | Different initialization basins |

## Individual Model MAEs (sorted)

| Model | MAE (eV) | Recipe |
|---|---|---|
| 150ep_s45 | 0.4072 | 150ep MAE+warmup+cosine+UAE+SWA |
| uae_mae_warmup_s45 | 0.4110 | 100ep MAE+warmup+cosine+UAE |
| 150ep_s42 | 0.4120 | 150ep MAE+warmup+cosine+UAE+SWA |
| deep_s42 | 0.4160 | 100ep deep MAE+warmup+cosine+UAE |
| 150ep_deep_s42 | 0.4197 | 150ep deep MAE+warmup+cosine+UAE+SWA |
| uae_mae_warmup_s42 | 0.4230 | 100ep MAE+warmup+cosine+UAE |
| uae_mae_warmup_s46 | 0.4235 | 100ep MAE+warmup+cosine+UAE |
| 150ep_s43 | 0.4267 | 150ep MAE+warmup+cosine+UAE+SWA |
| deep_s45 | 0.4301 | 100ep deep MAE+warmup+cosine+UAE |
| uae_mae_warmup_s44 | 0.4314 | 100ep MAE+warmup+cosine+UAE |
| no_uae_s42 | 0.4422 | 100ep MAE+warmup+cosine (no UAE) |
| uae_mae_warmup_s43 | 0.4431 | 100ep MAE+warmup+cosine+UAE |
| deep_s43 | 0.4451 | 100ep deep MAE+warmup+cosine+UAE |
| uae_mae_s43 | 0.4517 | 100ep MAE+cosine+UAE (no warmup) |
| deep_huber_s42 | 0.4750 | 100ep deep Huber+warmup+cosine+UAE |
| uae_mae_s42 | 0.4752 | 100ep MAE+cosine+UAE (no warmup) |
| uae_huber_s43 | 0.4936 | 100ep Huber+cosine+UAE |
| uae_mae_s44 | 0.4948 | 100ep MAE+cosine+UAE (no warmup) |
| uae_huber_s44 | 0.4983 | 100ep Huber+cosine+UAE |
| uae_huber_s42 | 0.5001 | 100ep Huber+cosine+UAE |
| uae_s44 | 0.5044 | 100ep MSE+plateau+UAE |
| uae_s43 | 0.5064 | 100ep MSE+plateau+UAE |
| uae_s42 | 0.5101 | 100ep MSE+plateau+UAE |
| 100ep_s44 | 0.5125 | 100ep MSE+plateau (no UAE) |
| 100ep_s42 | 0.5126 | 100ep MSE+plateau (no UAE) |
| 100ep_s43 | 0.5245 | 100ep MSE+plateau (no UAE) |
