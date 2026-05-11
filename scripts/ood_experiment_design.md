# Constrained OOD Experiment Design

## Motivation

Current results: ID MAE = 0.36 eV, full OOD (prospective DFT) = 2.66 eV.
Goal: show graduated generalization capability in between.

## IMP2D Data Summary

- 10641 samples (converged, |Ef|<20 eV)
- 44 hosts × 65 dopants, matrix 91.2% filled
- Average 4.1 samples per (host, dopant) cell
- Defect types: adsorbate + interstitial only

## Host Families

| Family | Hosts | N samples | Mean Ef | Intra-ρ |
|--------|-------|-----------|---------|---------|
| Group-6 TMD | MoS2, MoSe2, MoTe2, WS2, WSe2, WTe2, MoSSe | 2183 | 4.14 | 0.589 |
| Group-4/5 TMD | NbS2, NbSe2, TaS2, TaSe2, TiS2, ZrS2, ZrSe2, HfS2, HfSe2 | 2542 | 2.15 | 0.631 |
| Pt-group | PtS2, PtSe2, NiSe2, SnS2, SnSe2, Pd2Se4 | 1426 | 2.05 | 0.618 |
| MXene | V2CO2, Mo2CO2, Nb2CO2, Nb4C3 | 803 | 2.29 | 0.527 |
| Halide | Cr2I6, PbI2, BiITe, Bi2I6 | 512 | 1.97 | 0.721 |
| Re-based | Re4S8, Re4Se8 | 505 | 2.87 | 0.917 |
| Elemental/Other | As2, Si2, Sn2, Ge2, ... | 2670 | 2.22 | 0.725 |

## Dopant Groups

| Group | Elements | N samples | Intra-ρ | Cross with others |
|-------|----------|-----------|---------|-------------------|
| 3d TM | Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn | 1732 | 0.671 | 0.58-0.62 |
| 4d TM | Y,Zr,Nb,Mo,Ru,Rh,Pd,Ag | 1268 | 0.636 | 0.56-0.64 |
| 5d TM | Hf,Ta,W,Re,Os,Ir,Pt,Au | 1217 | 0.656 | 0.48-0.64 |
| Alkali/AE | Li,Na,K,Be,Mg,Ca,Sr,Ba | 1354 | 0.656 | 0.47-0.59 |
| p-block light | B,C,N,O,F,Si,P,S,Cl | 1828 | 0.643 | 0.47-0.60 |
| p-block heavy | Ga,Ge,As,Se,...,Bi | 1535 | 0.323 | 0.31-0.38 |

## Recommended Experiments (Priority Order)

### Experiment 1: Leave-One-G6-Host-Out (7-fold CV)

**Scientific question**: Can the model predict defect Ef in a new TMD given
knowledge of 6 sibling TMDs + 37 other hosts?

**Setup**: 7 folds, each holds out one Group-6 TMD
- Train: ~10,300 samples (all except held-out host)
- Test: ~260-460 samples (held-out host, all dopants)
- Baseline: mean of siblings' Ef per dopant (expected MAE ~1.1-1.7 eV)
- Model should significantly beat this baseline

**Why this is ideal**:
- Largest host family with moderate intra-ρ (0.589) → meaningful but not impossible
- Practically relevant (experimentalists work within TMD families)
- MoS2 is hardest (|Δ|=1.69 eV), MoSSe easiest (|Δ|=1.12 eV) → shows gradient
- 59/65 dopants common to all 7 hosts → clean evaluation

**Expected outcome**: MAE 0.5-0.9 eV (model > siblings-mean baseline)

### Experiment 2: Compositional Block-Out (Matrix Completion)

**Scientific question**: Can the model infer G6-TMD × 3d-TM combinations
from (G6 × other-dopants) and (other-hosts × 3d-TM) separately?

**Setup**:
- Hold out: all G6-host + 3d-dopant samples (372 samples)
- Train sees: G6 with non-3d dopants (1811), non-G6 with 3d dopants (1360)
- This tests "matrix completion" ability

**Why this is interesting**:
- Directly tests compositional generalization (the ML holy grail for materials)
- Analogous to collaborative filtering in recommender systems
- ct-UAE embeddings should help (encode atomic identity → compositional transfer)
- Clean ablation: model with vs without ct-UAE

**Expected outcome**: MAE 0.7-1.2 eV

### Experiment 3: Leave-One-Dopant-Group-Out

**Scientific question**: Can the model generalize to entirely unseen
element types (e.g., predict 3d TM behavior from 4d/5d/alkali/p-block)?

**Setup**: 5 folds (one per dopant group)
- Hold out: all samples with dopants in that group
- Most interesting: hold out 3d TM (1732 samples, ρ_cross ≈ 0.6)
- Hardest: hold out p-heavy (1535 samples, ρ_cross ≈ 0.33)

**Why this matters**:
- Tests whether ct-UAE captures periodic table relationships
- Multi-source training exposes model to more element types → should help here
- If Fe can be predicted from Ru (4d analog), that validates chemical intuition

**Expected outcome**: MAE 1.0-2.0 eV (3d TM easier; p-heavy much harder)

### Experiment 4: Leave-One-Family-Out (Cross-family)

**Scientific question**: Can Group-4/5 TMD knowledge help predict Group-6 TMDs?

**Setup**:
- Hold out: all G6-TMD samples (2183)
- Train on: 8458 samples (G45-TMD, Pt-group, MXene, Halide, etc.)
- Cross-family ρ only 0.44 → this is HARD

**Expected outcome**: MAE 1.5-2.5 eV (approaching full-OOD difficulty)

## Implementation Plan

1. Modify `make_splits()` to accept leave-out specification
2. Train best v4 recipe (150ep, MAE+warmup+cosine+UAE+SWA) on each split
3. Also train multi-source version to compare (does 4-DB help OOD?)
4. Compare: single-source vs multi-source at each OOD tier

## Compute Budget

- Exp 1 (7 folds × 1 model each): 7 × ~20 min = 2.3 GPU-hours
- Exp 2 (1 fold): ~20 min
- Exp 3 (5 folds): 5 × ~20 min = 1.7 GPU-hours  
- Exp 4 (1 fold): ~15 min (less training data)
- Total: ~5 GPU-hours (single L40S), plus multi-source variants

## Key Ablations

For each experiment, compare:
1. v4 recipe (single-source, best recipe)
2. v4 + multi-source (does 4-DB joint training help OOD?)
3. v4 without ct-UAE (does pretrained embedding help transfer?)
4. Naive baseline: predict mean of nearest-neighbor hosts/dopants

## Paper Narrative

```
Table: Graduated Generalization Assessment

| Tier | Scenario | MAE (eV) | Description |
|------|----------|----------|-------------|
| 0 | ID (random split) | 0.36 | Same distribution |
| 1 | Intra-family OOD | 0.X | New host in same TMD family |  
| 2 | Compositional OOD | 0.X | New (host,dopant) combination |
| 3 | Dopant-group OOD | X.X | Entirely new element type |
| 4 | Cross-family OOD | X.X | Entirely new host family |
| 5 | Full OOD (DFT) | 2.66 | Novel materials (prospective) |
```

This shows the model degrades gracefully rather than catastrophically,
and identifies the specific chemical knowledge gaps.
