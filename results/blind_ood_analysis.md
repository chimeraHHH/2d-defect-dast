# Blind cross-code OOD benchmark - statistical analysis

38 usable GPAW samples (graphene 16 / VS2 11 / CrS2 11), 4-seed ensemble, tau = 2.595.

## 1. Raw (uncorrected) performance

- MAE = **2.56 eV** (95% CI 2.12-2.98)
- bias = **+2.15 eV** (CI +1.50-+2.75) - systematic overprediction, consistent with cross-code / chemical-potential reference shift
- Pearson r = 0.50 (CI 0.23-0.69), Spearman rho = 0.45

## 2. Per-host constant-offset calibration

| host | n | bias (eV) | raw MAE | MAE oracle-offset | MAE LOO-offset | 3-shot MAE |
|---|---|---|---|---|---|---|
| graphene | 16 | +1.50 | 2.10 | 1.53 | 1.63 | 1.80±0.34 |
| VS2 | 11 | +3.45 | 3.45 | 0.93 | 1.02 | 1.14±0.27 |
| CrS2 | 11 | +1.79 | 2.33 | 1.77 | 1.95 | 2.07±0.41 |

- Pooled LOO-offset MAE = **1.55 eV** (CI 1.20-1.90) - a single per-host scalar (estimable from ~3 DFT calculations) removes most of the OOD gap, cf. LOHO MAE 1.4-2.1 eV in Sec 5.7.

## 3. UQ reliability under blind OOD

- mean sigma_cal = **3.77 eV** vs 0.87 eV in-distribution -> **4.3x self-aware amplification**
- empirical coverage (raw errors): 68% nominal -> 76%, 90% -> 100%, 95% -> 100% (conservative, no under-coverage even with the systematic shift included)
- coverage after per-host LOO debias: 68% -> 89%, 90% -> 97%
- corr(sigma_cal, |err|) = +0.27

## 4. Sample #81 anomaly corroboration

After removing the CrS2 host offset the model expects ~2.29 eV, i.e. the DFT value 5.28 eV is ~3 eV higher than the ML expectation -- independent corroboration of the suspected SCF high-energy state flagged in the raw data notes.

## Takeaways for the paper

1. Blind, cross-code transfer (different DFT code, different mu convention, unseen hosts) costs ~5x in raw MAE - but the error is dominated by a per-host constant.
2. A few-shot (~3 DFT) offset calibration recovers ~1.5 eV MAE, in line with the LOHO analysis.
3. Temperature-scaled ensemble UQ remains conservative and informative under this hardest OOD setting - the model 'knows it does not know'.
4. The UQ + host-offset view independently flags the one sample (#81) that the DFT-side notes had marked as a suspected SCF failure.