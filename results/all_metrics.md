# All metrics — automated aggregation

## per-run

| run | model | data | params (M) | epochs | seed | Test MAE | Test RMSE |
|---|---|---|---|---|---|---|---|
| baseline_h128_aug_long | baseline | augmented_dataset.pkl | 0.747 | 50 | 42 | 0.2058 | 0.31 |
| baseline_h128_aug_long_seed3 | baseline | augmented_dataset.pkl | 0.747 | 50 | 3 | 0.2393 | 0.3904 |
| baseline_h128_aug_long_seed1 | baseline | augmented_dataset.pkl | 0.747 | 50 | 1 | 0.2604 | 0.497 |
| baseline_h128_aug_long_seed2 | baseline | augmented_dataset.pkl | 0.747 | 50 | 2 | 0.2965 | 0.4281 |
| baseline_h128_aug_long_seed0 | baseline | augmented_dataset.pkl | 0.747 | 50 | 0 | 0.3172 | 0.4731 |
| baseline_aug_long | baseline | augmented_dataset.pkl | 0.198 | 50 | 42 | 0.4156 | 0.5955 |
| baseline_aug_seed2 | baseline | augmented_dataset.pkl | 0.198 | 30 | 2 | 0.4761 | 0.7158 |
| baseline_h128_aug_xlong_safe | baseline | augmented_dataset_safe.pkl | 0.747 | 100 | 42 | 0.478 | 1.1458 |
| baseline_h128_aug_xlong_safe_seed0 | baseline | augmented_dataset_safe.pkl | 0.747 | 100 | 0 | 0.5036 | 1.1637 |
| baseline_aug | baseline | augmented_dataset.pkl | 0.198 | 30 | 42 | 0.5112 | 0.7227 |
| baseline_aug_seed3 | baseline | augmented_dataset.pkl | 0.198 | 30 | 3 | 0.5131 | 0.7632 |
| dast_dense_aug | improved | augmented_dataset.pkl | 0.202 | 30 | 42 | 0.515 | 0.7444 |
| baseline_h128_aug_long_safe | baseline | augmented_dataset_safe.pkl | 0.747 | 50 | 42 | 0.5161 | 1.1314 |
| baseline_h128_aug_long_safe_seed0 | baseline | augmented_dataset_safe.pkl | 0.747 | 50 | 0 | 0.5332 | 1.1701 |
| baseline_aug_seed1 | baseline | augmented_dataset.pkl | 0.198 | 30 | 1 | 0.545 | 0.8611 |
| baseline_h128_aug_long_safe_seed1 | baseline | augmented_dataset_safe.pkl | 0.747 | 50 | 1 | 0.545 | 1.2019 |
| baseline_h128_aug_long_safe_seed2 | baseline | augmented_dataset_safe.pkl | 0.747 | 50 | 2 | 0.5517 | 1.1735 |
| baseline_h128_long | baseline | cleaned_dataset.pkl | 0.747 | 60 | 42 | 0.6221 | 1.1667 |
| baseline_aug_long_safe | baseline | augmented_dataset_safe.pkl | 0.198 | 50 | 42 | 0.6277 | 1.2516 |
| baseline_aug_safe | baseline | augmented_dataset_safe.pkl | 0.198 | 30 | 42 | 0.6718 | 1.2461 |
| baseline_h192_aug_long_safe | baseline | augmented_dataset_safe.pkl | 2.338 | 60 | 42 | 0.6736 | 1.291 |
| baseline_aug_seed0 | baseline | augmented_dataset.pkl | 0.198 | 30 | 0 | 0.6815 | 0.961 |
| baseline_long | baseline | cleaned_dataset.pkl | 0.198 | 60 | 42 | 0.7371 | 1.3285 |
| baseline | baseline | cleaned_dataset.pkl | 0.198 | 30 | 42 | 0.8621 | 1.5216 |
| ablate_local_only | baseline | cleaned_dataset.pkl | 0.093 | 30 | 42 | 1.3973 | 2.0274 |
| baseline_h128_aug | baseline | augmented_dataset.pkl | 1.06 | 30 | 42 | 1.4734 | 1.9917 |
| dast_dense | improved | cleaned_dataset.pkl | 0.202 | 30 | 42 | 1.4857 | 2.1149 |
| ablate_no_lattice | improved | cleaned_dataset.pkl | 0.198 | 30 | 42 | 1.6792 | 2.3537 |
| ablate_no_virtual | improved | cleaned_dataset.pkl | 0.202 | 30 | 42 | 1.6826 | 2.3339 |
| baseline_h128 | baseline | cleaned_dataset.pkl | 1.06 | 30 | 42 | 1.8156 | 2.506 |
| improved | improved | cleaned_dataset.pkl | 0.202 | 30 | 42 | 1.8269 | 2.5538 |

## multi-seed mean ± std

| run | model | data | params (M) | epochs | seed | Test MAE | Test RMSE |
|---|---|---|---|---|---|---|---|
| baseline_h128_aug_xlong_safe  (2 seeds) | baseline | augmented_dataset_safe.pkl | 0.747 | 100 | mean | 0.4908 ± 0.0128 | 1.1547 ± 0.0090 |
| baseline_h128_aug_long_safe  (4 seeds) | baseline | augmented_dataset_safe.pkl | 0.747 | 50 | mean | 0.5365 ± 0.0135 | 1.1692 ± 0.0251 |

## ensemble (raw)

| run | model | data | params (M) | epochs | seed | Test MAE | Test RMSE |
|---|---|---|---|---|---|---|---|
| deep ensemble (4 long + 2 xlong) | ensemble | leak_free_v1 | n×0.747 | — | — | 0.4428 | 1.0943 |
| deep ensemble (4 long seeds) | ensemble | leak_free_v1 | n×0.747 | — | — | 0.4644 | 1.1016 |

## ensemble (τ=2.60, eval-half)

| run | model | data | params (M) | epochs | seed | Test MAE | Test RMSE |
|---|---|---|---|---|---|---|---|
| deep ensemble (4 long seeds) | ensemble + τ | leak_free_v1 | n×0.747 | — | — | 0.4834 | 1.2014 |

## ensemble (τ=1.83, eval-half)

| run | model | data | params (M) | epochs | seed | Test MAE | Test RMSE |
|---|---|---|---|---|---|---|---|
| deep ensemble (4 long + 2 xlong) | ensemble + τ | leak_free_v1 | n×0.747 | — | — | 0.4582 | 1.192 |
