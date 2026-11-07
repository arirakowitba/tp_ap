# Independent two-stage replication attempt — environment-blocked

Date: 2026-11-07
Main code commit: `732389017d7e20908328fe90800c06a4ae73d0b2`
Runner: `scripts/codex_replicate_native_nb_twostage_20261107.py`
Local source SHA-256: `457de3d6a7a4b271163aa0f91c5f3dd9d73d8d781971ee238f568ff018121dfd`

## Method implemented

The independent runner now enforces:

1. Outer fit/tune/confirm seller-aware split.
2. Seller-disjoint inner train/validation inside outer fit.
3. Stage 1:
   - all frequency/relative-price features fitted only from inner train;
   - nested text OOF generated only inside inner train;
   - inner validation projected from inner train;
   - CatBoost and LightGBM early stopping only on inner validation.
4. Stage 2:
   - iterations frozen from Stage 1;
   - nested text OOF recomputed across 100% outer fit with GroupKFold seller;
   - final components refitted on 100% outer fit without eval_set;
   - tune and confirm projected once.
5. Frozen recipes:
   - 50/50 native-text + nested-NB, threshold 0.50;
   - 80/20 nested-NB + LightGBM, diagnostic only.
6. Report includes method, explicit code SHA, inner/outer split hashes, frozen iterations, paired probabilities and seller bootstrap metrics.

The runner deliberately does not use public scores or test labels.

## Execution attempt

Command intended:

`python scripts/codex_replicate_native_nb_twostage_20261107.py <train_zip> <out> --seed 20261107 --deep --replicate-only --threads 4 --code-sha 457de3d...`

Execution stopped before data loading/model fitting:

`ModuleNotFoundError: No module named 'catboost'`

The current runtime has numpy, pandas, scipy and scikit-learn, but lacks both CatBoost and LightGBM. No metric or partial model was produced. This is an environment dependency blocker, not a negative experimental result.

## Next action

Antigravity can execute this pinned runner (or its equivalent post-`0184622` runner) in the environment that previously trained CatBoost/LightGBM. Required runs:
- mixed seed 20261107;
- pure seed 20261108.

Do not compare results unless the emitted report contains `method=two_stage_refit`, the exact code SHA and frozen iterations. No Kaggle action was taken.
