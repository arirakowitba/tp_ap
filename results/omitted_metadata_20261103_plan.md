# Omitted listing metadata ablation — predeclared plan

Before training, inspect original 70k train and 30k test schema. The current 101-feature historical extractor omits listing `status`, description presence/count, and secure-thumbnail missingness. On train, `status=paused` occurs in 2,952 rows with a used rate different from active; description absence occurs in 1,860 rows. These descriptive rates are hypothesis generation only.

Predeclare seller-group seeds 20261134 and 20261135. Compare the exact strong word(1,2)+char_wb(3,5) TF-IDF OOF + CatBoost seller categorical baseline with an otherwise identical model adding: `status` categorical; `has_description`, `n_descriptions`, `secure_thumb_missing` numeric. All features use only each row's original input and no public scores or test labels. The feature bundle is fixed before either confirm is evaluated. Both arms share fit/tune/confirm, text OOF, supervised TE, CatBoost hyperparameters and tune-only early stopping/threshold selection. Report paired confirm accuracy and log loss, fixes/breaks, seen/new, seller bootstrap, dedup across splits, hashes and private row-level pairs. Do not promote based on an unconfirmed gain or a single cut.

Code: `scripts/ablate_omitted_metadata_20261103.py`. Input ZIP SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`.
