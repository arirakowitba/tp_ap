# Round 53 — unlabeled covariate-shift weighting (2026-10-26)

## Hypothesis and protocol

Weight the training rows by similarity to the unlabeled seller-disjoint holdout in numeric listing features and unsupervised FastText embeddings. A regularized logistic domain classifier estimated fit-vs-holdout odds, clipped to [0.25, 4] and mean-normalized. The supervised HistGB second stage retained the round39 3-fold seller-group OOF text teachers and original 100k text source. No holdout or Kaggle labels entered the weighting model. The unweighted reproduction matched the saved round39 FastText predictions exactly (<1e-8).

Each split has disjoint fit/tune/confirm sellers. The three split realizations overlap one another and must not be interpreted as independent trials. Decisions use the existing 0.475 threshold. Confirm was not used to choose a weighting variant.

| Split | Tune net / n | Confirm fixes/breaks/net / n | Confirm log loss baseline → weighted | Seller bootstrap 95% net |
|---|---:|---:|---:|---:|
| first | −12 / 13,787 | 73/55/+18 / 14,533 | .210628 → .210271 | [−5,+41] |
| confirm2 | +5 / 14,335 | 68/73/−5 / 14,490 | .205913 → .205585 | [−29,+20] |
| frozen42 | +10 / 14,123 | 64/77/−13 / 14,648 | .215678 → .215778 | [−37,+11] |

All holdout sellers are unseen in the corresponding fit. Apparent domain AUCs were .550/.560/.566; the discriminator found modest distribution differences. Confirm decision changes were 128/141/141. Directional net: used→new +13/+9/−14, new→used +5/−14/+1. These directions are inconsistent as well.

## Decision

Do not promote the density-ratio weighting for accuracy or test flips. The first two confirms slightly improve log loss but the third worsens, and accuracy reverses sign. The three seller bootstrap intervals each include zero. No Kaggle submission was made. Public scores were not used as per-row labels.

Reproducible checkpoint in Library: `round53_domain_weight_checkpoint_20261026.zip`, ID `libfile_579c24803be48191a590d560075d0272`, SHA-256 `ed657e597d785073d87f176001862ce1142417702437ffefe5c6d193473f6de1`. Includes both scripts, complete paired fit/hold/tune/confirm predictions and train weights for three splits, JSON metrics and file hashes. Raw 70k labels and original 100k texts remain in the prior project data and are not copied into this checkpoint.

Next direction: investigate a stronger complementary representation rather than further thresholding or reweighting this model. Keep the exact submission 41 (public .94628) as the observed best.
