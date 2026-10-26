# Round58 — category-relative price (2026-10-26)

Hypothesis: price position within a comparable listing group supplies signal for unseen sellers. Added six label-free fit-only features to the 145 HistGB+FastText baseline inputs: log-price residual from median, residual divided by max(.1,IQR), and empirical price CDF, for category/currency and category/listing-type/currency. Groups with fewer than20 fit rows fall back to the global fit distribution. No holdout labels used in feature construction. Fit rows include themselves in these unlabeled summaries; evaluate the resulting train/holdout feature mismatch before promoting.

Existing seller-disjoint fit41680/tune13787/confirm14533; this split has been reused and is not pristine independent evidence. Threshold .475 fixed. Original100k text embeddings and OOF teachers retained.

| Section | Fixes/breaks/net | Seller bootstrap 95% net | Log loss baseline → candidate |
|---|---:|---:|---:|
| Tune | 56/60/−4 | [−28,+19] | .204014 → .203858 |
| Confirm | 77/49/+28 | [+2,+54] | .210628 → .209452 |

Promising on confirm, mixed between sections; do not submit yet. Need a second seller split and relevant CatBoost proxy comparison with thresholds declared correctly. Bootstrap uses3000 seller resamples, seed5801. Confirm reuse limits its significance.

Checkpoint `round58_category_price_checkpoint_20261026.zip`, Library ID `libfile_9510c7edda3881919e5a7e779c273845`, SHA256 `e45bd6a41f96083f4e631b7cdd47f17bc79780740aa813a4dd8ca9e36c2c2e82`; code, pairs, metrics, hashes. Cached-quantile implementation included for efficiency but its equivalence was not yet verified at checkpoint time. No Kaggle quota used.
