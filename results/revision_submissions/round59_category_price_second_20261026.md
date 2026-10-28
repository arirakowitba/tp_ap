# Round59 — category price, second seller split (2026-10-26)

Fixed six fit-only unlabeled category price features from round58, same151 inputs and HistGB settings. Used the existing second seller split: fit41175/tune14335/confirm14490. Threshold .475 for candidate and HistGB+FastText; CatBoost reference .5. Original100k text embeddings retained. Cuts overlap earlier cuts and have prior reuse; not independent pristine confirmations.

| Section | Fixes/breaks/net vs HistGB+FT | Seller bootstrap95 net | Log loss baseline → candidate | Net vs CatBoost (95 interval) |
|---|---:|---:|---:|---:|
| Tune |82/73/+9|[−18,+36]|.203370 → .202312|+26 [−17,+67]|
| Confirm |68/67/+1|[−23,+27]|.205913 → .205384|+19 [−20,+57]|

Together with round58, all four tune/confirm log losses improve, but accuracy gains are mixed/weak (−4,+28,+9,+1). Do not prepare submission. The quantile-cached implementation was used on this split; equivalence to the first split's uncached predictions has not been established. Fit rows include themselves in unlabeled summaries; assess leave-one-out or cross-fit representation consistency before promotion.

Checkpoint `round59_category_price_second_checkpoint_20261026.zip`, Library ID `libfile_6445427711048191b24a1b6757d03c27`, SHA256 `508f7ce7a3919f7e50b67baa92a95d57b86a2972defc510dca8c5ddf253ce218`. Includes code, pairs, metrics, hashes. Bootstrap2000 seller resamples seed5901. No Kaggle quota used.
