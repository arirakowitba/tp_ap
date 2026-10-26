# Round 54 — fresh seller split, frozen used→new hypothesis

Seed 55555 was fixed before inspecting labels. Seller-disjoint fit 42,457, tune 13,487, confirm 14,056. HistGB and HistGB + FastText (unsupervised original 100k texts) were retrained with OOF text teachers. CatBoost second stage was independently retrained on the same fit. Threshold .475 for Hist/FastText and .5 for CatBoost. Test labels and public scores were not used.

| Segment | Tune | Confirm |
|---|---:|---:|
| FastText vs HistGB, all rows | +5/13,487 | +20/14,056 |
| FastText used→new vs HistGB | −3/119 | +15/119 |
| FastText used→new vs CatBoost | +1/119 | +3/119 |

The new confirm keeps the positive sign against CatBoost, but only +3 in 119 changes; tune does not support the same strong effect against HistGB. No test flips or submission follow from this result. Prior validation splits overlap this one and are not independent replications.

Checkpoint: `round54_fresh_seller_checkpoint_20261026.zip`, Library ID `libfile_454ba4773eb08191b8bf035f0f2dc844`, SHA-256 `89276c60e2542fc29e9c9e4519a02c174a536c3748bea209e8bccd2adae65a37`. Includes code, paired probabilities and split IDs. Submission 41 remains observed public best .94628.
