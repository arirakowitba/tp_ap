# Round60 — category-relative price, third seller cut (2026-10-26)

Fixed round58 six fit-only unlabeled price distribution features with the cached quantile implementation, 151 inputs, original100k FastText embeddings and seller-group OOF text predictors. Existing third split fit41229/tune14123/confirm14648, sellers disjoint within split; cuts overlap and have prior reuse. Thresholds .475 candidate/HistGB+FastText and .5 CatBoost.

| Section | Fixes/breaks/net vs HistGB+FT | Seller bootstrap95 net | Log loss baseline → candidate | Net vs CatBoost (95 interval) |
|---|---:|---:|---:|---:|
| Tune |74/76/−2|[−25,+21]|.222172 → .223201|+3 [−35,+41]|
| Confirm |68/74/−6|[−28,+17]|.215678 → .216405|+29 [−7,+66]|

Across three confirms, nets versus HistGB+FastText are +28,+1,−6. The third cut also reverses the log-loss improvement. Do not promote the fixed category-price variant or select test flips from it. Relative gains against CatBoost reflect reference choice and do not prove improvement against exact submission41. Bootstrap2000 seller resamples seed6001. No Kaggle quota used.

Checkpoint `round60_category_price_third_checkpoint_20261026.zip`, Library ID `libfile_65cabc8fd37c8191bccc9bd9ec8d58b7`, SHA256 `6091d616ec1e9e4ef7f2516c418f274777b10ac271cab7d7ff9e94238723fde6`; code, complete paired probabilities/split IDs, metrics, hashes. This finishes the planned third-cut check. Further variants require a distinct hypothesis and a new evaluation protocol.
