# OOF feature transport to inference (2026-11-03)

Predeclared plan `results/oof_feature_transport_20261103_plan.md`; code `scripts/ablate_oof_feature_transport_20261103.py`. Original train70k ZIP SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`; `used=1`. No public scores or test labels used.

Both CatBoost arms train on **identical fit rows and identical feature values**. The three supervised target encodings and word+char TF-IDF text probability are generated OOF on fit with seller GroupKFold(4). For tune/confirm, the historical arm applies a full-fit text logistic regression and full-fit TE maps; the candidate averages the four fold submodels/maps. Each arm chooses early stopping and accuracy threshold on tune only, then evaluates confirm once. The identical fit features led to identical best iterations within each seed. Both use the same 101 structural features, including the historical transductive feature context.

| Seed | Confirm N | Full-fit accuracy | Fold-average accuracy | Fold-average net | Fixes / breaks | Full-fit / fold-average log loss | Seller bootstrap 95% CI | Net at shared .500 threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261139 | 12,296 | .953806 | .953481 | -4 | 22 / 26 | .127754 / .127321 | [-18, +10] | -4 |
| 20261140 | 12,147 | .948959 | .950029 | +13 | 29 / 16 | .129275 / .129357 | [-2, +29] | +16 |

The confirm sets share 2,600 IDs. Of 24,443 rows, 21,843 IDs are unique. Deduplicating with seed39 priority yields net -1, seller-bootstrap CI [-21, +18]; seed40 priority yields +12, CI [-8, +31]. At a shared .500 threshold the deduplicated net is +8 or +13. The candidate improves log loss in one cut and worsens it in the other. The accuracy intervals include zero. This is a plausible feature-transport concern but **not a confirmed predictive gain**; do not change the production refit or submit a CSV based on it.

Private paired CSV SHA-256: seed39 `bc3b24db3a6e84c93011cf7f15f9089e93b2086584b3f309a54de21060b4809d`; seed40 `669c4e98333f8d2f4dbdeeeb0edac3233d39e407c63f8ce8dfd841cffddfbf60`. JSON reports include fit/tune/confirm sizes, threshold, seen/new and input hashes. Keep row-level labels/pairs private.
