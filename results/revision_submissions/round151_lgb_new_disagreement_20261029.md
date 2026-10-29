# Round 151 — LGBM disagreement gate against strong CatBoost (2026-10-29)

Main `60671d7` and Claude `72c790c` unchanged at review. On the round13 labeled valid set, using the seller-disjoint 7,038 tune / 6,962 confirm split from round149, we tested whether strong LGBM can correct strong CatBoost on *new sellers* when LGBM disagrees and has margin `abs(p_lgb-.5)>=m`. This diagnostic uses the historical 56k-fit models and true train labels only. Predictions are in round149 checkpoint, joined by checked row ID.

| Minimum LGBM margin | Tune selected/fixes/breaks/net | Confirm selected/fixes/breaks/net |
|---:|---:|---:|
| .05 | 24 / 7 / 17 / **−10** | 26 / 9 / 17 / **−8** |
| .10 | 13 / 2 / 11 / **−9** | 9 / 3 / 6 / **−3** |
| .20 | 1 / 0 / 1 / **−1** | 1 / 1 / 0 / **+1** |
| .30 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The rule is rejected: at useful sample sizes it consistently harms accuracy. Do not promote an LGBM-over-CatBoost new-seller disagreement probe to Kaggle. This also illustrates why margin alone is insufficient. No test labels or public-score-derived labels were used.
