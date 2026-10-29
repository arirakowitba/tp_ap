# Round 150 — exact high-margin new-seller gate on labeled confirm (2026-10-29)

Reviewed main through `037f6a66f98ab25dcb443cc8d21e9da59f6ffce8`; Claude remains at `72c790c`. Antigravity's new `0dea115` commit produced 64/65 but no new independent model validation.

We reconstructed the exact rule of candidate 65 on Antigravity's two labeled seller-group confirm files: seller not in fit, dual-text prediction disagrees with previous CatBoost prediction, and `abs(p_dual - .5) >= .25`. Join `paired_dual_text_seed_*.csv` with `paired_predictions_seed_*.csv` by listing ID; assert each ID exists. The previous CatBoost is a **proxy** for submission 48; no train analogue of the 48 policy is available here. No Kaggle score is interpreted as an individual label.

| Seed | Selected | Fixes | Breaks | Net | Direction used→new / new→used |
|---:|---:|---:|---:|---:|---:|
| 2026 | 17 | 6 | 11 | **−5** | 9 / 8 |
| 314159 | 16 | 10 | 6 | **+4** | 8 / 8 |

This gate has too few confirm cases for a stable inference, and the net sign reverses. The 141 test flips in candidate 65 are therefore **not validated as beneficial** by this proxy experiment. Neither candidate 64 nor 65 should be promoted on these data. Keep submission 48 as the known public reference. Next, recover a true train-only analogue of policy 48 on IDs and test the same rule on larger seller-group holdouts, or validate a new strong model directly. Avoid repeating probability blends and weak-baseline gains.

Calculation from stored CSV columns: `id,y_true,is_seen,p_dual,pred_dual` and `id,pred_cat_used`; count selected rows for which `pred_dual == y_true` as fixes and the others as breaks. The two confirm sets overlap and are not statistically independent.
