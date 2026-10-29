# Round 149 — strong-baseline integration check (2026-10-29)

Reviewed main HEAD `95a4b3800f9f18af6f824c654a32aae1437b943c` and Claude branch `72c790c16e28189e1d42fd659ab682da9a72b053` (identical to prior review; no new Claude commits). Antigravity work through `618af882` is already on main. No new model work beyond round 148.

## Protocol

Use the exact historical round13 valid IDs (14,000), checked against the original 70k labels and the round13 CatBoost/LGBM/HistGB predictions. Split this valid set by seller into tune 7,038 and confirm 6,962 (zero seller overlap between halves). The underlying strong predictions were trained on the historical 56k fit. NB-SVM-augmented compact CatBoost is from round148. Choose coefficients by tune accuracy, then tune log loss, then minimum weight. Evaluate confirm once. Bootstrap confirm net correctness by seller, 3,000 draws. This is a diagnostic on a historical holdout, not a fresh independent validation of all prior feature selection.

## Result

| Candidate | Tune net vs strong CatBoost | Confirm net | Confirm log loss | Seen/new confirm net |
|---|---:|---:|---:|---:|
| Strong CatBoost | 0 | 0 | .140549 | 0/0 |
| Strong CatBoost + 5% compact NB-SVM model | -2 | — | — | — |
| Strong CatBoost + 10% compact NB-SVM model | -3 | — | — | — |
| Strong CatBoost + 20% compact NB-SVM model | -14 | — | — | — |
| Strong CatBoost + 35% compact NB-SVM model | -24 | — | — | — |
| Selected NB-SVM blend | 0, weight 0 | 0 | .140549 | 0/0 |
| Strong CatBoost + 20% strong LGBM | +2 | 0 | .139614 | +2/-2 |

The constrained strong-model blend was selected among nonnegative CatBoost/LGBM/HistGB weights on tune only. On confirm it has 9 fixes and 9 breaks (net 0, seller bootstrap 95% CI [-8,+8]); accuracy .946998, exactly the same as strong CatBoost. Log loss improves from .140549 to .139614 but no accuracy gain. In unseen sellers it loses 2. Do not generate a Kaggle submission from these results.

Round148's NB-SVM signal remains promising *within its compact family* but is not additive as a raw probability blend to the strong model. The required next test is integration as a leak-free feature in the strong deep-feature architecture, with equivalent training and paired seller-group confirm. The original deep-feature table is not in this local checkpoint, so this round does not claim that test.

Checkpoint: `round149_checkpoint.zip`, containing both scripts, per-ID paired predictions and JSON metrics. Test labels were never used. Public scores were not used as per-ID labels.