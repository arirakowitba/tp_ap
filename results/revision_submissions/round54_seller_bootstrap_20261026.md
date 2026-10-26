# Round54 seller-bootstrap audit (2026-10-26)

This uses the saved round54 paired predictions and the corrected CatBoost .5 threshold. The frozen FastText `used→new` subset contains 119 rows in each seller-disjoint section.

| Section | Sellers | FastText vs CatBoost net | Seller bootstrap 95% net |
|---|---:|---:|---:|
| Tune | 102 | −3 | [−22,+17] |
| Confirm | 106 | +3 | [−16,+22] |

Bootstrap resampled sellers 5,000 times with seed 5454. Both intervals are wide and include zero. The +3 confirm result does not justify test flips. Data source: round54 checkpoint Library `libfile_454ba4773eb08191b8bf035f0f2dc844`; threshold correction `libfile_de1e883de3688191bd3a8769309c876c`. No Kaggle quota used.
