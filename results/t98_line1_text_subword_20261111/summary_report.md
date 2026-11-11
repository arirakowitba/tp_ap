# T98 Line 1 — Original Text & Subword Specialist Summary Report

- **Status:** `DEVELOPMENT_ONLY` (`submission_generated = false`, `kaggle_api_used = false`)
- **Script:** `scripts/t98_line1_text_subword.py` (`SHA256: 8e4d9b302e2b8b8ab36380090111bc5f67a92ce51ab1a608a38e0171dbf568e6`)
- **Output Directory:** `results/t98_line1_text_subword_20261111/`

---

## 1. Standalone Dual-Regime Performance Summary

### Regime A: `unseen_val` (70,000 items across 5 seller-disjoint outer folds)

| Model | Hits / 70,000 | Accuracy | LogLoss | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| `T97_p_text` (Reference outer-val from T97) | 56,552 | 0.807886 | 0.426914 | 0.884667 |
| `M0_podium_text` (`title + tags_str`, Word 1-2 + Char_wb 3-5, SAGA C=1.5) | 56,327 | 0.804671 | 0.432891 | 0.881329 |
| `M1_word_char_enriched_saga` (Enriched item-intrinsic + domain cues, Word 1-3 80k + Char_wb 3-5 80k) | 60,585 | 0.865500 | 0.314670 | 0.939408 |
| `M2_subword_char_nb_svm` (`char(2,6)` 60k + `char_wb(3,6)` 60k + NB log-count ratio `r`) | 60,417 | 0.863100 | 0.322090 | 0.937500 |
| **`M3_nested_text_blend` (Frozen inner-OOF logit blend of M1 + M2)** | **60,697** | **0.867100** | **0.311181** | **0.940836** |

### Regime B: `seen_val` (32,092 chronological latest items of multi-item fit sellers across 5 outer folds)

| Model | Hits / 32,092 | Accuracy | LogLoss | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| `T97_p_text` | 26,420 | 0.823258 | 0.407280 | 0.890950 |
| `M0_podium_text` | 26,660 | 0.830737 | 0.397087 | 0.897872 |
| **`M1_word_char_enriched_saga`** | **28,696** | **0.894179** | 0.263397 | 0.955758 |
| `M2_subword_char_nb_svm` | 28,391 | 0.884675 | 0.280496 | 0.950712 |
| **`M3_nested_text_blend`** | 28,671 | 0.893400 | **0.260807** | **0.956232** |

---

## 2. Paired Comparisons & Two-Way Cluster Bootstrap (`seller_id x category_id`, 2,000 reps, seed `20261111`)

### Unseen Sellers (`unseen_val`, $N = 70,000$)

| Comparison | Net $\Delta$ Hits | $\Delta$ Acc | Fixes / Breaks | $\Delta$ LogLoss | Pos Folds | Worst Fold $\Delta$ Hits | Bootstrap $\Delta$ Hits Median `[Q05, Q95]` | Bootstrap $P(\Delta > 0)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M1` vs `M0_podium_text` | **+4,258** | +0.060829 | 7,339 / 3,081 | -0.118221 | 5 / 5 | +663 (Fold 1) | `+4248.8 [+3754.7, +4821.2]` | 1.0000 |
| `M2` vs `M0_podium_text` | **+4,090** | +0.058429 | 7,754 / 3,664 | -0.110801 | 5 / 5 | +659 (Fold 1) | `+4082.4 [+3574.9, +4673.4]` | 1.0000 |
| **`M3` vs `M0_podium_text`** | **+4,370** | **+0.062429** | **7,430 / 3,060** | **-0.121710** | **5 / 5** | **+717 (Fold 1)** | **`+4363.0 [+3864.8, +4939.5]`** | **1.0000** |
| `M1` vs `T97_p_text` | **+4,033** | +0.057614 | 7,217 / 3,184 | -0.112244 | 5 / 5 | +645 (Fold 1) | `+4032.5 [+3516.0, +4574.3]` | 1.0000 |
| `M2` vs `T97_p_text` | **+3,865** | +0.055214 | 7,591 / 3,726 | -0.104824 | 5 / 5 | +641 (Fold 1) | `+3862.0 [+3333.6, +4437.7]` | 1.0000 |
| **`M3` vs `T97_p_text`** | **+4,145** | **+0.059214** | **7,297 / 3,152** | **-0.115733** | **5 / 5** | **+699 (Fold 1)** | **`+4142.8 [+3640.1, +4697.4]`** | **1.0000** |

### Seen Sellers (`seen_val`, $N = 32,092$ across 5 folds)

| Comparison | Net $\Delta$ Hits | $\Delta$ Acc | Fixes / Breaks | $\Delta$ LogLoss | Pos Folds | Worst Fold $\Delta$ Hits | Bootstrap $\Delta$ Hits Median `[Q05, Q95]` | Bootstrap $P(\Delta > 0)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M1` vs `M0_podium_text` | **+2,036** | +0.063443 | 2,971 / 935 | -0.133690 | 5 / 5 | +372 (Fold 3) | `+2038.6 [+1738.6, +2354.8]` | 1.0000 |
| `M2` vs `M0_podium_text` | **+1,731** | +0.053939 | 3,090 / 1,359 | -0.116590 | 5 / 5 | +308 (Fold 3) | `+1735.2 [+1416.2, +2083.9]` | 1.0000 |
| **`M3` vs `M0_podium_text`** | **+2,011** | **+0.062664** | **3,007 / 996** | **-0.136279** | **5 / 5** | **+373 (Fold 3)** | **`+2013.9 [+1711.4, +2329.3]`** | **1.0000** |
| **`M3` vs `T97_p_text`** | **+2,251** | **+0.070142** | **3,305 / 1,054** | **-0.146473** | **5 / 5** | **+439 (Fold 0)** | **`+2263.2 [+1908.7, +2621.5]`** | **1.0000** |

---

## 3. Per-Fold Metrics Breakdown (`M3_nested_text_blend` vs `M0_podium_text`)

| Outer Fold | `n_fit` | `n_seen_val` | `n_unseen_val` | Inner OOF Acc (`M0` $\to$ `M3`) | Unseen Acc (`M0` $\to$ `M3`) | Unseen $\Delta$ Hits vs `M0` (Fixes/Breaks) | Seen Acc (`M0` $\to$ `M3`) | Seen $\Delta$ Hits vs `M0` (Fixes/Breaks) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 49,547 | 6,453 | 14,000 | 0.79518 $\to$ 0.85904 | 0.80100 $\to$ 0.86950 | **+959** (1554 / 595) | 0.83372 $\to$ 0.89369 | **+387** (586 / 199) |
| **1** | 49,599 | 6,401 | 14,000 | 0.78949 $\to$ 0.85808 | 0.82636 $\to$ 0.87757 | **+717** (1435 / 718) | 0.82753 $\to$ 0.89174 | **+411** (628 / 217) |
| **2** | 49,594 | 6,406 | 14,000 | 0.79175 $\to$ 0.85954 | 0.80836 $\to$ 0.87229 | **+895** (1433 / 538) | 0.82735 $\to$ 0.89135 | **+410** (614 / 204) |
| **3** | 49,538 | 6,462 | 14,000 | 0.80262 $\to$ 0.86612 | 0.78636 $\to$ 0.85879 | **+1,014** (1579 / 565) | 0.83627 $\to$ 0.89400 | **+373** (572 / 199) |
| **4** | 49,630 | 6,370 | 14,000 | 0.79859 $\to$ 0.86675 | 0.80129 $\to$ 0.85736 | **+785** (1429 / 644) | 0.82873 $\to$ 0.89623 | **+430** (607 / 177) |

---

## 4. Saved Deliverables & SHA-256 Hashes

- `fold_0_predictions.npz`: `f922a879cbda5a2d054d784a4e6f1d83608ec7bfecb19b9930f999ff87122802`
- `fold_1_predictions.npz`: `40c5cb189e8879697708b3f6de6858ca4733ad9975cf1d21bf496180aff2f683`
- `fold_2_predictions.npz`: `157edc9ca033e209f24ee2d19ae6591ad19fbb7d8422bf6d38952477c2dbfb11`
- `fold_3_predictions.npz`: `ae3cab79839ccf13382a92ce92a73f3b69524b5b156f4b6bfb09cf8fba2fbe8b`
- `fold_4_predictions.npz`: `f8a4b3fa80b83d8137e4d04d5c0011b11ea66b297ae1cdaa5d8767ec49efe605`
- `inner_oof_fit_predictions.csv`: `cd205266d07286ab221c3d83465def643f9c01a8021cb7d776fd3cc762496a68`
- `oof_unseen_70k.csv`: `f05ee6e258fd897699c9c8e40138f5902193fe67a3322d046e031e9e751c71af`
- `oof_seen_all_folds.csv`: `bd1d47fffd9dd760b9d696c7babc613c2a1c849657a0e247ed146b40772d1005`
- `test_predictions_30k.csv`: `263bf32e6754e35938b9aa3dae5424f50b76c9c7eac0fc874af0e0e4f357712e`
- `test_disagreements_vs_s4_probs_only.csv`: `3e90a487c8d3616fcd8c8b5601836015d054441cd921aa7bff7307d6ec850736` (2,728 rows where `(p_m3_blend >= 0.5)` disagrees with S4, containing only `ID` and model probabilities)
- `per_fold_metrics.csv`: `393880715d9c2c441fa045c8574335d99253a17443555686f6de102143f7c5e4`
- `manifest.json`: Saved in `results/t98_line1_text_subword_20261111/manifest.json`
