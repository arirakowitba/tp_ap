# Category-conditioned text OOF ablation (2026-11-03)

Plans prepublished: `results/category_text_interaction_20261103_plan.md` (seeds 20261136/37) and `results/category_text_blend_fresh_20261103_plan.md` (fixed .25 blend, seed 20261138). Code: `scripts/ablate_category_text_interaction_20261103.py`. Original train70k ZIP SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`; `used=1`. No public scores or test labels entered training or selection.

The candidate adds one grouped OOF probability from category-prefixed word unigrams/bigrams, hashed into 2^18 dimensions and IDF-fitted on fit only, to the historical strong word+char TF-IDF OOF + CatBoost seller categorical baseline. The candidate is a category/text interaction representation; it does not use attribute `value_name` (Antigravity's separate line). Both arms use identical seller-group fit/tune/confirm. Early stopping and standalone decision threshold are tune-only.

| Seed | Confirm N | Standalone candidate net vs historical | Fixes / breaks | Historical / candidate log loss | Seller bootstrap 95% CI | Net at shared .500 threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261136 | 12,735 | -12 | 52 / 64 | .129365 / .128525 | [-33, +10] | -3 |
| 20261137 | 12,725 | -3 | 37 / 40 | .133877 / .132968 | [-19, +13] | -8 |
| 20261138 | 12,389 | -1 | 61 / 62 | .132183 / .132658 | [-24, +22] | -2 |

The first two confirms became **development** after inspecting a fixed probability blend. They share 2,933 IDs; deduplicated 22,527 unique IDs give standalone candidate net -13 or -16 depending on which seed has priority. A .25 candidate probability blend at common .500 threshold gave exploratory +9 or +5 on the deduplicated development set, not a confirmation result.

Before looking at seed 20261138, the blend was frozen as `.75*p_historical + .25*p_category_text` with decision threshold `.500`. On its fresh 12,389-row confirm, the historical model got 11,769 correct and the blend got 11,774: **net +5**, 17 fixes / 12 breaks, seen/new +2/+3, seller bootstrap 95% CI **[-5,+16]**. Log loss improved .132183→.131838. The improvement is too small and uncertain to promote, and the standalone candidate did not improve accuracy on any of three cuts. It does not address the ~158 public correct-answer gap to 0.95404.

Private paired CSV SHA-256: 20261136 `5ce6420f3e2d57f5f208e290d57c9a602cb399236a171e06d986d2f7398dc296`, 20261137 `3b9e13b51fcd19d32f62d94c3900c984b22c9731c09d40887f259850044ef46d`, 20261138 `8f77019f034b567911c33383fc6b48cb68ba6d2a94b2651043a90abd7115e974`. JSON reports include split hashes, selected thresholds, seen/new and bootstrap. Paired rows contain labels and must remain private.

**Decision:** retain only as a weak research signal. No Kaggle submission, no public-score inference by ID. Prioritize a base model with larger, independently confirmed gains over further fine-tuning this blend.
