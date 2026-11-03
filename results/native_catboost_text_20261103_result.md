# Native CatBoost title/warranty text ablation (2026-11-03)

Plan prepublished: `results/native_catboost_text_20261103_plan.md`. Code: `scripts/ablate_native_catboost_text_20261103.py`. Original 70k train labels `used=1`, 30k test schema checked without using test labels, ZIP SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`.

The candidate adds raw original `title` and `warranty` columns as CatBoost native text features to the historical strong CatBoost with word+char TF-IDF OOF probability, structural features and seller categorical. Native text uses CatBoost CPU BoW (top 1000 tokens) and NaiveBayes feature calcers as documented by CatBoost. Both arms share seller-group fit/tune/confirm, supervised TE and grouped text OOF. Early stopping and accuracy threshold are tune-only; confirm is evaluated once per predeclared seed.

| Seed | Confirm N | Historical / native-text accuracy | Native net | Fixes / breaks | Historical / native log loss | Seller bootstrap 95% CI | Net at common .500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261141 | 12,448 | .955816 / .952442 | -42 | 53 / 95 | .119579 / .125210 | [-78, -11] | -45 |
| 20261142 | 12,629 | .956608 / .955262 | -17 | 54 / 71 | .119119 / .122284 | [-44, +8] | -17 |

The confirm sets share 2,668 IDs. Deduplicating yields 22,409 unique IDs. Keeping the first seed on overlap gives net -50, seller-bootstrap CI [-92, -11]; keeping the second gives -47, CI [-86, -11]. The candidate loses in both seen and new seller subsets and worsens log loss in both cuts. This native text configuration is rejected as a route to a larger base improvement. It does not rule out other text models, but no Kaggle CSV should be produced from it.

Private paired CSV SHA-256: seed41 `a41d0283f17c5ab2e3b6d1268340a803bc578ee6f36314d38a495b41ec7667e6`; seed42 `da9a7bd05440bd02722f40ba1f5bf6b95af6e0c9dcf3cbdab3ef0cf5d314667a`. Reports include input and split hashes, tune thresholds, seen/new and bootstrap. Keep row-level labels private.
