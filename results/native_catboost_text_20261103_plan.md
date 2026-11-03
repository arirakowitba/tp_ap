# Native CatBoost text interaction — predeclared plan

The strong baseline compresses word+char TF-IDF text into one grouped OOF logistic probability before CatBoost. A separate model using CatBoost's native text features may learn interactions between words, categories and structural features. Test it against the *same* strong baseline, not a weaker char-only model.

Use original 70k labels (`used=1`) and original title/warranty text. Predeclare seller-group seed 20261141 for the first feasibility/paired confirm and 20261142 as a second fresh confirmation if the pipeline is tractable. Both models use identical fit/tune/confirm, structural extraction, target encoding fit-only, global word+char TF-IDF OOF, seller categorical, depth6/lr.07/l2=6 and tune-only early stopping/accuracy threshold. Candidate adds `title_text` and `warranty_text` as native CatBoost text features, with no test labels or public scores. For each cut, report paired accuracy/logloss, fixes/breaks, seen/new, seller bootstrap and full hashes/pairs. Inspect training time and fit feasibility; no public CSV without replicated sizable gain.

Code: `scripts/ablate_native_catboost_text_20261103.py`. ZIP SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`.
