# Rich CatBoost challenger

Same original row partitions as 21 October: 49k fit / 7k tune, refit 56k / confirm 14k. The confirmation set has prior historical use and is not a pristine final estimate. No leaderboard-derived or recovered test labels used.

Features: fold-scoped mean/median/std/count of logprice for category and seller, relative price, seller category diversity; deterministic availability/sales/quantity gap, variation aggregates, brand/model/material/gender, catalog/store/status fields; CatBoost BoW, NaiveBayes and BM25 text with 10k unigram dictionary on title, warranty, safe attribute tokens. Attribute keys containing CONDITION excluded.

Two configurations selected on tune accuracy, then log loss as tie breaker. Depth6/1099 trees selected at 92.5429% tune; depth7/900 at 92.5%. Confirmation selected model 92.7714%, below native baseline 93.1571%. Rich repairs 84 native errors but breaks 138. Simple confirmation blend diagnostics do not improve baseline; they are not held-out selection results. No claim of superiority or recommendation to submit this model.

`progress.json` is the completed tuning/confirmation checkpoint; `report.json`, `model.cbm` and `test.npz` appear after full fit finishes. Script: `scripts/train_rich_next.py`. Public 26 remains separate from this experiment.
