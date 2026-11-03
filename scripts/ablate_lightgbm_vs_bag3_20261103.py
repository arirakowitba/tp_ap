"""Corrected train-only LightGBM comparison vs strong text OOF+seller bag3.

Unlike AG-0040, no blend weight or threshold is selected on confirm. The
LightGBM model early-stops on tune, then is evaluated at a fixed .5 threshold.
The .5 blend with the pre-existing bag3 probabilities is fixed in advance.
"""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ablate_attr_value_vs_historical_text import make_te, text_oof
from scripts.train_deep_text_20261101 import split_testlike
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261125
ZIP = Path(os.environ["AP_ZIP"])
OUT = ROOT / "results/lightgbm_vs_bag3"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with zipfile.ZipFile(ZIP) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000
    y = np.array([r["condition"] == "used" for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])
    fit, tune, confirm = split_testlike(sellers, SEED)
    a, cat, titles, _, combos = extract_deep_features(rows)
    make_te(a, cat, y, fit, tune, confirm, sellers)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                           sublinear_tf=True, strip_accents="unicode", lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           max_features=150000, sublinear_tf=True)
    xf = sp.hstack([word.fit_transform([combos[i] for i in fit]),
                    char.fit_transform([titles[i] for i in fit])], format="csr")
    xt = sp.hstack([word.transform([combos[i] for i in tune]),
                    char.transform([titles[i] for i in tune])], format="csr")
    xv = sp.hstack([word.transform([combos[i] for i in confirm]),
                    char.transform([titles[i] for i in confirm])], format="csr")
    oof, pt, pv = text_oof(xf, xt, xv, y, fit, sellers, 3.)
    a["txt_sparse_p"] = np.nan
    for idx, prob in ((fit, oof), (tune, pt), (confirm, pv)):
        a.loc[idx, "txt_sparse_p"] = prob
    a["seller_id_cat"] = cat.seller_id.astype(str)
    cats = ["category_id", "listing_type_id", "buying_mode", "state",
            "ship_mode", "seller_id_cat"]
    for col in cats:
        a[col] = a[col].astype("category")
    model = lgb.LGBMClassifier(objective="binary", n_estimators=1200,
                               learning_rate=.04, num_leaves=31,
                               min_child_samples=40, max_depth=-1,
                               colsample_bytree=.85, subsample=.85,
                               subsample_freq=1, reg_lambda=6.,
                               random_state=SEED, n_jobs=7, verbosity=-1)
    model.fit(a.iloc[fit], y[fit], categorical_feature=cats,
              eval_set=[(a.iloc[tune], y[tune])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    p_lgb = model.predict_proba(a.iloc[confirm])[:, 1]
    prior = pd.read_csv(ROOT / f"results/catboost_bag3/paired_{SEED}.csv")
    assert len(prior) == len(confirm)
    assert np.array_equal(prior.item_id.astype(str), ids[confirm])
    assert np.array_equal(prior.y_true.to_numpy(), y[confirm])
    p_bag = prior.p_bag3.to_numpy()
    p_blend = .5 * p_lgb + .5 * p_bag
    p_single = prior.p_single.to_numpy()
    seen = np.isin(sellers[confirm], sellers[fit])
    p_new_gate = np.where(seen, p_bag, p_blend)
    probs = {"historical_single": p_single, "bag3": p_bag,
             "lightgbm": p_lgb, "fixed_half_blend": p_blend,
             "new_seller_only_blend": p_new_gate}
    metrics = {name: {"accuracy05": float(np.mean((p >= .5) == y[confirm])),
                      "logloss": float(log_loss(y[confirm], p))}
               for name, p in probs.items()}
    pairs = pd.DataFrame({"item_id": ids[confirm], "seller_id": sellers[confirm],
                          "is_seen": seen.astype(int), "y_true": y[confirm],
                          "p_single": p_single, "p_bag3": p_bag,
                          "p_lgb": p_lgb, "p_fixed_half_blend": p_blend,
                          "p_new_seller_only_blend": p_new_gate})
    path = OUT / f"paired_{SEED}.csv"
    pairs.to_csv(path, index=False)
    reports = {}
    for name in ("lightgbm", "fixed_half_blend", "new_seller_only_blend"):
        delta = ((probs[name] >= .5) == y[confirm]).astype(int) - \
                ((p_bag >= .5) == y[confirm]).astype(int)
        by = pairs.assign(delta=delta).groupby("seller_id").delta.sum().to_numpy()
        rng = np.random.default_rng(SEED)
        boot = np.array([by[rng.integers(0, len(by), len(by))].sum()
                         for _ in range(3000)])
        reports[name] = {"net_vs_bag3": int(delta.sum()),
                         "fixes_breaks": [int((delta == 1).sum()), int((delta == -1).sum())],
                         "seen_new_net": {str(i): int(delta[seen == bool(i)].sum())
                                          for i in (0, 1)},
                         "seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist()}
    report = {"seed": SEED, "fit_tune_confirm": [len(fit), len(tune), len(confirm)],
              "train_zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "best_iteration_tune": model.best_iteration_, "threshold": .5,
              "blend_weight_fixed": .5, "metrics": metrics,
              "paired_comparisons": reports,
              "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / f"report_{SEED}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
