"""Predeclared depth-8 CatBoost comparison on historical text OOF + seller.

This uses only original train labels. It joins an existing private bag3 confirm
checkpoint by item_id; no public score or test label enters model selection.
"""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ablate_attr_value_vs_historical_text import make_te, text_oof
from scripts.train_deep_text_20261101 import split_testlike
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

SEED = 20261124
ZIP = Path(os.environ["AP_ZIP"])
OUT = ROOT / "results/catboost_depth8"
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
    model = CatBoostClassifier(iterations=1200, depth=8, learning_rate=.045,
                               l2_leaf_reg=8., random_seed=SEED,
                               thread_count=7, verbose=False, loss_function="Logloss")
    model.fit(a.iloc[fit], y[fit], cat_features=cats,
              eval_set=(a.iloc[tune], y[tune]), early_stopping_rounds=100,
              use_best_model=True)
    p8 = model.predict_proba(a.iloc[confirm])[:, 1]
    prior = pd.read_csv(ROOT / f"results/catboost_bag3/paired_{SEED}.csv")
    assert len(prior) == len(confirm)
    assert np.array_equal(prior.item_id.astype(str), ids[confirm])
    assert np.array_equal(prior.y_true.to_numpy(), y[confirm])
    for name, p in (("depth8", p8), ("bag3", prior.p_bag3),
                    ("single", prior.p_single)):
        print(name, "accuracy05", float(np.mean((p >= .5) == y[confirm])),
              "logloss", float(log_loss(y[confirm], p)), flush=True)
    delta = ((p8 >= .5) == y[confirm]).astype(int) - \
            ((prior.p_bag3.to_numpy() >= .5) == y[confirm]).astype(int)
    seen = np.isin(sellers[confirm], sellers[fit])
    pairs = pd.DataFrame({"item_id": ids[confirm], "seller_id": sellers[confirm],
                          "is_seen": seen.astype(int), "y_true": y[confirm],
                          "p_depth8": p8, "p_bag3": prior.p_bag3,
                          "p_single": prior.p_single})
    path = OUT / f"paired_{SEED}.csv"
    pairs.to_csv(path, index=False)
    by = pairs.assign(delta=delta).groupby("seller_id").delta.sum().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([by[rng.integers(0, len(by), len(by))].sum()
                     for _ in range(3000)])
    report = {"seed": SEED, "train_zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "best_iteration": model.best_iteration_, "fixed_threshold": .5,
              "accuracy05": {name: float(np.mean((p >= .5) == y[confirm])) for name, p in
                             (("depth8", p8), ("bag3", prior.p_bag3),
                              ("single", prior.p_single))},
              "logloss": {name: float(log_loss(y[confirm], p)) for name, p in
                          (("depth8", p8), ("bag3", prior.p_bag3),
                           ("single", prior.p_single))},
              "net_depth8_vs_bag3": int(delta.sum()),
              "fixes_breaks_vs_bag3": [int((delta == 1).sum()), int((delta == -1).sum())],
              "seen_new_net": {str(i): int(delta[seen == bool(i)].sum()) for i in (0, 1)},
              "seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
              "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / f"report_{SEED}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
