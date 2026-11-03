"""Predeclared seller leave-one-out history feature on strong text OOF CatBoost.

The candidate uses only fit labels. Fit rows exclude their own label from
seller aggregates; tune and confirm use the full fit aggregate. No public
score, test label, or confirmation label enters feature construction.
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
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ablate_attr_value_vs_historical_text import make_te, text_oof
from scripts.train_deep_text_20261101 import split_testlike
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261129
ZIP = Path(os.environ["AP_ZIP"])
OUT = ROOT / "results/seller_loo_feature"
OUT.mkdir(parents=True, exist_ok=True)


def seller_history(sellers, y, fit, tune, confirm, prior, alpha=10.):
    stats = pd.DataFrame({"s": sellers[fit], "y": y[fit]}).groupby("s").y.agg(["sum", "count"])
    total = pd.Series(sellers).map(stats["sum"]).fillna(0).to_numpy(dtype=float)
    count = pd.Series(sellers).map(stats["count"]).fillna(0).to_numpy(dtype=float)
    rate = np.full(len(sellers), prior, dtype=np.float32)
    n = np.zeros(len(sellers), dtype=np.float32)
    rate[fit] = ((total[fit] - y[fit] + alpha * prior) /
                 (count[fit] - 1 + alpha)).astype(np.float32)
    n[fit] = count[fit] - 1
    for ix in (tune, confirm):
        rate[ix] = ((total[ix] + alpha * prior) / (count[ix] + alpha)).astype(np.float32)
        n[ix] = count[ix]
    assert np.all(np.isfinite(rate)) and np.all(n >= 0)
    return rate, n


def main():
    with zipfile.ZipFile(ZIP) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000
    y = np.array([r["condition"] == "used" for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])
    fit, tune, confirm = split_testlike(sellers, SEED)
    assert len(set(ids[fit]) & set(ids[tune])) == 0
    assert len(set(ids[fit]) & set(ids[confirm])) == 0
    assert len(set(ids[tune]) & set(ids[confirm])) == 0
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
    for ix, p in ((fit, oof), (tune, pt), (confirm, pv)):
        a.loc[ix, "txt_sparse_p"] = p
    a["seller_id_cat"] = cat.seller_id.astype(str)
    rate, n = seller_history(sellers, y, fit, tune, confirm, float(y[fit].mean()))
    b = a.copy()
    b["seller_loo_rate"] = rate
    b["seller_hist_n"] = n
    cats = ["category_id", "listing_type_id", "buying_mode", "state",
            "ship_mode", "seller_id_cat"]
    probs = {}
    iterations = {}
    for name, frame in (("base", a), ("loo", b)):
        model = CatBoostClassifier(iterations=900, depth=6, learning_rate=.07,
                                   l2_leaf_reg=6., random_seed=SEED,
                                   thread_count=7, verbose=False,
                                   loss_function="Logloss")
        model.fit(frame.iloc[fit], y[fit], cat_features=cats,
                  eval_set=(frame.iloc[tune], y[tune]),
                  early_stopping_rounds=100, use_best_model=True)
        probs[name] = (model.predict_proba(frame.iloc[tune])[:, 1],
                       model.predict_proba(frame.iloc[confirm])[:, 1])
        iterations[name] = model.best_iteration_
        print(name, model.best_iteration_, flush=True)
    grid = np.arange(.44, .561, .005)
    thresholds = {name: float(max(grid, key=lambda t: accuracy_score(y[tune], p[0] >= t)))
                  for name, p in probs.items()}
    pred = {name: p[1] >= thresholds[name] for name, p in probs.items()}
    delta = (pred["loo"] == y[confirm]).astype(int) - \
            (pred["base"] == y[confirm]).astype(int)
    seen = np.isin(sellers[confirm], sellers[fit])
    pairs = pd.DataFrame({"item_id": ids[confirm], "seller_id": sellers[confirm],
                          "is_seen": seen.astype(int), "y_true": y[confirm],
                          "p_base": probs["base"][1], "p_loo": probs["loo"][1],
                          "pred_base": pred["base"].astype(int),
                          "pred_loo": pred["loo"].astype(int), "delta": delta})
    path = OUT / f"paired_{SEED}.csv"
    pairs.to_csv(path, index=False)
    by = pairs.groupby("seller_id").delta.sum().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([by[rng.integers(0, len(by), len(by))].sum()
                     for _ in range(3000)])
    report = {"seed": SEED, "train_zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "fit_tune_confirm": [len(fit), len(tune), len(confirm)],
              "confirm_ids_sha256": hashlib.sha256(ids[confirm].astype("U").tobytes()).hexdigest(),
              "iterations": iterations, "thresholds_tune": thresholds,
              "accuracy": {k: float(accuracy_score(y[confirm], v)) for k, v in pred.items()},
              "logloss": {k: float(log_loss(y[confirm], v[1])) for k, v in probs.items()},
              "net": int(delta.sum()),
              "fixes_breaks": [int((delta == 1).sum()), int((delta == -1).sum())],
              "common05_net": int(((probs["loo"][1] >= .5) == y[confirm]).sum() -
                                  ((probs["base"][1] >= .5) == y[confirm]).sum()),
              "seen_new_net": {str(i): int(delta[seen == bool(i)].sum()) for i in (0, 1)},
              "seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
              "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / f"report_{SEED}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
