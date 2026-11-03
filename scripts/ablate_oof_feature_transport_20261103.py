"""Paired seller-group test of full-fit vs fold-ensemble inference features."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_deep_text_20261101 import split_testlike
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

SEED = int(os.environ.get("AP_SEED", "20261139"))
ZIP = Path(os.environ["AP_ZIP"])
OUT = ROOT / "results/oof_feature_transport"
OUT.mkdir(parents=True, exist_ok=True)


def smooth_map(keys, labels, prior):
    stats = pd.DataFrame({"key": np.asarray(keys), "y": labels}).groupby("key").y.agg(["sum", "count"])
    return (stats["sum"] + 5 * prior) / (stats["count"] + 5)


def main():
    with zipfile.ZipFile(ZIP) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000
    y = np.array([r["condition"] == "used" for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])
    fit, tune, confirm = split_testlike(sellers, SEED)
    a, cat, titles, _, combos = extract_deep_features(rows)
    b = a.copy()
    folds = list(GroupKFold(4).split(fit, y[fit], sellers[fit]))
    prior = float(y[fit].mean())
    for col in ("seller_id", "category_id", "city"):
        train_oof = np.full(len(fit), prior, np.float32)
        tune_folds, confirm_folds = [], []
        for tr, va in folds:
            mapping = smooth_map(cat[col].iloc[fit[tr]], y[fit[tr]], prior)
            train_oof[va] = cat[col].iloc[fit[va]].map(mapping).fillna(prior).to_numpy()
            tune_folds.append(cat[col].iloc[tune].map(mapping).fillna(prior).to_numpy())
            confirm_folds.append(cat[col].iloc[confirm].map(mapping).fillna(prior).to_numpy())
        full = smooth_map(cat[col].iloc[fit], y[fit], prior)
        name = f"te_{col}"
        for frame in (a, b):
            frame[name] = np.nan
            frame.loc[fit, name] = train_oof
        a.loc[tune, name] = cat[col].iloc[tune].map(full).fillna(prior).to_numpy()
        a.loc[confirm, name] = cat[col].iloc[confirm].map(full).fillna(prior).to_numpy()
        b.loc[tune, name] = np.mean(tune_folds, axis=0)
        b.loc[confirm, name] = np.mean(confirm_folds, axis=0)
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
    train_oof = np.empty(len(fit), np.float32)
    tune_folds, confirm_folds = [], []
    for tr, va in folds:
        lr = LogisticRegression(C=3., max_iter=250, solver="liblinear", random_state=SEED)
        lr.fit(xf[tr], y[fit[tr]])
        train_oof[va] = lr.predict_proba(xf[va])[:, 1]
        tune_folds.append(lr.predict_proba(xt)[:, 1])
        confirm_folds.append(lr.predict_proba(xv)[:, 1])
    full_lr = LogisticRegression(C=3., max_iter=300, solver="liblinear", random_state=SEED)
    full_lr.fit(xf, y[fit])
    for frame in (a, b):
        frame["txt_sparse_p"] = np.nan
        frame.loc[fit, "txt_sparse_p"] = train_oof
        frame["seller_id_cat"] = cat.seller_id.astype(str)
    a.loc[tune, "txt_sparse_p"] = full_lr.predict_proba(xt)[:, 1]
    a.loc[confirm, "txt_sparse_p"] = full_lr.predict_proba(xv)[:, 1]
    b.loc[tune, "txt_sparse_p"] = np.mean(tune_folds, axis=0)
    b.loc[confirm, "txt_sparse_p"] = np.mean(confirm_folds, axis=0)
    assert a.iloc[fit].equals(b.iloc[fit])
    cats = ["category_id", "listing_type_id", "buying_mode", "state",
            "ship_mode", "seller_id_cat"]
    probs, iterations = {}, {}
    for name, frame in (("full", a), ("fold_avg", b)):
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=.07,
                                l2_leaf_reg=6., random_seed=SEED, thread_count=7,
                                verbose=False, loss_function="Logloss")
        cb.fit(frame.iloc[fit], y[fit], cat_features=cats,
               eval_set=(frame.iloc[tune], y[tune]),
               early_stopping_rounds=100, use_best_model=True)
        probs[name] = (cb.predict_proba(frame.iloc[tune])[:, 1],
                       cb.predict_proba(frame.iloc[confirm])[:, 1])
        iterations[name] = cb.best_iteration_
        print(name, cb.best_iteration_, flush=True)
    grid = np.arange(.44, .561, .005)
    thresholds = {name: float(max(grid, key=lambda t: accuracy_score(y[tune], p[0] >= t)))
                  for name, p in probs.items()}
    pred = {name: p[1] >= thresholds[name] for name, p in probs.items()}
    delta = (pred["fold_avg"] == y[confirm]).astype(int) - \
            (pred["full"] == y[confirm]).astype(int)
    seen = np.isin(sellers[confirm], sellers[fit])
    pairs = pd.DataFrame({"item_id": ids[confirm], "seller_id": sellers[confirm],
                          "is_seen": seen.astype(int), "y_true": y[confirm],
                          "p_full": probs["full"][1], "p_fold_avg": probs["fold_avg"][1],
                          "pred_full": pred["full"].astype(int),
                          "pred_fold_avg": pred["fold_avg"].astype(int), "delta": delta})
    path = OUT / f"paired_{SEED}.csv"
    pairs.to_csv(path, index=False)
    by = pairs.groupby("seller_id").delta.sum().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([by[rng.integers(len(by), size=len(by))].sum() for _ in range(3000)])
    report = {"seed": SEED, "fit_tune_confirm": [len(fit), len(tune), len(confirm)],
              "input_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "confirm_ids_sha256": hashlib.sha256(ids[confirm].astype("U").tobytes()).hexdigest(),
              "best_iterations": iterations, "tune_thresholds": thresholds,
              "accuracy": {k: float(accuracy_score(y[confirm], v)) for k, v in pred.items()},
              "logloss": {k: float(log_loss(y[confirm], p[1])) for k, p in probs.items()},
              "net": int(delta.sum()),
              "fixes_breaks": [int((delta == 1).sum()), int((delta == -1).sum())],
              "common05_net": int(((probs["fold_avg"][1] >= .5) == y[confirm]).sum() -
                                  ((probs["full"][1] >= .5) == y[confirm]).sum()),
              "seen_new_net": {str(i): int(delta[seen == bool(i)].sum()) for i in (0, 1)},
              "seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
              "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / f"report_{SEED}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
