"""Paired validation of unlabeled 70k vs 100k extractor context.

Only 70k original train labels are used. Test rows enter only unsupervised
feature aggregation in the 100k arm; neither test labels nor public scores
enter model selection or evaluation.
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

SEED = int(os.environ.get("AP_SEED", "20261130"))
ZIP = Path(os.environ["AP_ZIP"])
OUT = ROOT / "results/refit_context_transport"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with zipfile.ZipFile(ZIP) as z:
        train = [json.loads(line) for line in z.open("train_data.jsonlines")]
        test = [json.loads(line) for line in z.open("test_data.jsonlines")]
    assert len(train) == 70000 and len(test) == 30000
    assert all("condition" not in row for row in test)
    n = len(train)
    y = np.array([r["condition"] == "used" for r in train], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in train])
    ids = np.array([str(r["id"]) for r in train])
    fit, tune, confirm = split_testlike(sellers, SEED)
    a70, cat70, titles70, _, combos70 = extract_deep_features(train)
    a100, cat100, titles100, _, combos100 = extract_deep_features(train + test)
    a100 = a100.iloc[:n].copy()
    cat100 = cat100.iloc[:n].copy()
    assert titles70 == titles100[:n] and combos70 == combos100[:n]
    assert cat70.equals(cat100)
    changed = {}
    for col in a70.columns:
        if col in ("category_id", "listing_type_id", "buying_mode", "state", "ship_mode"):
            assert a70[col].equals(a100[col])
            continue
        d = np.abs(a70[col].to_numpy(dtype=float) - a100[col].to_numpy(dtype=float))
        count = int((d > 1e-8).sum())
        if count:
            changed[col] = {"rows_changed": count, "mean_abs_change": float(d.mean()),
                            "max_abs_change": float(d.max())}
    for frame, cat in ((a70, cat70), (a100, cat100)):
        make_te(frame, cat, y, fit, tune, confirm, sellers)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                           sublinear_tf=True, strip_accents="unicode", lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           max_features=150000, sublinear_tf=True)
    xf = sp.hstack([word.fit_transform([combos70[i] for i in fit]),
                    char.fit_transform([titles70[i] for i in fit])], format="csr")
    xt = sp.hstack([word.transform([combos70[i] for i in tune]),
                    char.transform([titles70[i] for i in tune])], format="csr")
    xv = sp.hstack([word.transform([combos70[i] for i in confirm]),
                    char.transform([titles70[i] for i in confirm])], format="csr")
    oof, pt, pv = text_oof(xf, xt, xv, y, fit, sellers, 3.)
    for frame, cat in ((a70, cat70), (a100, cat100)):
        frame["txt_sparse_p"] = np.nan
        for ix, p in ((fit, oof), (tune, pt), (confirm, pv)):
            frame.loc[ix, "txt_sparse_p"] = p
        frame["seller_id_cat"] = cat.seller_id.astype(str)
    assert a70.columns.equals(a100.columns)
    cats = ["category_id", "listing_type_id", "buying_mode", "state",
            "ship_mode", "seller_id_cat"]
    probs, iterations = {}, {}
    for name, frame in (("ctx70", a70), ("ctx100", a100)):
        model = CatBoostClassifier(iterations=900, depth=6, learning_rate=.07,
                                   l2_leaf_reg=6., random_seed=SEED,
                                   thread_count=7, verbose=False, loss_function="Logloss")
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
    delta = (pred["ctx100"] == y[confirm]).astype(int) - \
            (pred["ctx70"] == y[confirm]).astype(int)
    seen = np.isin(sellers[confirm], sellers[fit])
    pairs = pd.DataFrame({"item_id": ids[confirm], "seller_id": sellers[confirm],
                          "is_seen": seen.astype(int), "y_true": y[confirm],
                          "p_ctx70": probs["ctx70"][1], "p_ctx100": probs["ctx100"][1],
                          "pred_ctx70": pred["ctx70"].astype(int),
                          "pred_ctx100": pred["ctx100"].astype(int), "delta": delta})
    path = OUT / f"paired_{SEED}.csv"
    pairs.to_csv(path, index=False)
    by = pairs.groupby("seller_id").delta.sum().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([by[rng.integers(0, len(by), len(by))].sum()
                     for _ in range(3000)])
    report = {"seed": SEED, "fit_tune_confirm": [len(fit), len(tune), len(confirm)],
              "train_zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "confirm_ids_sha256": hashlib.sha256(ids[confirm].astype("U").tobytes()).hexdigest(),
              "changed_unsupervised_features": changed,
              "iterations": iterations, "thresholds_tune": thresholds,
              "accuracy": {k: float(accuracy_score(y[confirm], v)) for k, v in pred.items()},
              "logloss": {k: float(log_loss(y[confirm], p[1])) for k, p in probs.items()},
              "net": int(delta.sum()),
              "fixes_breaks": [int((delta == 1).sum()), int((delta == -1).sum())],
              "common05_net": int(((probs["ctx100"][1] >= .5) == y[confirm]).sum() -
                                  ((probs["ctx70"][1] >= .5) == y[confirm]).sum()),
              "seen_new_net": {str(i): int(delta[seen == bool(i)].sum()) for i in (0, 1)},
              "seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
              "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / f"report_{SEED}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "changed_unsupervised_features"},
                     indent=2), flush=True)
    print("changed_feature_columns", len(changed), flush=True)


if __name__ == "__main__":
    main()
