"""Predeclared equal-weight bag of three CatBoost seeds on strong text+seller features.

Only original 70k labels. Same seller-group fit/tune/confirm for base and bag.
The first model is the historical single-model architecture; bag weights fixed.
Each model early-stops on tune; thresholds selected on tune only.
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

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261124
ZIP = Path(os.environ["AP_ZIP"])
OUT = ROOT / "results" / "catboost_bag3"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with zipfile.ZipFile(ZIP) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000
    y = np.array([r["condition"] == "used" for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])
    fit, tune, confirm = split_testlike(sellers, SEED)
    assert not (set(ids[fit]) & set(ids[tune]) or set(ids[fit]) & set(ids[confirm])
                or set(ids[tune]) & set(ids[confirm]))
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
    oof, p_tune_text, p_confirm_text = text_oof(xf, xt, xv, y, fit, sellers, 3.)
    a["txt_sparse_p"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof, p_tune_text, p_confirm_text)):
        a.loc[idx, "txt_sparse_p"] = pred
    a["seller_id_cat"] = cat.seller_id.astype(str)
    cats = ["category_id", "listing_type_id", "buying_mode", "state",
            "ship_mode", "seller_id_cat"]
    seeds = [SEED, SEED + 997, SEED + 1994]
    tune_probs, confirm_probs, iterations = [], [], []
    for seed in seeds:
        model = CatBoostClassifier(iterations=900, depth=6, learning_rate=.07,
                                   l2_leaf_reg=6., random_seed=seed, thread_count=7,
                                   verbose=False, loss_function="Logloss")
        model.fit(a.iloc[fit], y[fit], cat_features=cats,
                  eval_set=(a.iloc[tune], y[tune]), early_stopping_rounds=100,
                  use_best_model=True)
        iterations.append(model.best_iteration_)
        tune_probs.append(model.predict_proba(a.iloc[tune])[:, 1])
        confirm_probs.append(model.predict_proba(a.iloc[confirm])[:, 1])
        print(f"finished model seed={seed} iter={model.best_iteration_}", flush=True)
    pt_single, pc_single = tune_probs[0], confirm_probs[0]
    pt_bag = np.mean(tune_probs, axis=0)
    pc_bag = np.mean(confirm_probs, axis=0)
    grid = np.arange(.44, .561, .005)
    th_single = max(grid, key=lambda t: accuracy_score(y[tune], pt_single >= t))
    th_bag = max(grid, key=lambda t: accuracy_score(y[tune], pt_bag >= t))
    pred_single = (pc_single >= th_single).astype(int)
    pred_bag = (pc_bag >= th_bag).astype(int)
    seen = np.isin(sellers[confirm], sellers[fit]).astype(int)
    delta = (pred_bag == y[confirm]).astype(int) - (pred_single == y[confirm]).astype(int)
    pairs = pd.DataFrame({"item_id": ids[confirm], "seller_id": sellers[confirm],
                          "is_seen": seen, "y_true": y[confirm],
                          "p_single": pc_single, "p_bag3": pc_bag,
                          "pred_single": pred_single, "pred_bag3": pred_bag,
                          "delta_bag3": delta})
    path = OUT / f"paired_{SEED}.csv"
    pairs.to_csv(path, index=False)
    by = pairs.groupby("seller_id").delta_bag3.sum().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([by[rng.integers(0, len(by), len(by))].sum() for _ in range(3000)])
    report = {"seed": SEED, "model_seeds": seeds, "bag_weights": [1/3]*3,
              "fit_tune_confirm": [len(fit), len(tune), len(confirm)],
              "train_zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "confirm_ids_sha256": hashlib.sha256(ids[confirm].astype("U").tobytes()).hexdigest(),
              "best_iterations": iterations,
              "thresholds_single_bag": [float(th_single), float(th_bag)],
              "tune_accuracy_single_bag": [float(accuracy_score(y[tune], pt_single >= th_single)),
                                           float(accuracy_score(y[tune], pt_bag >= th_bag))],
              "confirm_accuracy_single_bag": [float(accuracy_score(y[confirm], pred_single)),
                                              float(accuracy_score(y[confirm], pred_bag))],
              "confirm_logloss_single_bag": [float(log_loss(y[confirm], pc_single)),
                                             float(log_loss(y[confirm], pc_bag))],
              "net": int(delta.sum()), "fixes": int((delta == 1).sum()),
              "breaks": int((delta == -1).sum()),
              "common05_net": int(((pc_bag >= .5) == y[confirm]).sum()
                                  - ((pc_single >= .5) == y[confirm]).sum()),
              "seen_new": {str(k): {"n": int((seen == k).sum()),
                                        "net": int(delta[seen == k].sum())} for k in (0, 1)},
              "seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
              "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "caveat": "Pre-split unsupervised extractor aggregates are shared by both arms; confirm is not a public test."}
    (OUT / f"report_{SEED}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
