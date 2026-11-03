"""Seller-segment threshold ablation on the historical strong text+seller model.

Train only on genuine train labels. Both global and segment thresholds are selected
on tune, then frozen for confirm. Seeds are historical reused development splits.
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

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261109
ZIP = Path(os.environ["AP_ZIP"])
OUT = ROOT / "results" / "segment_threshold_historical"
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
    oof, p_tune_text, p_confirm_text = text_oof(xf, xt, xv, y, fit, sellers, 3.)
    a["txt_sparse_p"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof, p_tune_text, p_confirm_text)):
        a.loc[idx, "txt_sparse_p"] = pred
    a["seller_id_cat"] = cat.seller_id.astype(str)
    cats = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]
    model = CatBoostClassifier(iterations=900, depth=6, learning_rate=.07,
                               l2_leaf_reg=6., random_seed=SEED, thread_count=7,
                               verbose=False, loss_function="Logloss")
    model.fit(a.iloc[fit], y[fit], cat_features=cats,
              eval_set=(a.iloc[tune], y[tune]), early_stopping_rounds=100,
              use_best_model=True)
    pt = model.predict_proba(a.iloc[tune])[:, 1]
    pc = model.predict_proba(a.iloc[confirm])[:, 1]
    seen_t = np.isin(sellers[tune], sellers[fit])
    seen_c = np.isin(sellers[confirm], sellers[fit])
    grid = np.arange(.40, .601, .005)
    global_t = max(grid, key=lambda t: accuracy_score(y[tune], pt >= t))
    segment_t = {}
    for key, mask in ((0, ~seen_t), (1, seen_t)):
        segment_t[key] = max(grid, key=lambda t: accuracy_score(y[tune][mask], pt[mask] >= t))
    pred_global = (pc >= global_t).astype(int)
    pred_segment = (pc >= np.where(seen_c, segment_t[1], segment_t[0])).astype(int)
    delta = (pred_segment == y[confirm]).astype(int) - (pred_global == y[confirm]).astype(int)
    pairs = pd.DataFrame({"item_id": ids[confirm], "seller_id": sellers[confirm],
                          "is_seen": seen_c.astype(int), "y_true": y[confirm],
                          "p_historical": pc, "pred_global": pred_global,
                          "pred_segment": pred_segment, "delta_segment": delta})
    path = OUT / f"paired_{SEED}.csv"
    pairs.to_csv(path, index=False)
    by = pairs.groupby("seller_id").delta_segment.sum().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([by[rng.integers(0, len(by), len(by))].sum() for _ in range(3000)])
    report = {"seed": SEED, "fit_tune_confirm": [len(fit), len(tune), len(confirm)],
              "train_zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "confirm_ids_sha256": hashlib.sha256(ids[confirm].astype("U").tobytes()).hexdigest(),
              "threshold_global": float(global_t), "threshold_new": float(segment_t[0]),
              "threshold_seen": float(segment_t[1]),
              "tune_global_accuracy": float(accuracy_score(y[tune], pt >= global_t)),
              "tune_segment_accuracy": float(accuracy_score(y[tune], pt >= np.where(seen_t, segment_t[1], segment_t[0]))),
              "confirm_global_accuracy": float(accuracy_score(y[confirm], pred_global)),
              "confirm_segment_accuracy": float(accuracy_score(y[confirm], pred_segment)),
              "confirm_logloss": float(log_loss(y[confirm], pc)),
              "net": int(delta.sum()), "fixes": int((delta == 1).sum()),
              "breaks": int((delta == -1).sum()),
              "seen_new": {str(k): {"n": int((seen_c == k).sum()),
                                       "net": int(delta[seen_c == k].sum())} for k in (0, 1)},
              "seller_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
              "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "caveat": "Both confirm splits have been reused for development; shared feature extractor includes pre-split unsupervised aggregates."}
    (OUT / f"report_{SEED}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
