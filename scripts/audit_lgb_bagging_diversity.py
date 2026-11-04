"""Audit LightGBM seed diversity with bagging_freq=0 vs bagging_freq=1."""

import json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_tree_diversity_clean_te import split_testlike, make_te_clean, generate_hist_text_oof
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
SEED = 20261144

with zipfile.ZipFile(ZIP_PATH) as z:
    rows = [json.loads(line) for line in z.open("train_data.jsonlines")]

y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
sellers = np.array([str(r["seller_id"]) for r in rows])
ids = np.array([str(r["id"]) for r in rows])
fit, tune, confirm = split_testlike(sellers, SEED)

a, cat, titles_raw, _, combos_raw = extract_deep_features(rows)
make_te_clean(a, cat, y, fit, tune, confirm, sellers)

oof_hist, pt_hist, pc_hist = generate_hist_text_oof(combos_raw, titles_raw, y, fit, tune, confirm, sellers, SEED)
a["txt_sparse_p"] = np.nan
for idx, pred in zip((fit, tune, confirm), (oof_hist, pt_hist, pc_hist)):
    a.loc[idx, "txt_sparse_p"] = pred

a["seller_id_cat"] = cat.seller_id.astype(str)
cat_cols = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]
for col in cat_cols:
    a[col] = a[col].astype("category")

cb_seeds = [SEED, SEED + 997, SEED + 1994]

results = {}
for bf in [0, 1]:
    d_fit = lgb.Dataset(a.loc[fit], label=y[fit])
    d_tune = lgb.Dataset(a.loc[tune], label=y[tune], reference=d_fit)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "bagging_freq": bf,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": 14
    }
    probs = []
    best_iters = []
    for s in cb_seeds:
        p = lgb_params.copy()
        p["random_state"] = s
        m = lgb.train(p, d_fit, num_boost_round=1000, valid_sets=[d_tune], callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iters.append(int(m.best_iteration))
        probs.append(m.predict(a.loc[confirm]))

    arr = np.array(probs)
    pearson = float(np.corrcoef(arr[0], arr[1])[0, 1])
    spearman = float(spearmanr(arr[0], arr[1]).statistic)
    diff = np.abs(arr[0] - arr[1])
    mean_diff = float(diff.mean())
    med_diff = float(np.median(diff))
    max_diff = float(diff.max())
    disagreements = int(((arr[0] >= 0.5) != (arr[1] >= 0.5)).sum())
    p_bag3 = np.mean(arr, axis=0)
    acc = float(accuracy_score(y[confirm], p_bag3 >= 0.5))
    ll = float(log_loss(y[confirm], p_bag3))

    results[f"bagging_freq_{bf}"] = {
        "best_iters": best_iters,
        "pearson_s0_s1": pearson,
        "spearman_s0_s1": spearman,
        "mean_diff": mean_diff,
        "med_diff": med_diff,
        "max_diff": max_diff,
        "binary_disagreements_at_05": disagreements,
        "bag3_acc_at_05": acc,
        "bag3_logloss": ll
    }

out_path = ROOT / "results" / "tree_diversity_clean_te" / "lgb_bagging_freq_comparison.json"
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
