"""Refit Full 70k Complementary Ensemble (CatBoost + LightGBM).

Leverages existing full 70k Pipeline A CatBoost predictions (test_probs_pipeline_A.npy),
trains full 70k LightGBM (leaf-wise) on the exact same 101 deep features + leak-free TE + txt_sparse_p,
and blends with frozen tune ensemble weights (w_cb=0.75, w_lgb=0.25, th=0.48).

Produces:
- results/complementary_20261030/test_probs_complementary.npy
- results/complementary_20261030/complementary_test_manifest_vs_s48.json
"""
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import (
    load_dataset as load_train_dataset, extract_deep_features
)

TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"
RESULTS_DIR = ROOT / "results" / "complementary_20261030"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.time()
    print("Loading 70k train dataset...", flush=True)
    rows_tr, ids_tr, sellers_tr, y_tr = load_train_dataset()
    
    print("Loading 30k test dataset...", flush=True)
    rows_te = [json.loads(line) for line in open(TEST_JSON, "r", encoding="utf-8")]
    ids_te = np.array([str(r["id"]) for r in rows_te])
    sellers_te = np.array([str(r["seller_id"]) for r in rows_te])
    
    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = {int(k): v for k, v in json.load(f).items()}
        
    print("Extracting deep features for Train (70k) and Test (30k)...", flush=True)
    all_rows = rows_tr + rows_te
    F_all, cat_all, titles_all, warranties_all, combos_all = extract_deep_features(all_rows)
    
    F_tr = F_all.iloc[:len(rows_tr)].copy().reset_index(drop=True)
    F_te = F_all.iloc[len(rows_tr):].copy().reset_index(drop=True)
    
    cat_tr = cat_all.iloc[:len(rows_tr)].copy().reset_index(drop=True)
    cat_te = cat_all.iloc[len(rows_tr):].copy().reset_index(drop=True)
    
    combos_tr = combos_all[:len(rows_tr)]
    combos_te = combos_all[len(rows_tr):]
    titles_tr = titles_all[:len(rows_tr)]
    titles_te = titles_all[len(rows_tr):]
    
    # 1. Target Encodings (alpha=5.0)
    print("Computing Target Encodings (alpha=5.0)...", flush=True)
    cat_cols = ["seller_id", "category_id", "city"]
    alpha = 5.0
    global_mean = float(y_tr.mean())
    
    oof_te_tr = {col: np.full(len(rows_tr), global_mean, dtype=np.float32) for col in cat_cols}
    gkf = GroupKFold(n_splits=4)
    for tr_idx, val_idx in gkf.split(cat_tr, y_tr, groups=sellers_tr):
        for col in cat_cols:
            cats = cat_tr[col].values
            tr_cats = cats[tr_idx]
            tr_y = y_tr[tr_idx]
            sums = defaultdict(float)
            counts = defaultdict(int)
            for c, y in zip(tr_cats, tr_y):
                sums[c] += y
                counts[c] += 1
            oof_te_tr[col][val_idx] = np.array([
                (sums[c] + alpha * global_mean) / (counts[c] + alpha)
                for c in cats[val_idx]
            ], dtype=np.float32)
            
    te_test = {}
    for col in cat_cols:
        cats = cat_tr[col].values
        sums = defaultdict(float)
        counts = defaultdict(int)
        for c, y in zip(cats, y_tr):
            sums[c] += y
            counts[c] += 1
        te_test[col] = np.array([
            (sums[c] + alpha * global_mean) / (counts[c] + alpha) if c in counts else global_mean
            for c in cat_te[col].values
        ], dtype=np.float32)
        
    for col in cat_cols:
        F_tr[f"te_{col}"] = oof_te_tr[col]
        F_te[f"te_{col}"] = te_test[col]
        
    # 2. txt_sparse_p
    print("Computing OOF and Test txt_sparse_p via TF-IDF + SGDClassifier...", flush=True)
    vec_combo = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vec_title = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    
    X_tr_combo = vec_combo.fit_transform(combos_tr)
    X_tr_title = vec_title.fit_transform(titles_tr)
    X_tr_text = sp.hstack([X_tr_combo, X_tr_title], format="csr")
    
    X_te_combo = vec_combo.transform(combos_te)
    X_te_title = vec_title.transform(titles_te)
    X_te_text = sp.hstack([X_te_combo, X_te_title], format="csr")
    
    oof_txt_tr = np.full(len(rows_tr), global_mean, dtype=np.float32)
    for tr_idx, val_idx in gkf.split(X_tr_text, y_tr, groups=sellers_tr):
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=20, random_state=2026)
        clf.fit(X_tr_text[tr_idx], y_tr[tr_idx])
        oof_txt_tr[val_idx] = clf.predict_proba(X_tr_text[val_idx])[:, 1]
        
    clf_full = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=20, random_state=2026)
    clf_full.fit(X_tr_text, y_tr)
    txt_te = clf_full.predict_proba(X_te_text)[:, 1].astype(np.float32)
    
    F_tr["txt_sparse_p"] = oof_txt_tr
    F_te["txt_sparse_p"] = txt_te
    
    # LightGBM categorical type
    for c in CAT_FEATURES:
        F_tr[c] = F_tr[c].astype("category")
        F_te[c] = F_te[c].astype("category")
        
    # 3. Train Full 70k LightGBM
    print("Training Leaf-Wise LightGBM on Full 70k...", flush=True)
    lgbm = lgb.LGBMClassifier(
        n_estimators=900,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=2026,
        n_jobs=8,
        verbose=-1
    )
    lgbm.fit(F_tr, y_tr)
    p_lgb_test = lgbm.predict_proba(F_te)[:, 1].astype(np.float32)
    
    # 4. Load Pipeline A test probabilities
    p_pipe_a_path = ROOT / "results" / "benchmark_20261030" / "test_probs_pipeline_A.npy"
    assert p_pipe_a_path.exists(), "Pipeline A test probs not found"
    p_cb_test = np.load(p_pipe_a_path).astype(np.float32)
    
    # 5. Blend with frozen tune weights
    w_cb = 0.75
    w_lgb = 0.25
    th = 0.48
    
    p_comp_test = (w_cb * p_cb_test + w_lgb * p_lgb_test).astype(np.float32)
    
    p_comp_path = RESULTS_DIR / "test_probs_complementary.npy"
    np.save(p_comp_path, p_comp_test)
    comp_sha = compute_sha256(p_comp_path)
    print(f"Saved complementary test probabilities to {p_comp_path} (SHA256: {comp_sha})")
    
    # 6. Analyze flips vs Submission 48
    pred_test = np.where(p_comp_test >= th, "used", "new")
    test_ids = np.arange(1, 30001)
    s48_preds = np.array([s48_dict[i] for i in test_ids])
    train_sellers_set = set(sellers_tr)
    is_seen = np.isin(sellers_te, list(train_sellers_set))
    
    flips = (pred_test != s48_preds)
    n_flips = int(flips.sum())
    
    flip_details = []
    for idx in np.where(flips)[0]:
        t_id = int(test_ids[idx])
        flip_details.append({
            "id": t_id,
            "item_id": ids_te[idx],
            "seller_id": sellers_te[idx],
            "is_seen": int(is_seen[idx]),
            "s48_pred": s48_preds[idx],
            "comp_pred": pred_test[idx],
            "p_comp_used": float(p_comp_test[idx]),
            "margin": float(abs(p_comp_test[idx] - th))
        })
        
    manifest = {
        "model": "Complementary Ensemble (0.75 Deep CatBoost + 0.25 Leaf-Wise LightGBM)",
        "train_rows": len(rows_tr),
        "test_rows": len(rows_te),
        "weights": {"catboost": w_cb, "lightgbm": w_lgb},
        "threshold": th,
        "n_flips_vs_s48": n_flips,
        "flips_seen": int(sum(1 for f in flip_details if f["is_seen"] == 1)),
        "flips_new": int(sum(1 for f in flip_details if f["is_seen"] == 0)),
        "flips_new_to_used": int(sum(1 for f in flip_details if f["s48_pred"] == "new" and f["comp_pred"] == "used")),
        "flips_used_to_new": int(sum(1 for f in flip_details if f["s48_pred"] == "used" and f["comp_pred"] == "new")),
        "test_probs_sha256": comp_sha,
        "elapsed_seconds": round(time.time() - t0, 1)
    }
    
    man_path = RESULTS_DIR / "complementary_test_manifest_vs_s48.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {man_path}")
    print(f"Complementary Flips vs S48: {n_flips} (Seen: {manifest['flips_seen']}, New: {manifest['flips_new']})")

if __name__ == "__main__":
    main()
