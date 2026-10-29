"""Refit Strong Deep Ensemble (LightGBM + CatBoost) on Full 70k Train Records.

Produces:
1. results/revision_submissions/strong_ensemble_test_probs.npy
2. results/revision_submissions/manifest_flips_strong_ensemble_vs_s48.json
3. submissions/20261029/submission_71_strong_ensemble_candidate.csv (candidate for Codex, unsubmitted)

Protocol:
- 101 Deep features + 4-fold GroupKFold OOF TEs + 4-fold GroupKFold OOF txt_sparse_p on full 70k.
- CatBoost 900 trees (w=0.68) + LightGBM 700 trees (w=0.32) blended with frozen tune parameters.
- Evaluates flip differences against Submission 48 (best known public .94633).
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
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys_path = str(ROOT / "src" / "revision_submissions")
if sys_path not in sys.path:
    sys.path.append(sys_path)

from strong_ensemble_audit_and_refit_20261029 import (
    dims, extract_deep_features, load_dataset, lst, picnum
)

TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"
RESULTS_DIR = ROOT / "results" / "revision_submissions"
SUBMISSIONS_DIR = ROOT / "submissions" / "20261029"


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    t_start = time.time()
    rows_tr, ids_tr, sellers_tr, y_tr = load_dataset()

    print("Loading test dataset...", flush=True)
    rows_te = [json.loads(line) for line in open(TEST_JSON, "r", encoding="utf-8")]
    ids_te = np.array([str(r["id"]) for r in rows_te])
    sellers_te = np.array([str(r["seller_id"]) for r in rows_te])
    titles_te = [r.get("title") or "" for r in rows_te]
    warranties_te = [r.get("warranty") or "" for r in rows_te]
    combos_te = [f"{t} {w}".strip() for t, w in zip(titles_te, warranties_te)]

    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = json.load(f)

    # 1. Feature Extraction on train and test
    print("\nExtracting deep features for Train and Test...", flush=True)
    all_rows = rows_tr + rows_te
    F_all, cat_all, titles_all, warranties_all, combos_all = extract_deep_features(all_rows)

    F_tr = F_all.iloc[:len(rows_tr)].copy().reset_index(drop=True)
    F_te = F_all.iloc[len(rows_tr):].copy().reset_index(drop=True)

    cat_tr = cat_all.iloc[:len(rows_tr)].copy().reset_index(drop=True)
    cat_te = cat_all.iloc[len(rows_tr):].copy().reset_index(drop=True)

    combos_tr = combos_all[:len(rows_tr)]
    titles_tr = titles_all[:len(rows_tr)]

    # 2. Target Encodings on 70k Train via 4-fold GroupKFold by seller
    print("\nComputing 4-fold GroupKFold OOF Target Encodings on 70k...", flush=True)
    cat_cols = ["seller_id", "category_id", "city"]
    alpha = 10.0
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

    # Full stats for test
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

    # 3. Text Feature (txt_sparse_p) on 70k Train and Test
    print("\nComputing txt_sparse_p on 70k Train (OOF) and Test...", flush=True)
    vec_combo = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000, sublinear_tf=True, strip_accents="unicode")
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=150000, sublinear_tf=True, strip_accents="unicode")

    X_w_tr = vec_combo.fit_transform(combos_tr)
    X_c_tr = vec_char.fit_transform(titles_tr)
    X_sp_tr = sp.hstack([X_w_tr, X_c_tr], format="csr")

    X_w_te = vec_combo.transform(combos_te)
    X_c_te = vec_char.transform(titles_te)
    X_sp_te = sp.hstack([X_w_te, X_c_te], format="csr")

    oof_txt_tr = np.zeros(len(rows_tr), dtype=np.float32)
    for tr_idx, val_idx in gkf.split(rows_tr, y_tr, sellers_tr):
        lr = LogisticRegression(C=3.0, max_iter=250, solver="liblinear", random_state=42)
        lr.fit(X_sp_tr[tr_idx], y_tr[tr_idx])
        oof_txt_tr[val_idx] = lr.predict_proba(X_sp_tr[val_idx])[:, 1]

    lr_full = LogisticRegression(C=3.0, max_iter=300, solver="liblinear", random_state=42).fit(X_sp_tr, y_tr)
    p_txt_te = lr_full.predict_proba(X_sp_te)[:, 1].astype(np.float32)

    # 4. Assemble DataFrames
    for col in cat_cols:
        F_tr[f"te_{col}"] = oof_te_tr[col]
        F_te[f"te_{col}"] = te_test[col]
    F_tr["txt_sparse_p"] = oof_txt_tr
    F_te["txt_sparse_p"] = p_txt_te

    CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]
    for c in CAT_FEATURES:
        F_tr[c] = F_tr[c].astype(str)
        F_te[c] = F_te[c].astype(str)

    # 5. Fit Deep CatBoost on Full 70k
    print("\nFitting Deep CatBoost on Full 70k (900 trees)...", flush=True)
    t0 = time.time()
    cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                            random_seed=42, thread_count=12, verbose=100)
    cb.fit(Pool(F_tr, y_tr, cat_features=CAT_FEATURES))
    print(f"CatBoost 70k fitted in {time.time()-t0:.2f}s", flush=True)
    p_cb_test = cb.predict_proba(Pool(F_te, cat_features=CAT_FEATURES))[:, 1]

    # 6. Fit Deep LightGBM on Full 70k
    print("\nFitting Deep LightGBM on Full 70k (700 trees)...", flush=True)
    t0 = time.time()
    F_tr_lgb = F_tr.copy()
    F_te_lgb = F_te.copy()
    for c in CAT_FEATURES:
        F_tr_lgb[c] = F_tr_lgb[c].astype("category")
        F_te_lgb[c] = F_te_lgb[c].astype("category")

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 700,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(F_tr_lgb, y_tr)
    print(f"LightGBM 70k fitted in {time.time()-t0:.2f}s", flush=True)
    p_lgb_test = model_lgb.predict_proba(F_te_lgb)[:, 1]

    # 7. Blend using Frozen Tune Parameters (w_LGBM = 0.32, w_CB = 0.68)
    p_ens_test = 0.32 * p_lgb_test + 0.68 * p_cb_test
    test_probs_path = RESULTS_DIR / "strong_ensemble_test_probs.npy"
    np.save(test_probs_path, p_ens_test)
    print(f"\nSaved test probabilities to {test_probs_path} (mean p_used: {p_ens_test.mean():.4f})", flush=True)

    # 8. Compare against Submission 48 (.94633)
    train_sellers_set = set(sellers_tr)
    is_seen_te = np.array([s in train_sellers_set for s in sellers_te], dtype=bool)

    pred_ens_labels = ["used" if p >= 0.51 else "new" for p in p_ens_test]
    flips = []
    for i in range(30000):
        pid = i + 1
        s48_label = s48_dict[str(pid)]
        ens_label = pred_ens_labels[i]
        if ens_label != s48_label:
            flips.append({
                "ID": pid,
                "s48": s48_label,
                "ens": ens_label,
                "direction": f"{s48_label}->{ens_label}",
                "is_seen": int(is_seen_te[i]),
                "p_ens": float(p_ens_test[i]),
                "margin": float(abs(p_ens_test[i] - 0.50)),
                "price": float(rows_te[i].get("price") or 0),
                "title": rows_te[i].get("title"),
            })

    print(f"\nTotal Flips vs Submission 48: {len(flips)}", flush=True)
    n_seen_flips = sum(f["is_seen"] == 1 for f in flips)
    n_new_flips = sum(f["is_seen"] == 0 for f in flips)
    print(f"  In Seen Sellers: {n_seen_flips}")
    print(f"  In New Sellers:  {n_new_flips}")

    # Generate Candidate CSV (unsubmitted, for Codex's inspection)
    candidate_csv = SUBMISSIONS_DIR / "submission_71_strong_ensemble_candidate.csv"
    with open(candidate_csv, "w", newline="\n", encoding="utf-8") as f:
        f.write("ID,condition\n")
        for i in range(1, 30001):
            f.write(f"{i},{pred_ens_labels[i-1]}\n")
    sha_cand = compute_sha256(candidate_csv)
    print(f"Saved Candidate CSV: {candidate_csv} (SHA256: {sha_cand})", flush=True)

    # Generate Flips Manifest
    manifest = {
        "candidate_file": candidate_csv.name,
        "sha256": sha_cand,
        "baseline": "submission_48_best_known (.94633)",
        "total_flips": len(flips),
        "flips_seen_sellers": n_seen_flips,
        "flips_new_sellers": n_new_flips,
        "flips_used_to_new": sum(f["direction"] == "used->new" for f in flips),
        "flips_new_to_used": sum(f["direction"] == "new->used" for f in flips),
        "top_margin_flips": sorted(flips, key=lambda x: -x["margin"])[:20],
        "note": "Candidate model prepared for Codex inspection. Not submitted to Kaggle.",
    }
    manifest_path = RESULTS_DIR / "manifest_flips_strong_ensemble_vs_s48.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved Flips Manifest to {manifest_path}", flush=True)
    print(f"\nAll operations completed in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
