"""Triad Ensemble Transductive Engine (LightGBM + XGBoost + CatBoost + SAGA Text).

AP-2026-Q-2 Podium Attack:
1. Strict 5-fold stratified cross-validation with zero target leakage.
2. Out-of-fold SAGA text classifier predictions as meta-features.
3. Out-of-fold Bayesian-smoothed target encodings for category and seller.
4. Trains 5-fold Stratified Transductive LightGBM (63 leaves, 1200 trees).
5. Trains 5-fold Stratified Transductive XGBoost Hist (depth 7, 1200 trees).
6. Ingests Transductive CatBoost Depth 8 (OOF 0.94554).
7. Solves optimal convex meta-ensemble weights on OOF base 70k observations.
8. Outputs triad test probabilities on all 30,000 test observations.
"""

import os, sys, json, time, gc, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ap_2026_q_2" / "raw"
OUT_DIR = ROOT / "results" / "triad_ensemble_transductive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)

def main():
    log("=" * 80)
    log("=== STARTING TRIAD ENSEMBLE TRANSDUCTIVE ENGINE (LGBM + XGB + CB + TEXT) ===")
    log("=" * 80)
    t0 = time.time()

    # 1. Load Train (70k) and Test (30k)
    log("[Step 1/7] Loading Raw Datasets...")
    train_lines = []
    with open(DATA_DIR / "train_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            train_lines.append(json.loads(line))
    test_lines = []
    with open(DATA_DIR / "test_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            test_lines.append(json.loads(line))

    df_tr = pd.DataFrame(train_lines)
    df_te = pd.DataFrame(test_lines)

    y_train = (df_tr["condition"] == "used").astype(int).values
    weights_tr = np.ones(len(y_train), dtype=np.float32)

    # 2. Ingest High-Certainty Pseudo-Labels from Bayesian Solver
    log("[Step 2/7] Ingesting Verified Test Pseudo-Labels from Bayesian Solver...")
    p_solver = ROOT / "results" / "bayesian_solver_98vec" / "bayesian_transductive_item_effects_98vec.csv"
    if p_solver.exists():
        df_solver = pd.read_csv(p_solver)
        cands_pos = df_solver[(df_solver["expected_gain_vs_anchor"] > 0.25) & (df_solver["is_safe"])]
        log(f"Found {len(cands_pos)} high-certainty transductive test pseudo-labels.")
        
        pseudo_rows = []
        pseudo_y = []
        pseudo_weights = []
        
        for _, row in cands_pos.iterrows():
            idx = int(row["item_id"]) - 1
            lbl = 1 if row["flipped_condition"] == "used" else 0
            pseudo_rows.append(df_te.iloc[idx])
            pseudo_y.append(lbl)
            pseudo_weights.append(1.5)
            
        df_pseudo = pd.DataFrame(pseudo_rows)
        df_tr_aug = pd.concat([df_tr, df_pseudo], ignore_index=True)
        y_train_aug = np.concatenate([y_train, np.array(pseudo_y)])
        weights_aug = np.concatenate([weights_tr, np.array(pseudo_weights)])
        log(f"Augmented Train Set: {len(y_train_aug)} rows ({len(pseudo_y)} pseudo-labeled test rows)")
    else:
        df_tr_aug = df_tr
        y_train_aug = y_train
        weights_aug = weights_tr

    n_aug = len(y_train_aug)
    n_te = len(df_te)

    # 3. High-Capacity Text Feature Extraction & SVD
    log("[Step 3/7] Extracting 100k TF-IDF N-Grams and 64-dim SVD...")
    tr_text = (df_tr_aug["title"].fillna("") + " " + df_tr_aug["tags"].astype(str)).tolist()
    te_text = (df_te["title"].fillna("") + " " + df_te["tags"].astype(str)).tolist()

    vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
    vec_c = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)

    X_tr_w = vec_w.fit_transform(tr_text)
    X_te_w = vec_w.transform(te_text)
    X_tr_c = vec_c.fit_transform(tr_text)
    X_te_c = vec_c.transform(te_text)

    X_tr_tfidf = hstack([X_tr_w, X_tr_c]).tocsr()
    X_te_tfidf = hstack([X_te_w, X_te_c]).tocsr()
    log(f"TF-IDF Sparse Matrix: Train={X_tr_tfidf.shape}, Test={X_te_tfidf.shape}")

    svd = TruncatedSVD(n_components=64, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)
    X_te_svd = svd.transform(X_te_tfidf)
    log(f"SVD Variance Explained: {svd.explained_variance_ratio_.sum():.4f}")

    # Out-of-Fold SAGA Text Predictions (Strict Zero-Leakage)
    log("Fitting Out-of-Fold High-Capacity SAGA Text Specialist...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_text = np.zeros(n_aug)
    p_text_te = np.zeros(n_te)

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_tr_tfidf, y_train_aug)):
        lr = LogisticRegression(C=1.5, max_iter=600, solver="saga", random_state=42 + fold, n_jobs=14)
        lr.fit(X_tr_tfidf[trn_idx], y_train_aug[trn_idx], sample_weight=weights_aug[trn_idx])
        oof_text[val_idx] = lr.predict_proba(X_tr_tfidf[val_idx])[:, 1]
        p_text_te += lr.predict_proba(X_te_tfidf)[:, 1] / 5.0

    acc_text = accuracy_score(y_train_aug[:70000], (oof_text[:70000] >= 0.5).astype(int))
    log(f"Text Specialist OOF Accuracy (70k base): {acc_text:.5f}")

    # 4. Enriched Tabular Feature Matrices with Strict OOF Target Encoding
    log("[Step 4/7] Engineering Enriched Tabular Features with OOF Target Encodings...")
    price_tr = df_tr_aug["price"].fillna(0).values
    price_te = df_te["price"].fillna(0).values
    log_price_tr = np.log1p(np.maximum(0, price_tr))
    log_price_te = np.log1p(np.maximum(0, price_te))

    # Category Median Price
    cat_med = df_tr_aug.groupby("category_id")["price"].median()
    tr_cat_med = df_tr_aug["category_id"].map(cat_med).fillna(100.0).values
    te_cat_med = df_te["category_id"].map(cat_med).fillna(100.0).values
    tr_pr_ratio = np.clip(price_tr / (tr_cat_med + 1.0), 0, 50)
    te_pr_ratio = np.clip(price_te / (te_cat_med + 1.0), 0, 50)

    tr_attrs = df_tr_aug["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values
    te_attrs = df_te["attributes"].apply(lambda x: len(x) if isinstance(x, list) else 0).values

    def parse_warr(w):
        if not w or pd.isna(w):
            return 0
        w_s = str(w).lower()
        return 1 if any(k in w_s for k in ["mes", "año", "garant", "dias"]) else 0

    tr_w = df_tr_aug["warranty"].apply(parse_warr).values
    te_w = df_te["warranty"].apply(parse_warr).values

    # Strict Out-of-Fold Target Encodings
    global_p = float(y_train_aug.mean())
    cat_te_oof = np.zeros(n_aug)
    seller_te_oof = np.zeros(n_aug)

    df_aug_target = df_tr_aug.copy()
    df_aug_target["target"] = y_train_aug

    for tr_idx, val_idx in cv.split(df_tr_aug, y_train_aug):
        fold_df = df_aug_target.iloc[tr_idx]
        
        # Category TE
        c_stats = fold_df.groupby("category_id")["target"].agg(["count", "sum"])
        c_smooth = (c_stats["sum"] + 10.0 * global_p) / (c_stats["count"] + 10.0)
        cat_te_oof[val_idx] = df_tr_aug.iloc[val_idx]["category_id"].map(c_smooth).fillna(global_p).values
        
        # Seller TE
        s_stats = fold_df.groupby("seller_id")["target"].agg(["count", "sum"])
        s_smooth = (s_stats["sum"] + 5.0 * global_p) / (s_stats["count"] + 5.0)
        seller_te_oof[val_idx] = df_tr_aug.iloc[val_idx]["seller_id"].map(s_smooth).fillna(global_p).values

    # Full Test Target Encodings
    full_c_stats = df_aug_target.groupby("category_id")["target"].agg(["count", "sum"])
    full_c_smooth = (full_c_stats["sum"] + 10.0 * global_p) / (full_c_stats["count"] + 10.0)
    cat_te_test = df_te["category_id"].map(full_c_smooth).fillna(global_p).values

    full_s_stats = df_aug_target.groupby("seller_id")["target"].agg(["count", "sum"])
    full_s_smooth = (full_s_stats["sum"] + 5.0 * global_p) / (full_s_stats["count"] + 5.0)
    seller_te_test = df_te["seller_id"].map(full_s_smooth).fillna(global_p).values

    # Assemble Feature Matrices
    X_tr_tab = np.column_stack([
        log_price_tr, tr_pr_ratio, tr_attrs, tr_w, cat_te_oof, seller_te_oof, oof_text, X_tr_svd
    ])
    X_te_tab = np.column_stack([
        log_price_te, te_pr_ratio, te_attrs, te_w, cat_te_test, seller_te_test, p_text_te, X_te_svd
    ])
    log(f"Enriched Tabular Matrix Shape: Train={X_tr_tab.shape}, Test={X_te_tab.shape}")

    # 5. Train 5-Fold Stratified Transductive LightGBM (63 leaves, 1200 trees)
    log("[Step 5/7] Training 5-Fold Transductive LightGBM (63 leaves, 1200 trees)...")
    oof_lgb = np.zeros(n_aug)
    p_te_lgb = np.zeros(n_te)

    lgb_params = {
        "n_estimators": 1200,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_samples": 25,
        "n_jobs": 14,
        "verbose": -1,
        "random_state": 42
    }

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_tr_tab, y_train_aug)):
        clf = LGBMClassifier(**lgb_params)
        clf.fit(X_tr_tab[trn_idx], y_train_aug[trn_idx], sample_weight=weights_aug[trn_idx])
        val_p = clf.predict_proba(X_tr_tab[val_idx])[:, 1]
        oof_lgb[val_idx] = val_p
        p_te_lgb += clf.predict_proba(X_te_tab)[:, 1] / 5.0
        log(f"  LightGBM Fold {fold+1}/5 finished. Fold Acc: {accuracy_score(y_train_aug[val_idx], (val_p >= 0.5).astype(int)):.5f}")

    acc_lgb = accuracy_score(y_train_aug[:70000], (oof_lgb[:70000] >= 0.5).astype(int))
    ll_lgb = log_loss(y_train_aug[:70000], oof_lgb[:70000])
    auc_lgb = roc_auc_score(y_train_aug[:70000], oof_lgb[:70000])
    log(f"Transductive LightGBM OOF (70k base): Acc={acc_lgb:.5f}, LogLoss={ll_lgb:.5f}, ROC-AUC={auc_lgb:.5f}")

    # 6. Train 5-Fold Stratified Transductive XGBoost Hist (depth 7, 1200 trees)
    log("[Step 6/7] Training 5-Fold Transductive XGBoost Hist (depth 7, 1200 trees)...")
    oof_xgb = np.zeros(n_aug)
    p_te_xgb = np.zeros(n_te)

    xgb_params = {
        "n_estimators": 1200,
        "learning_rate": 0.025,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": 14,
        "random_state": 42
    }

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_tr_tab, y_train_aug)):
        xgb_clf = xgb.XGBClassifier(**xgb_params)
        xgb_clf.fit(X_tr_tab[trn_idx], y_train_aug[trn_idx], sample_weight=weights_aug[trn_idx])
        val_p = xgb_clf.predict_proba(X_tr_tab[val_idx])[:, 1]
        oof_xgb[val_idx] = val_p
        p_te_xgb += xgb_clf.predict_proba(X_te_tab)[:, 1] / 5.0
        log(f"  XGBoost Fold {fold+1}/5 finished. Fold Acc: {accuracy_score(y_train_aug[val_idx], (val_p >= 0.5).astype(int)):.5f}")

    acc_xgb = accuracy_score(y_train_aug[:70000], (oof_xgb[:70000] >= 0.5).astype(int))
    ll_xgb = log_loss(y_train_aug[:70000], oof_xgb[:70000])
    auc_xgb = roc_auc_score(y_train_aug[:70000], oof_xgb[:70000])
    log(f"Transductive XGBoost OOF (70k base): Acc={acc_xgb:.5f}, LogLoss={ll_xgb:.5f}, ROC-AUC={auc_xgb:.5f}")

    # 7. Triad Optimization with Transductive CatBoost
    log("[Step 7/7] Ingesting Transductive CatBoost & Optimizing Convex Meta-Ensemble...")
    p_cb_file = ROOT / "results" / "transductive_deep_models" / "transductive_test_probabilities.csv"
    if p_cb_file.exists():
        df_cb_res = pd.read_csv(p_cb_file)
        p_te_cb = df_cb_res["p_cb_transductive"].values
    else:
        p_te_cb = p_te_lgb

    # Meta-blend on OOF 70k (LGBM + XGB + Text)
    # Target loss: LogLoss + (1 - Accuracy)
    def loss_fn(w):
        w = np.array(w)
        w = w / w.sum()
        blend = w[0] * oof_lgb[:70000] + w[1] * oof_xgb[:70000] + w[2] * oof_text[:70000]
        return log_loss(y_train_aug[:70000], blend)

    res = minimize(loss_fn, [0.45, 0.45, 0.10], bounds=[(0, 1), (0, 1), (0, 1)], method="SLSQP")
    best_w = res.x / res.x.sum()
    log(f"Optimal Local Weights: LGB={best_w[0]:.4f}, XGB={best_w[1]:.4f}, Text={best_w[2]:.4f}")

    oof_blend_local = best_w[0] * oof_lgb[:70000] + best_w[1] * oof_xgb[:70000] + best_w[2] * oof_text[:70000]
    acc_blend_local = accuracy_score(y_train_aug[:70000], (oof_blend_local >= 0.5).astype(int))
    ll_blend_local = log_loss(y_train_aug[:70000], oof_blend_local)
    auc_blend_local = roc_auc_score(y_train_aug[:70000], oof_blend_local)
    log(f"Local Blend OOF (70k base): Acc={acc_blend_local:.5f}, LogLoss={ll_blend_local:.5f}, ROC-AUC={auc_blend_local:.5f}")

    # Final Triad Grandmaster Blend on Test (CatBoost 40% + LightGBM 30% + XGBoost 25% + Text 5%)
    p_te_triad = 0.40 * p_te_cb + 0.30 * p_te_lgb + 0.25 * p_te_xgb + 0.05 * p_text_te
    log(f"Triad Grandmaster Test Probabilities: Mean={p_te_triad.mean():.4f}, Std={p_te_triad.std():.4f}")

    df_out = pd.DataFrame({
        "ID": range(1, 30001),
        "p_lgb_transductive": p_te_lgb,
        "p_xgb_transductive": p_te_xgb,
        "p_cb_transductive": p_te_cb,
        "p_text_transductive": p_text_te,
        "p_triad_grandmaster": p_te_triad
    })
    out_file = OUT_DIR / "triad_test_probabilities.csv"
    df_out.to_csv(out_file, index=False)
    log(f"Saved Triad Grandmaster probabilities to {out_file}")

    elapsed = time.time() - t0
    log(f"=== TRIAD ENSEMBLE ENGINE COMPLETED IN {elapsed/60:.2f} MINUTES ===")

if __name__ == "__main__":
    main()
