"""Unified Super-Ensemble: High-Capacity Trees + Sparse Lexical Text + Margin Curation.

Author: Antigravity
Date: 2026-11-09
Policy: Strictly leak-free integration of subagent trees with high-dimensional lexical model.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(".")
DATA_DIR = ROOT / "data/ap_2026_q_2/raw"
OUT_DIR = ROOT / "results/dual_regime_20261109"


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("TRAINING UNIFIED SUPER-ENSEMBLE (TREES + LEXICAL TEXT + MARGIN CURATION)")
    print("=" * 80, flush=True)

    t0 = time.time()
    
    # 1. Load subagent predictions
    npz_p = OUT_DIR / "subagent_trees_predictions.npz"
    sub_data = np.load(npz_p)
    print(f"Loaded subagent trees predictions from {npz_p}")
    
    oof_cat = sub_data["oof_cat"]
    oof_lgb = sub_data["oof_lgb"]
    y_fit = sub_data["y_fit"]
    
    holdout_seen_cat = sub_data["holdout_seen_cat"]
    holdout_seen_lgb = sub_data["holdout_seen_lgb"]
    y_seen = sub_data["y_seen"]
    base_pred_seen = sub_data["base_pred_seen"]
    
    holdout_unseen_cat = sub_data["holdout_unseen_cat"]
    holdout_unseen_lgb = sub_data["holdout_unseen_lgb"]
    y_unseen = sub_data["y_unseen"]
    base_pred_unseen = sub_data["base_pred_unseen"]
    
    test_cat = sub_data["test_cat"]
    test_lgb = sub_data["test_lgb"]
    
    # 2. Extract fold-local Sparse Ridge Text predictions
    print("\nExtracting high-resolution Sparse Ridge Text features...")
    df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
    df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)
    
    with open(OUT_DIR / "frozen_dual_regime_indices.json", "r") as f:
        idx_data = json.load(f)
    fit_idx = np.array(idx_data["fit_idx"])
    seen_ho_idx = np.array(idx_data["seen_holdout_idx"])
    unseen_ho_idx = np.array(idx_data["unseen_holdout_idx"])
    
    df_fit = df_train.iloc[fit_idx].reset_index(drop=True)
    df_seen_ho = df_train.iloc[seen_ho_idx].reset_index(drop=True)
    df_unseen_ho = df_train.iloc[unseen_ho_idx].reset_index(drop=True)
    
    # Text representations
    titles_fit = df_fit["title"].fillna("").astype(str).str.lower()
    titles_seen = df_seen_ho["title"].fillna("").astype(str).str.lower()
    titles_unseen = df_unseen_ho["title"].fillna("").astype(str).str.lower()
    titles_test = df_test["title"].fillna("").astype(str).str.lower()
    
    # 5-fold cross-fitting for Ridge Text OOF on fit pool
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_ridge = np.zeros(len(df_fit), dtype=np.float32)
    
    vec_word = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
    vec_char = TfidfVectorizer(max_features=25000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df_fit, y_fit, df_fit["seller_id"].values)):
        # Fit vectorizers fold-local
        w_tr = vec_word.fit_transform(titles_fit.iloc[tr_idx])
        c_tr = vec_char.fit_transform(titles_fit.iloc[tr_idx])
        X_tr = sparse.hstack([w_tr, c_tr], format="csr")
        
        w_va = vec_word.transform(titles_fit.iloc[va_idx])
        c_va = vec_char.transform(titles_fit.iloc[va_idx])
        X_va = sparse.hstack([w_va, c_va], format="csr")
        
        m_rdg = RidgeClassifier(alpha=2.0, random_state=42)
        m_rdg.fit(X_tr, y_fit[tr_idx])
        scores = m_rdg.decision_function(X_va)
        oof_ridge[va_idx] = 1.0 / (1.0 + np.exp(-scores))
        
    print(f"OOF Ridge Text Acc: {accuracy_score(y_fit, (oof_ridge >= 0.5)):.5f} | LogLoss: {log_loss(y_fit, oof_ridge):.5f}")
    
    # Full fit for holdout and test
    w_fit = vec_word.fit_transform(titles_fit)
    c_fit = vec_char.fit_transform(titles_fit)
    X_fit_full = sparse.hstack([w_fit, c_fit], format="csr")
    
    m_rdg_full = RidgeClassifier(alpha=2.0, random_state=42)
    m_rdg_full.fit(X_fit_full, y_fit)
    
    # Holdout seen
    w_seen = vec_word.transform(titles_seen)
    c_seen = vec_char.transform(titles_seen)
    X_seen_sparse = sparse.hstack([w_seen, c_seen], format="csr")
    holdout_seen_ridge = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_seen_sparse)))
    
    # Holdout unseen
    w_unseen = vec_word.transform(titles_unseen)
    c_unseen = vec_char.transform(titles_unseen)
    X_unseen_sparse = sparse.hstack([w_unseen, c_unseen], format="csr")
    holdout_unseen_ridge = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_unseen_sparse)))
    
    # Test
    w_test = vec_word.transform(titles_test)
    c_test = vec_char.transform(titles_test)
    X_test_sparse = sparse.hstack([w_test, c_test], format="csr")
    test_ridge = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_test_sparse)))
    
    # 3. Train Regime-Conditioned Meta-Stackers
    print("\nTraining Unified Meta-Stackers (Seen vs Unseen)...")
    seller_counts = df_fit["seller_id"].value_counts()
    is_multi_seller = df_fit["seller_id"].map(seller_counts > 1).values
    
    meta_X_fit = np.column_stack([oof_cat, oof_lgb, oof_ridge])
    meta_seen = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    meta_seen.fit(meta_X_fit[is_multi_seller], y_fit[is_multi_seller])
    print(f"Meta-Seen Weights [CatB, LGBM, Ridge]: {meta_seen.coef_[0].round(3)} | Intercept: {meta_seen.intercept_[0]:.3f}")
    
    meta_unseen = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    meta_unseen.fit(meta_X_fit[~is_multi_seller], y_fit[~is_multi_seller])
    print(f"Meta-Unseen Weights [CatB, LGBM, Ridge]: {meta_unseen.coef_[0].round(3)} | Intercept: {meta_unseen.intercept_[0]:.3f}")
    
    # Predict Meta on Holdout
    meta_X_seen = np.column_stack([holdout_seen_cat, holdout_seen_lgb, holdout_seen_ridge])
    p_seen_meta = meta_seen.predict_proba(meta_X_seen)[:, 1]
    
    meta_X_unseen = np.column_stack([holdout_unseen_cat, holdout_unseen_lgb, holdout_unseen_ridge])
    p_unseen_meta = meta_unseen.predict_proba(meta_X_unseen)[:, 1]
    
    # Predict Meta on Test
    meta_X_test = np.column_stack([test_cat, test_lgb, test_ridge])
    test_sellers = df_test["seller_id"].values
    is_seen_in_fit = np.isin(test_sellers, df_fit["seller_id"].unique())
    p_test_meta = np.zeros(len(df_test), dtype=np.float32)
    p_test_meta[is_seen_in_fit] = meta_seen.predict_proba(meta_X_test[is_seen_in_fit])[:, 1]
    p_test_meta[~is_seen_in_fit] = meta_unseen.predict_proba(meta_X_test[~is_seen_in_fit])[:, 1]
    
    # Base predictions for holdout and test
    base_holdout_pred = np.concatenate([base_pred_seen, base_pred_unseen])
    y_holdout = np.concatenate([y_seen, y_unseen])
    sellers_holdout = np.concatenate([df_seen_ho["seller_id"].values, df_unseen_ho["seller_id"].values])
    base_test_pred = (test_lgb >= 0.5).astype(np.int8)
    
    # Clustered Bootstrap Setup
    unique_sellers = np.unique(sellers_holdout)
    seller_to_indices = {s: np.where(sellers_holdout == s)[0] for s in unique_sellers}
    N_pub = 20998
    gamma = 2.9557
    H_anchor = 19896
    
    # 4. Sweep over Margins
    margins = [0.00, 0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
    print("\n" + "=" * 98)
    print(f"{'Margin':6s} | {'Flips Test':10s} | {'Flips Ho':8s} | {'Fixes':5s} | {'Breaks':6s} | {'Net Ho':6s} | {'Prec%':6s} | {'Raw P5':7s} | {'Raw P50':7s} | {'Adj P5':7s} | {'P(G>=53)':8s}")
    print("=" * 98)
    
    rng = np.random.RandomState(42)
    B = 1000
    best_candidate_df = None
    best_adj_p5 = -999.0
    best_margin = 0.0
    best_stats = {}
    
    for m in margins:
        # Seen holdout candidate
        cand_seen = base_pred_seen.copy()
        flip_s = (np.abs(p_seen_meta - 0.5) >= m) & ((p_seen_meta >= 0.5).astype(np.int8) != base_pred_seen)
        cand_seen[flip_s] = (p_seen_meta[flip_s] >= 0.5).astype(np.int8)
        
        # Unseen holdout candidate
        cand_unseen = base_pred_unseen.copy()
        flip_u = (np.abs(p_unseen_meta - 0.5) >= m) & ((p_unseen_meta >= 0.5).astype(np.int8) != base_pred_unseen)
        cand_unseen[flip_u] = (p_unseen_meta[flip_u] >= 0.5).astype(np.int8)
        
        cand_holdout = np.concatenate([cand_seen, cand_unseen])
        
        # Test candidate
        cand_test = base_test_pred.copy()
        flip_t = (np.abs(p_test_meta - 0.5) >= m) & ((p_test_meta >= 0.5).astype(np.int8) != base_test_pred)
        cand_test[flip_t] = (p_test_meta[flip_t] >= 0.5).astype(np.int8)
        
        n_flips_t = int(np.sum(flip_t))
        n_flips_ho = int(np.sum(flip_s) + np.sum(flip_u))
        
        fixes = int(np.sum((base_holdout_pred != y_holdout) & (cand_holdout == y_holdout)))
        breaks = int(np.sum((base_holdout_pred == y_holdout) & (cand_holdout != y_holdout)))
        net_ho = fixes - breaks
        prec = (fixes / max(fixes + breaks, 1)) * 100.0
        
        # Clustered Bootstrap
        item_gains = ((base_holdout_pred != y_holdout) & (cand_holdout == y_holdout)).astype(np.int8) - \
                     ((base_holdout_pred == y_holdout) & (cand_holdout != y_holdout)).astype(np.int8)
                     
        boot_gains = np.zeros(B, dtype=np.float32)
        for b in range(B):
            s_sellers = rng.choice(unique_sellers, size=len(unique_sellers), replace=True)
            s_idx = np.concatenate([seller_to_indices[s] for s in s_sellers])
            boot_gains[b] = np.sum(item_gains[s_idx]) * (N_pub / len(s_idx))
            
        p5_raw = float(np.percentile(boot_gains, 5))
        p50_raw = float(np.percentile(boot_gains, 50))
        p95_raw = float(np.percentile(boot_gains, 95))
        
        half_width = (p95_raw - p5_raw) / 2.0
        half_width_adj = half_width * gamma
        p5_adj = p50_raw - half_width_adj
        
        sigma_adj = max(half_width_adj / 1.645, 0.01)
        z_to_53 = (53.0 - p50_raw) / sigma_adj
        prob_adj = float(1.0 - norm.cdf(z_to_53))
        
        print(f"{m:6.2f} | {n_flips_t:10d} | {n_flips_ho:8d} | {fixes:5d} | {breaks:6d} | {net_ho:+6d} | {prec:5.1f}% | {p5_raw:+7.1f} | {p50_raw:+7.1f} | {p5_adj:+7.1f} | {prob_adj*100:7.1f}%")
        
        if p5_adj > best_adj_p5:
            best_adj_p5 = p5_adj
            best_margin = m
            best_stats = {
                "margin": m, "test_flips": n_flips_t, "holdout_flips": n_flips_ho,
                "fixes": fixes, "breaks": breaks, "net_holdout": net_ho, "precision": prec,
                "raw_p5": p5_raw, "raw_p50": p50_raw, "raw_p95": p95_raw,
                "adj_p5": p5_adj, "prob_stress": prob_adj
            }
            best_candidate_df = pd.DataFrame({
                "ID": np.arange(1, len(df_test) + 1),
                "condition": np.where(cand_test == 1, "used", "new")
            })

    print("=" * 98)
    print(f"\nUnified Super-Ensemble Optimal Margin: delta = {best_margin:.2f}")
    print(f"Optimal Adjusted P5: {best_adj_p5:+.1f} hits | Raw P5: {best_stats['raw_p5']:+.1f} hits | Raw P50: {best_stats['raw_p50']:+.1f} hits")
    
    # Save candidate submission
    if best_candidate_df is not None:
        csv_p = OUT_DIR / "candidate_c8_unified_super_ensemble.csv"
        best_candidate_df.to_csv(csv_p, index=False)
        c_hash = sha256_file(csv_p)
        print(f"\nSaved Candidate C8: {csv_p}")
        print(f"SHA-256: {c_hash}")
        
    # Save report
    rep_p = OUT_DIR / "unified_super_ensemble_report.json"
    with open(rep_p, "w") as f:
        json.dump(best_stats, f, indent=2)
    print(f"Saved Report: {rep_p}")
    print(f"Completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
