"""Iterative Aligned Cycle 2: Cross-Model Feature Injection & Soft Label Regularization.

Author: Antigravity
Date: 2026-11-09
Policy: Full zero-leakage fold-local feature alignment across tree and lexical modalities.
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
import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path(".")
DATA_DIR = ROOT / "data/ap_2026_q_2/raw"
OUT_DIR = ROOT / "results/dual_regime_20261109"

sys.path.append(str((ROOT / "scripts").resolve()))
from train_corrected_dual_regime_framework import FoldFeatureExtractor


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("ITERATIVE ALIGNED CYCLE 2: MUTUAL MODALITY ALIGNMENT & TREE ENRICHMENT")
    print("Policy: Full zero-leakage fold-local cross-fitting")
    print("=" * 80, flush=True)

    t0 = time.time()
    
    # 1. Load data and frozen indices
    df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
    df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)
    
    y_train = (df_train["condition"] == "used").astype(np.int8).values
    df_train["target"] = y_train
    
    with open(OUT_DIR / "frozen_dual_regime_indices.json", "r") as f:
        idx_data = json.load(f)
        
    fit_idx = np.array(idx_data["fit_idx"])
    seen_ho_idx = np.array(idx_data["seen_holdout_idx"])
    unseen_ho_idx = np.array(idx_data["unseen_holdout_idx"])
    
    df_fit = df_train.iloc[fit_idx].reset_index(drop=True)
    df_seen_ho = df_train.iloc[seen_ho_idx].reset_index(drop=True)
    df_unseen_ho = df_train.iloc[unseen_ho_idx].reset_index(drop=True)
    
    y_fit = y_train[fit_idx]
    y_seen = y_train[seen_ho_idx]
    y_unseen = y_train[unseen_ho_idx]
    y_holdout = np.concatenate([y_seen, y_unseen])
    sellers_holdout = np.concatenate([df_seen_ho["seller_id"].values, df_unseen_ho["seller_id"].values])
    
    print(f"Fit pool: {len(df_fit)} items | Seen holdout: {len(df_seen_ho)} | Unseen holdout: {len(df_unseen_ho)}")

    # 2. Honest 5-fold cross-fitting with mutual feature injection
    print("\n[1/4] Running Fold-Local Cross-Fitting with Lexical Alignment...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_cat = np.zeros(len(df_fit), dtype=np.float32)
    oof_lgb = np.zeros(len(df_fit), dtype=np.float32)
    oof_ridge = np.zeros(len(df_fit), dtype=np.float32)
    
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df_fit, y_fit, df_fit["seller_id"].values)):
        f_t0 = time.time()
        print(f"  Fold {fold+1}/5...", end="", flush=True)
        
        df_tr = df_fit.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_fit[tr_idx]
        df_va = df_fit.iloc[va_idx].reset_index(drop=True)
        y_va = y_fit[va_idx]
        
        # A. Fold-local feature extractor
        fe = FoldFeatureExtractor()
        fe.fit(df_tr, y_tr)
        
        X_tr_dense = fe.transform_dense(df_tr)
        X_va_dense = fe.transform_dense(df_va)
        
        # B. Fold-local high-dim sparse text model
        vec_word = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
        vec_char = TfidfVectorizer(max_features=20000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        
        titles_tr = df_tr["title"].fillna("").astype(str).str.lower()
        titles_va = df_va["title"].fillna("").astype(str).str.lower()
        
        w_tr = vec_word.fit_transform(titles_tr)
        c_tr = vec_char.fit_transform(titles_tr)
        X_tr_sp = sparse.hstack([w_tr, c_tr], format="csr")
        
        w_va = vec_word.transform(titles_va)
        c_va = vec_char.transform(titles_va)
        X_va_sp = sparse.hstack([w_va, c_va], format="csr")
        
        m_rdg = RidgeClassifier(alpha=2.0, random_state=42)
        m_rdg.fit(X_tr_sp, y_tr)
        
        p_tr_ridge = 1.0 / (1.0 + np.exp(-m_rdg.decision_function(X_tr_sp)))
        p_va_ridge = 1.0 / (1.0 + np.exp(-m_rdg.decision_function(X_va_sp)))
        oof_ridge[va_idx] = p_va_ridge
        
        # C. Inject text probability and logit into dense features
        tr_logit = np.log(np.clip(p_tr_ridge, 1e-4, 1 - 1e-4) / (1.0 - np.clip(p_tr_ridge, 1e-4, 1 - 1e-4)))
        va_logit = np.log(np.clip(p_va_ridge, 1e-4, 1 - 1e-4) / (1.0 - np.clip(p_va_ridge, 1e-4, 1 - 1e-4)))
        
        X_tr_enriched = np.column_stack([X_tr_dense, p_tr_ridge, tr_logit])
        X_va_enriched = np.column_stack([X_va_dense, p_va_ridge, va_logit])
        
        # D. Train Enriched LightGBM
        m_lgb = lgb.LGBMClassifier(n_estimators=380, learning_rate=0.05, num_leaves=35, random_state=42, verbose=-1, n_jobs=-1)
        m_lgb.fit(X_tr_enriched, y_tr)
        oof_lgb[va_idx] = m_lgb.predict_proba(X_va_enriched)[:, 1]
        
        # E. Train Enriched CatBoost
        m_cat = CatBoostClassifier(iterations=350, learning_rate=0.06, depth=6, random_seed=42, verbose=0, thread_count=-1)
        m_cat.fit(X_tr_enriched, y_tr)
        oof_cat[va_idx] = m_cat.predict_proba(X_va_enriched)[:, 1]
        
        acc_fold = accuracy_score(y_va, (oof_cat[va_idx] >= 0.5))
        print(f" Done ({time.time()-f_t0:.1f}s) | CatB Acc: {acc_fold:.4f}", flush=True)

    print("\nHonest OOF Performance with Aligned Features on 56k Fit Pool:")
    print(f"  Ridge Text Alone: Acc {accuracy_score(y_fit, (oof_ridge >= 0.5)):.5f} | LogLoss {log_loss(y_fit, oof_ridge):.5f}")
    print(f"  Enriched LGBM:    Acc {accuracy_score(y_fit, (oof_lgb >= 0.5)):.5f} | LogLoss {log_loss(y_fit, oof_lgb):.5f}")
    print(f"  Enriched CatB:    Acc {accuracy_score(y_fit, (oof_cat >= 0.5)):.5f} | LogLoss {log_loss(y_fit, oof_cat):.5f}")

    # 3. Train full fit models and evaluate on holdout and test
    print("\n[2/4] Training Full Aligned Models on 56k Fit Pool & Transforming Holdout/Test...")
    fe_full = FoldFeatureExtractor()
    fe_full.fit(df_fit, y_fit)
    
    X_fit_dense = fe_full.transform_dense(df_fit)
    X_seen_dense = fe_full.transform_dense(df_seen_ho)
    X_unseen_dense = fe_full.transform_dense(df_unseen_ho)
    X_test_dense = fe_full.transform_dense(df_test)
    
    # Sparse text fit
    vec_word_full = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
    vec_char_full = TfidfVectorizer(max_features=20000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    
    w_fit = vec_word_full.fit_transform(df_fit["title"].fillna("").astype(str).str.lower())
    c_fit = vec_char_full.fit_transform(df_fit["title"].fillna("").astype(str).str.lower())
    X_fit_sp = sparse.hstack([w_fit, c_fit], format="csr")
    
    m_rdg_full = RidgeClassifier(alpha=2.0, random_state=42)
    m_rdg_full.fit(X_fit_sp, y_fit)
    
    p_fit_rdg = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_fit_sp)))
    fit_logit = np.log(np.clip(p_fit_rdg, 1e-4, 1 - 1e-4) / (1.0 - np.clip(p_fit_rdg, 1e-4, 1 - 1e-4)))
    X_fit_enriched = np.column_stack([X_fit_dense, p_fit_rdg, fit_logit])
    
    # Seen holdout
    w_seen = vec_word_full.transform(df_seen_ho["title"].fillna("").astype(str).str.lower())
    c_seen = vec_char_full.transform(df_seen_ho["title"].fillna("").astype(str).str.lower())
    X_seen_sp = sparse.hstack([w_seen, c_seen], format="csr")
    p_seen_rdg = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_seen_sp)))
    seen_logit = np.log(np.clip(p_seen_rdg, 1e-4, 1 - 1e-4) / (1.0 - np.clip(p_seen_rdg, 1e-4, 1 - 1e-4)))
    X_seen_enriched = np.column_stack([X_seen_dense, p_seen_rdg, seen_logit])
    
    # Unseen holdout
    w_unseen = vec_word_full.transform(df_unseen_ho["title"].fillna("").astype(str).str.lower())
    c_unseen = vec_char_full.transform(df_unseen_ho["title"].fillna("").astype(str).str.lower())
    X_unseen_sp = sparse.hstack([w_unseen, c_unseen], format="csr")
    p_unseen_rdg = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_unseen_sp)))
    unseen_logit = np.log(np.clip(p_unseen_rdg, 1e-4, 1 - 1e-4) / (1.0 - np.clip(p_unseen_rdg, 1e-4, 1 - 1e-4)))
    X_unseen_enriched = np.column_stack([X_unseen_dense, p_unseen_rdg, unseen_logit])
    
    # Test
    w_test = vec_word_full.transform(df_test["title"].fillna("").astype(str).str.lower())
    c_test = vec_char_full.transform(df_test["title"].fillna("").astype(str).str.lower())
    X_test_sp = sparse.hstack([w_test, c_test], format="csr")
    p_test_rdg = 1.0 / (1.0 + np.exp(-m_rdg_full.decision_function(X_test_sp)))
    test_logit = np.log(np.clip(p_test_rdg, 1e-4, 1 - 1e-4) / (1.0 - np.clip(p_test_rdg, 1e-4, 1 - 1e-4)))
    X_test_enriched = np.column_stack([X_test_dense, p_test_rdg, test_logit])
    
    # Train full trees on enriched features
    m_lgb_full = lgb.LGBMClassifier(n_estimators=420, learning_rate=0.05, num_leaves=35, random_state=42, verbose=-1, n_jobs=-1)
    m_lgb_full.fit(X_fit_enriched, y_fit)
    
    m_cat_full = CatBoostClassifier(iterations=380, learning_rate=0.06, depth=6, random_seed=42, verbose=0, thread_count=-1)
    m_cat_full.fit(X_fit_enriched, y_fit)
    
    p_seen_lgb = m_lgb_full.predict_proba(X_seen_enriched)[:, 1]
    p_seen_cat = m_cat_full.predict_proba(X_seen_enriched)[:, 1]
    
    p_unseen_lgb = m_lgb_full.predict_proba(X_unseen_enriched)[:, 1]
    p_unseen_cat = m_cat_full.predict_proba(X_unseen_enriched)[:, 1]
    
    p_test_lgb = m_lgb_full.predict_proba(X_test_enriched)[:, 1]
    p_test_cat = m_cat_full.predict_proba(X_test_enriched)[:, 1]
    
    # Baseline comparison (LGBM on standard features from C6)
    # Load base predictions from holdout
    npz_p = OUT_DIR / "subagent_trees_predictions.npz"
    sub_data = np.load(npz_p)
    base_pred_seen = sub_data["base_pred_seen"]
    base_pred_unseen = sub_data["base_pred_unseen"]
    base_holdout_pred = np.concatenate([base_pred_seen, base_pred_unseen])
    base_test_pred = (sub_data["test_lgb"] >= 0.5).astype(np.int8)
    
    # 4. Meta-Stacker on OOF
    print("\n[3/4] Fitting Regime-Conditioned Meta-Stackers...")
    seller_counts = df_fit["seller_id"].value_counts()
    is_multi_seller = df_fit["seller_id"].map(seller_counts > 1).values
    
    meta_X_fit = np.column_stack([oof_cat, oof_lgb, oof_ridge])
    meta_seen = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    meta_seen.fit(meta_X_fit[is_multi_seller], y_fit[is_multi_seller])
    
    meta_unseen = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    meta_unseen.fit(meta_X_fit[~is_multi_seller], y_fit[~is_multi_seller])
    
    meta_X_seen = np.column_stack([p_seen_cat, p_seen_lgb, p_seen_rdg])
    p_seen_meta = meta_seen.predict_proba(meta_X_seen)[:, 1]
    
    meta_X_unseen = np.column_stack([p_unseen_cat, p_unseen_lgb, p_unseen_rdg])
    p_unseen_meta = meta_unseen.predict_proba(meta_X_unseen)[:, 1]
    
    meta_X_test = np.column_stack([p_test_cat, p_test_lgb, p_test_rdg])
    is_seen_in_fit = np.isin(df_test["seller_id"].values, df_fit["seller_id"].unique())
    p_test_meta = np.zeros(len(df_test), dtype=np.float32)
    p_test_meta[is_seen_in_fit] = meta_seen.predict_proba(meta_X_test[is_seen_in_fit])[:, 1]
    p_test_meta[~is_seen_in_fit] = meta_unseen.predict_proba(meta_X_test[~is_seen_in_fit])[:, 1]
    
    # 5. Margin Curation Sweep
    print("\n[4/4] Executing Margin Curation Sweep & Clustered Bootstrap...")
    unique_sellers = np.unique(sellers_holdout)
    seller_to_indices = {s: np.where(sellers_holdout == s)[0] for s in unique_sellers}
    N_pub = 20998
    gamma = 2.9557
    B = 1000
    rng = np.random.RandomState(42)
    
    margins = [0.00, 0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25]
    print("\n" + "=" * 98)
    print(f"{'Margin':6s} | {'Flips Test':10s} | {'Flips Ho':8s} | {'Fixes':5s} | {'Breaks':6s} | {'Net Ho':6s} | {'Prec%':6s} | {'Raw P5':7s} | {'Raw P50':7s} | {'Adj P5':7s} | {'P(G>=53)':8s}")
    print("=" * 98)
    
    best_candidate_df = None
    best_adj_p5 = -999.0
    best_margin = 0.0
    best_stats = {}
    
    for m in margins:
        cand_seen = base_pred_seen.copy()
        flip_s = (np.abs(p_seen_meta - 0.5) >= m) & ((p_seen_meta >= 0.5).astype(np.int8) != base_pred_seen)
        cand_seen[flip_s] = (p_seen_meta[flip_s] >= 0.5).astype(np.int8)
        
        cand_unseen = base_pred_unseen.copy()
        flip_u = (np.abs(p_unseen_meta - 0.5) >= m) & ((p_unseen_meta >= 0.5).astype(np.int8) != base_pred_unseen)
        cand_unseen[flip_u] = (p_unseen_meta[flip_u] >= 0.5).astype(np.int8)
        
        cand_holdout = np.concatenate([cand_seen, cand_unseen])
        
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
    print(f"\nCycle 2 Aligned Optimal Margin: delta = {best_margin:.2f}")
    print(f"Optimal Adjusted P5: {best_adj_p5:+.1f} hits | Raw P5: {best_stats['raw_p5']:+.1f} hits | Raw P50: {best_stats['raw_p50']:+.1f} hits")
    
    # Save candidate submission C9
    if best_candidate_df is not None:
        csv_p = OUT_DIR / "candidate_c9_aligned_cycle2.csv"
        best_candidate_df.to_csv(csv_p, index=False)
        c_hash = sha256_file(csv_p)
        print(f"\nSaved Candidate C9: {csv_p}")
        print(f"SHA-256: {c_hash}")
        
    rep_p = OUT_DIR / "aligned_cycle2_report.json"
    with open(rep_p, "w") as f:
        json.dump(best_stats, f, indent=2)
    print(f"Saved Report: {rep_p}")
    print(f"Completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
