"""High-Confidence Margin Curation & Variance Contraction Analysis (AP-2026-Q-2).

Author: Antigravity
Date: 2026-11-09
Policy: Evaluate variance reduction through selective margin filtering to maximize P5 under prequential stress testing.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score

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
    print("ANALYSIS OF HIGH-CONFIDENCE MARGIN CURATION FOR VARIANCE CONTRACTION")
    print("Policy: Full compliance with CX-0138 and Ariel directives")
    print("=" * 80, flush=True)

    t0 = time.time()
    
    # 1. Load holdout split
    with open(OUT_DIR / "frozen_dual_regime_indices.json", "r") as f:
        idx_data = json.load(f)
    seen_ho_idx = np.array(idx_data["seen_holdout_idx"])
    unseen_ho_idx = np.array(idx_data["unseen_holdout_idx"])
    
    df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
    df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)
    
    y_train = (df_train["condition"] == "used").astype(np.int8).values
    df_train["target"] = y_train
    
    df_seen_ho = df_train.iloc[seen_ho_idx].reset_index(drop=True)
    df_unseen_ho = df_train.iloc[unseen_ho_idx].reset_index(drop=True)
    y_seen_ho = y_train[seen_ho_idx]
    y_unseen_ho = y_train[unseen_ho_idx]
    y_holdout = np.concatenate([y_seen_ho, y_unseen_ho])
    sellers_holdout = np.concatenate([df_seen_ho["seller_id"].values, df_unseen_ho["seller_id"].values])
    
    # Run train_corrected_dual_regime_framework to extract probabilities or train fast
    sys.path.append(str((ROOT / "scripts").resolve()))
    from train_corrected_dual_regime_framework import FoldFeatureExtractor
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    
    fit_idx = np.array(idx_data["fit_idx"])
    df_fit = df_train.iloc[fit_idx].reset_index(drop=True)
    y_fit = y_train[fit_idx]
    
    print(f"Loaded {len(y_fit)} fit items and {len(y_holdout)} holdout items.")
    print("Extracting features on fit pool and transforming holdout and test...")
    
    fe_full = FoldFeatureExtractor()
    fe_full.fit(df_fit, y_fit)
    
    X_fit_dense = fe_full.transform_dense(df_fit)
    X_fit_sparse = fe_full.transform_sparse_text(df_fit)
    
    X_seen_dense = fe_full.transform_dense(df_seen_ho)
    X_seen_sparse = fe_full.transform_sparse_text(df_seen_ho)
    
    X_unseen_dense = fe_full.transform_dense(df_unseen_ho)
    X_unseen_sparse = fe_full.transform_sparse_text(df_unseen_ho)
    
    X_test_dense = fe_full.transform_dense(df_test)
    X_test_sparse = fe_full.transform_sparse_text(df_test)
    
    # Train base models
    print("Training base models on fit pool...")
    m_lgb = lgb.LGBMClassifier(n_estimators=420, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    m_lgb.fit(X_fit_dense, y_fit)
    
    m_xgb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.06, max_depth=6, random_state=42, eval_metric="logloss", n_jobs=-1)
    m_xgb.fit(X_fit_dense, y_fit)
    
    m_cat = CatBoostClassifier(iterations=350, learning_rate=0.06, depth=6, random_seed=42, verbose=0, thread_count=-1)
    m_cat.fit(X_fit_dense, y_fit)
    
    m_rdg = RidgeClassifier(alpha=2.0, random_state=42)
    m_rdg.fit(X_fit_sparse, y_fit)
    
    # Holdout base predictions
    p_seen_lgb = m_lgb.predict_proba(X_seen_dense)[:, 1]
    p_seen_xgb = m_xgb.predict_proba(X_seen_dense)[:, 1]
    p_seen_cat = m_cat.predict_proba(X_seen_dense)[:, 1]
    p_seen_rdg = 1.0 / (1.0 + np.exp(-m_rdg.decision_function(X_seen_sparse)))
    
    p_unseen_lgb = m_lgb.predict_proba(X_unseen_dense)[:, 1]
    p_unseen_xgb = m_xgb.predict_proba(X_unseen_dense)[:, 1]
    p_unseen_cat = m_cat.predict_proba(X_unseen_dense)[:, 1]
    p_unseen_rdg = 1.0 / (1.0 + np.exp(-m_rdg.decision_function(X_unseen_sparse)))
    
    # Test base predictions
    p_test_lgb = m_lgb.predict_proba(X_test_dense)[:, 1]
    p_test_xgb = m_xgb.predict_proba(X_test_dense)[:, 1]
    p_test_cat = m_cat.predict_proba(X_test_dense)[:, 1]
    p_test_rdg = 1.0 / (1.0 + np.exp(-m_rdg.decision_function(X_test_sparse)))
    
    # Fit meta-models on fit pool OOF
    from train_corrected_dual_regime_framework import run_honest_cross_fitting, fit_dual_regime_meta_models
    oof_lgb, oof_xgb, oof_cat, oof_rdg = run_honest_cross_fitting(df_fit)
    meta_seen, meta_unseen, _ = fit_dual_regime_meta_models(df_fit, oof_lgb, oof_xgb, oof_cat, oof_rdg)
    
    # Holdout meta probabilities
    meta_in_seen = np.column_stack([p_seen_lgb, p_seen_xgb, p_seen_cat, p_seen_rdg])
    p_seen_meta = meta_seen.predict_proba(meta_in_seen)[:, 1]
    
    meta_in_unseen = np.column_stack([p_unseen_lgb, p_unseen_xgb, p_unseen_cat, p_unseen_rdg])
    p_unseen_meta = meta_unseen.predict_proba(meta_in_unseen)[:, 1]
    
    # Test meta probabilities
    meta_in_test = np.column_stack([p_test_lgb, p_test_xgb, p_test_cat, p_test_rdg])
    is_seen_in_fit = np.isin(df_test["seller_id"].values, list(fe_full.seller_te.keys()))
    p_test_meta = np.zeros(len(df_test), dtype=np.float32)
    p_test_meta[is_seen_in_fit] = meta_seen.predict_proba(meta_in_test[is_seen_in_fit])[:, 1]
    p_test_meta[~is_seen_in_fit] = meta_unseen.predict_proba(meta_in_test[~is_seen_in_fit])[:, 1]
    
    # Baseline holdout and test
    base_seen_pred = (p_seen_lgb >= 0.5).astype(np.int8)
    base_unseen_pred = (p_unseen_lgb >= 0.5).astype(np.int8)
    base_holdout_pred = np.concatenate([base_seen_pred, base_unseen_pred])
    base_test_pred = (p_test_lgb >= 0.5).astype(np.int8)
    
    # Clustered bootstrap setup
    unique_sellers = np.unique(sellers_holdout)
    seller_to_indices = {s: np.where(sellers_holdout == s)[0] for s in unique_sellers}
    N_pub = 20998
    gamma = 2.9557
    H_anchor = 19896
    
    # SWEEP OVER MARGIN DELTA
    margins = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    sweep_results = []
    
    print("\n" + "=" * 95)
    print(f"{'Margin':6s} | {'Flips Test':10s} | {'Flips Ho':8s} | {'Fixes':5s} | {'Breaks':6s} | {'Net Ho':6s} | {'Prec%':6s} | {'Raw P5':7s} | {'Raw P50':7s} | {'Adj P5':7s} | {'P(G>=53)':8s}")
    print("=" * 95)
    
    rng = np.random.RandomState(42)
    B = 1000
    
    best_candidate_df = None
    best_adj_p5 = -999.0
    best_margin = 0.0
    
    for m in margins:
        # Candidate seen holdout
        cand_seen = base_seen_pred.copy()
        flip_s = (np.abs(p_seen_meta - 0.5) >= m) & ((p_seen_meta >= 0.5).astype(np.int8) != base_seen_pred)
        cand_seen[flip_s] = (p_seen_meta[flip_s] >= 0.5).astype(np.int8)
        
        # Candidate unseen holdout
        cand_unseen = base_unseen_pred.copy()
        flip_u = (np.abs(p_unseen_meta - 0.5) >= m) & ((p_unseen_meta >= 0.5).astype(np.int8) != base_unseen_pred)
        cand_unseen[flip_u] = (p_unseen_meta[flip_u] >= 0.5).astype(np.int8)
        
        cand_holdout = np.concatenate([cand_seen, cand_unseen])
        
        # Candidate test
        cand_test = base_test_pred.copy()
        flip_t = (np.abs(p_test_meta - 0.5) >= m) & ((p_test_meta >= 0.5).astype(np.int8) != base_test_pred)
        cand_test[flip_t] = (p_test_meta[flip_t] >= 0.5).astype(np.int8)
        
        n_flips_t = int(np.sum(flip_t))
        n_flips_ho = int(np.sum(flip_s) + np.sum(flip_u))
        
        fixes = int(np.sum((base_holdout_pred != y_holdout) & (cand_holdout == y_holdout)))
        breaks = int(np.sum((base_holdout_pred == y_holdout) & (cand_holdout != y_holdout)))
        net_ho = fixes - breaks
        prec = (fixes / max(fixes + breaks, 1)) * 100.0
        
        # Bootstrap
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
        
        row = {
            "margin": m, "test_flips": n_flips_t, "holdout_flips": n_flips_ho,
            "fixes": fixes, "breaks": breaks, "net_holdout": net_ho, "precision": prec,
            "raw_p5": p5_raw, "raw_p50": p50_raw, "raw_p95": p95_raw,
            "adj_p5": p5_adj, "prob_ge_53_stress": prob_adj
        }
        sweep_results.append(row)
        
        if p5_adj > best_adj_p5:
            best_adj_p5 = p5_adj
            best_margin = m
            best_candidate_df = pd.DataFrame({
                "ID": np.arange(1, len(df_test) + 1),
                "condition": np.where(cand_test == 1, "used", "new")
            })

    print("=" * 95)
    print(f"\nBest Adjusted P5 achieved at margin delta = {best_margin:.2f} (P5_adj: {best_adj_p5:+.1f} hits)")
    
    # Save sweep report
    with open(OUT_DIR / "margin_curation_sweep_report.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
        
    # Save optimized candidate submission
    if best_candidate_df is not None:
        csv_p = OUT_DIR / "candidate_c7_optimized_margin_variance_reduced.csv"
        best_candidate_df.to_csv(csv_p, index=False)
        c_hash = sha256_file(csv_p)
        print(f"Optimized Submission saved to: {csv_p}")
        print(f"SHA-256: {c_hash}")
        
    print(f"\nOptimization completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
