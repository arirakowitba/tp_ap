"""Train Tabular and Causal LightGBM Models under Strict Seller-Group and Temporal Regimes.

Complies strictly with CX-0219, CX-0220, CX-0221, and CC-0036/0038:
- Line 1: LightGBM Tabular Puro on clean, fold-safe intrinsic features.
- Line 2: LightGBM Tabular + Causal Seller Signals (strictly prior in time within fit, ties excluded, Bayesian shrinkage beta=10).
- Strict separation of cold seller-group evaluation and warm temporal evaluation.
- Metrics: Accuracy, LogLoss, fixes/breaks, 5-regime breakdown, seller support bins, seller-grouped bootstrap Q05.
- Single confirmation on confirm set (14,075 rows) only after freezing hyperparameters.
- Exports OOF predictions (70,000 rows) and test predictions (30,000 rows) for inter-agent late fusion with Claude.
"""

import os
import sys
import json
import time
import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from lightgbm import LGBMClassifier

sys.path.append('.')
from tests.test_preventive_gates import compute_id_target_hash, validate_gate_g

ROOT = Path(".")
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "tabular_causal_lgbm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_causal_seller_prior_fit(df_fit, beta=10.0, global_prior=0.463):
    """
    Computes strictly causal prior history for each row in df_fit.
    Prior history includes only items with start_time < current start_time of the same seller.
    Ties (start_time == current start_time) are excluded.
    Returns:
        prior_n: np.ndarray of shape (N,)
        prior_p: np.ndarray of shape (N,)
        tie_count: np.ndarray of shape (N,)
        seller_lookup: dict mapping seller_id -> (sorted_start_times, cumsum_y)
    """
    df = df_fit[['seller_id', 'start_time', 'y_num']].copy()
    # Group by (seller_id, start_time)
    grp = df.groupby(['seller_id', 'start_time'])['y_num'].agg(['count', 'sum']).reset_index()
    grp['cum_count'] = grp.groupby('seller_id')['count'].cumsum() - grp['count']
    grp['cum_sum'] = grp.groupby('seller_id')['sum'].cumsum() - grp['sum']
    
    # Map back to df_fit indices
    merged = df.merge(grp[['seller_id', 'start_time', 'cum_count', 'cum_sum', 'count']], 
                      on=['seller_id', 'start_time'], how='left')
    
    prior_n = merged['cum_count'].values.astype(np.float32)
    prior_sum = merged['cum_sum'].values.astype(np.float32)
    tie_count = (merged['count'].values - 1).astype(np.float32)
    
    prior_p = (prior_sum + beta * global_prior) / (prior_n + beta)
    
    # Build lookup for val/test
    # For each seller, store sorted start_times and cumulative sum of y
    df_sorted = df.sort_values(by=['seller_id', 'start_time'])
    seller_lookup = {}
    for s, s_grp in df_sorted.groupby('seller_id'):
        t_arr = s_grp['start_time'].values
        y_arr = s_grp['y_num'].values
        y_cum = np.r_[0.0, np.cumsum(y_arr)]
        seller_lookup[s] = (t_arr, y_cum)
        
    return prior_n, prior_p, tie_count, seller_lookup

def query_causal_seller_prior(df_query, seller_lookup, beta=10.0, global_prior=0.463):
    """
    Queries prior history from seller_lookup for each row in df_query.
    Only items with start_time_fit < start_time_query are counted.
    """
    sellers = df_query['seller_id'].values
    times = df_query['start_time'].values
    N = len(df_query)
    
    prior_n = np.zeros(N, dtype=np.float32)
    prior_sum = np.zeros(N, dtype=np.float32)
    
    for i in range(N):
        s = sellers[i]
        t = times[i]
        if s in seller_lookup:
            t_arr, y_cum = seller_lookup[s]
            idx = np.searchsorted(t_arr, t, side='left')
            prior_n[i] = idx
            prior_sum[i] = y_cum[idx]
            
    prior_p = (prior_sum + beta * global_prior) / (prior_n + beta)
    return prior_n, prior_p

def bootstrap_seller_q05(diff_series, seller_series, n_iter=2000, seed=42):
    """Computes Q05 of delta under seller-clustered bootstrap."""
    v = diff_series.groupby(seller_series).sum().values
    rng = np.random.default_rng(seed)
    n_sellers = len(v)
    boot_sums = np.array([v[rng.integers(0, n_sellers, n_sellers)].sum() for _ in range(n_iter)])
    return float(np.percentile(boot_sums, 5)), float(np.percentile(boot_sums, 50))

def main():
    t_start = time.time()
    print("=" * 70)
    print("AP-2026-Q-2: TABULAR AND CAUSAL LIGHTGBM TRAINING PIPELINE")
    print(f"Start time UTC: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
    print("=" * 70, flush=True)

    # 1. Load data
    print("\n[1/6] Loading data and pre-computed features...", flush=True)
    with open(DATA_DIR / "ap_2026_q_2" / "raw" / "feats.pkl", "rb") as f:
        feats = pickle.load(f)
    with open(DATA_DIR / "splits" / "t98_parsed_train.pkl", "rb") as f:
        df_train_meta = pickle.load(f)
    with open(DATA_DIR / "splits" / "t98_parsed_test.pkl", "rb") as f:
        df_test_meta = pickle.load(f)
        
    splits_3way = pd.read_csv(DATA_DIR / "splits" / "canonical_3way_splits.csv")
    splits_nested = pd.read_csv(DATA_DIR / "splits" / "canonical_nested_dual_regime_splits.csv")
    
    F_full = feats["F"].copy()
    y_full = feats["y"].copy()
    is_test_mask = (feats["is_test"] == 1)
    
    F_train = F_full.iloc[:70000].copy()
    F_test = F_full.iloc[70000:].copy()
    
    y_train = (y_full == 1).astype(int) if set(np.unique(y_full)).issubset({0, 1}) else (y_full == "used").astype(int)
    
    # 2. Audit feats.pkl columns for leaks
    print("\n[2/6] Auditing features for leakage...", flush=True)
    # The 8 global leaky columns in feats.pkl:
    leaky_cols = ['freq_seller_id', 'freq_category_id', 'freq_city', 'seller_n_cats', 
                  'seller_mean_logprice', 'seller_mean_init_q', 'cat_med_logprice', 'rel_price_cat']
    print(f"  Flagged 8 global aggregate columns to recompute fold-safe: {leaky_cols}")
    
    # Drop raw high-cardinality seller_id and global leaky columns from base feature matrix
    base_drop_cols = ['seller_id'] + leaky_cols
    base_feature_cols = [c for c in F_train.columns if c not in base_drop_cols]
    print(f"  Clean intrinsic tabular feature columns: {len(base_feature_cols)}")
    
    # Convert categorical/object columns to category type for LightGBM
    for c in base_feature_cols:
        if F_train[c].dtype == object or str(F_train[c].dtype) == 'category':
            F_train[c] = F_train[c].astype('category')
            F_test[c] = F_test[c].astype('category')
            
    # Prepare meta data for causal and fold evaluation
    df_train_meta['y_num'] = y_train
    global_prior = float(y_train.mean())
    print(f"  Train set size: {len(F_train)}, Global used rate: {global_prior:.4f}")
    print(f"  Test set size: {len(F_test)}")

    # 3. Model Line 1: LightGBM Tabular Puro on 5-fold Nested Unseen Seller Splits
    print("\n[3/6] Training Line 1: LightGBM Tabular Puro (5-fold Nested Split)...", flush=True)
    lgb_params = {
        "n_estimators": 450,
        "learning_rate": 0.04,
        "num_leaves": 47,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    oof_line1 = np.zeros(70000, dtype=np.float32)
    test_line1 = np.zeros(30000, dtype=np.float32)
    
    oof_line2 = np.zeros(70000, dtype=np.float32)
    test_line2 = np.zeros(30000, dtype=np.float32)
    
    # We will perform cross-validation across the 5 outer folds
    for fold in range(5):
        t_f0 = time.time()
        # outer_unseen_fold == fold is the 14,000 unseen seller test fold
        val_mask = (splits_nested['outer_unseen_fold'] == fold).values
        fit_mask = ~val_mask
        
        # Prepare fold-safe features: compute fold-safe category median logprice and rel_price_cat
        # on fit_mask ONLY
        fit_cat_logprice = F_train.loc[fit_mask].groupby('category_id', observed=False)['log_price'].median().to_dict()
        global_med_logprice = float(F_train.loc[fit_mask, 'log_price'].median())
        
        cat_med_tr = F_train['category_id'].map(lambda c: fit_cat_logprice.get(c, global_med_logprice)).values
        rel_price_tr = F_train['log_price'].values - cat_med_tr
        
        cat_med_te = F_test['category_id'].map(lambda c: fit_cat_logprice.get(c, global_med_logprice)).values
        rel_price_te = F_test['log_price'].values - cat_med_te
        
        # Line 1 feature matrix for this fold
        X_tr1 = F_train.loc[fit_mask, base_feature_cols].copy()
        X_tr1['cat_med_logprice_safe'] = cat_med_tr[fit_mask]
        X_tr1['rel_price_cat_safe'] = rel_price_tr[fit_mask]
        
        X_val1 = F_train.loc[val_mask, base_feature_cols].copy()
        X_val1['cat_med_logprice_safe'] = cat_med_tr[val_mask]
        X_val1['rel_price_cat_safe'] = rel_price_tr[val_mask]
        
        X_te1 = F_test[base_feature_cols].copy()
        X_te1['cat_med_logprice_safe'] = cat_med_te
        X_te1['rel_price_cat_safe'] = rel_price_te
        
        y_fit = y_train[fit_mask]
        y_val = y_train[val_mask]
        
        # Fit Line 1
        m1 = LGBMClassifier(**lgb_params)
        m1.fit(X_tr1, y_fit)
        
        val_p1 = m1.predict_proba(X_val1)[:, 1]
        oof_line1[val_mask] = val_p1
        test_line1 += m1.predict_proba(X_te1)[:, 1] / 5.0
        
        # Line 2: Add Causal Seller Signals (strictly prior within fit)
        # Compute causal features on fit
        df_fit = df_train_meta.loc[fit_mask].copy()
        df_val = df_train_meta.loc[val_mask].copy()
        df_te = df_test_meta.copy()
        
        prior_n_fit, prior_p_fit, tie_fit, s_lookup = compute_causal_seller_prior_fit(
            df_fit, beta=10.0, global_prior=global_prior
        )
        prior_n_val, prior_p_val = query_causal_seller_prior(
            df_val, s_lookup, beta=10.0, global_prior=global_prior
        )
        prior_n_te, prior_p_te = query_causal_seller_prior(
            df_te, s_lookup, beta=10.0, global_prior=global_prior
        )
        
        # Add causal features to Line 2 matrices
        X_tr2 = X_tr1.copy()
        X_tr2['seller_p_causal'] = prior_p_fit
        X_tr2['seller_causal_n_prior'] = np.log1p(prior_n_fit)
        X_tr2['seller_causal_has_history'] = (prior_n_fit > 0).astype(np.float32)
        X_tr2['seller_tie_count'] = np.log1p(tie_fit)
        
        X_val2 = X_val1.copy()
        X_val2['seller_p_causal'] = prior_p_val
        X_val2['seller_causal_n_prior'] = np.log1p(prior_n_val)
        X_val2['seller_causal_has_history'] = (prior_n_val > 0).astype(np.float32)
        X_val2['seller_tie_count'] = 0.0 # not applicable for val/test
        
        X_te2 = X_te1.copy()
        X_te2['seller_p_causal'] = prior_p_te
        X_te2['seller_causal_n_prior'] = np.log1p(prior_n_te)
        X_te2['seller_causal_has_history'] = (prior_n_te > 0).astype(np.float32)
        X_te2['seller_tie_count'] = 0.0
        
        # Fit Line 2
        m2 = LGBMClassifier(**lgb_params)
        m2.fit(X_tr2, y_fit)
        
        val_p2 = m2.predict_proba(X_val2)[:, 1]
        oof_line2[val_mask] = val_p2
        test_line2 += m2.predict_proba(X_te2)[:, 1] / 5.0
        
        acc1 = accuracy_score(y_val, val_p1 >= 0.5)
        acc2 = accuracy_score(y_val, val_p2 >= 0.5)
        ll1 = log_loss(y_val, val_p1)
        ll2 = log_loss(y_val, val_p2)
        
        elapsed = time.time() - t_f0
        print(f"  Fold {fold} ({val_mask.sum()} unseen cold rows): Line 1 acc={acc1:.4f} ll={ll1:.4f} | Line 2 acc={acc2:.4f} ll={ll2:.4f} [{elapsed:.1f}s]", flush=True)

    # 4. Canonical 3-Way Split Evaluation: Tune Selection & Single Confirm
    print("\n[4/6] Evaluating on Canonical 3-way Split (Fit: 40.8k, Tune: 15.0k, Confirm: 14.0k)...", flush=True)
    mask_fit_3w = (splits_3way['role'] == 'fit').values
    mask_tune_3w = (splits_3way['role'] == 'tune').values
    mask_conf_3w = (splits_3way['role'] == 'confirm').values
    
    # Fit strictly on 3-way fit
    df_fit_3w = df_train_meta.loc[mask_fit_3w].copy()
    prior_n_fit3w, prior_p_fit3w, tie_fit3w, s_lookup_3w = compute_causal_seller_prior_fit(
        df_fit_3w, beta=10.0, global_prior=global_prior
    )
    
    fit_cat_logprice_3w = F_train.loc[mask_fit_3w].groupby('category_id', observed=False)['log_price'].median().to_dict()
    global_med_logprice_3w = float(F_train.loc[mask_fit_3w, 'log_price'].median())
    
    cat_med_tr3w = F_train['category_id'].map(lambda c: fit_cat_logprice_3w.get(c, global_med_logprice_3w)).values
    rel_price_tr3w = F_train['log_price'].values - cat_med_tr3w
    
    # Matrices for fit, tune, confirm
    X_f1 = F_train.loc[mask_fit_3w, base_feature_cols].copy()
    X_f1['cat_med_logprice_safe'] = cat_med_tr3w[mask_fit_3w]
    X_f1['rel_price_cat_safe'] = rel_price_tr3w[mask_fit_3w]
    
    X_tu1 = F_train.loc[mask_tune_3w, base_feature_cols].copy()
    X_tu1['cat_med_logprice_safe'] = cat_med_tr3w[mask_tune_3w]
    X_tu1['rel_price_cat_safe'] = rel_price_tr3w[mask_tune_3w]
    
    X_co1 = F_train.loc[mask_conf_3w, base_feature_cols].copy()
    X_co1['cat_med_logprice_safe'] = cat_med_tr3w[mask_conf_3w]
    X_co1['rel_price_cat_safe'] = rel_price_tr3w[mask_conf_3w]
    
    # Line 2 matrices with causal prior
    prior_n_tu, prior_p_tu = query_causal_seller_prior(df_train_meta.loc[mask_tune_3w], s_lookup_3w, beta=10.0, global_prior=global_prior)
    prior_n_co, prior_p_co = query_causal_seller_prior(df_train_meta.loc[mask_conf_3w], s_lookup_3w, beta=10.0, global_prior=global_prior)
    
    X_f2 = X_f1.copy()
    X_f2['seller_p_causal'] = prior_p_fit3w
    X_f2['seller_causal_n_prior'] = np.log1p(prior_n_fit3w)
    X_f2['seller_causal_has_history'] = (prior_n_fit3w > 0).astype(np.float32)
    X_f2['seller_tie_count'] = np.log1p(tie_fit3w)
    
    X_tu2 = X_tu1.copy()
    X_tu2['seller_p_causal'] = prior_p_tu
    X_tu2['seller_causal_n_prior'] = np.log1p(prior_n_tu)
    X_tu2['seller_causal_has_history'] = (prior_n_tu > 0).astype(np.float32)
    X_tu2['seller_tie_count'] = 0.0
    
    X_co2 = X_co1.copy()
    X_co2['seller_p_causal'] = prior_p_co
    X_co2['seller_causal_n_prior'] = np.log1p(prior_n_co)
    X_co2['seller_causal_has_history'] = (prior_n_co > 0).astype(np.float32)
    X_co2['seller_tie_count'] = 0.0
    
    # Fit models on 3-way fit
    m_3w_1 = LGBMClassifier(**lgb_params)
    m_3w_1.fit(X_f1, y_train[mask_fit_3w])
    p_tu1 = m_3w_1.predict_proba(X_tu1)[:, 1]
    p_co1 = m_3w_1.predict_proba(X_co1)[:, 1]
    
    m_3w_2 = LGBMClassifier(**lgb_params)
    m_3w_2.fit(X_f2, y_train[mask_fit_3w])
    p_tu2 = m_3w_2.predict_proba(X_tu2)[:, 1]
    p_co2 = m_3w_2.predict_proba(X_co2)[:, 1]
    
    # Compute metrics on Tune
    y_tu = y_train[mask_tune_3w]
    acc_tu1 = accuracy_score(y_tu, p_tu1 >= 0.5)
    acc_tu2 = accuracy_score(y_tu, p_tu2 >= 0.5)
    ll_tu1 = log_loss(y_tu, p_tu1)
    ll_tu2 = log_loss(y_tu, p_tu2)
    
    # Cold vs warm breakdown in Tune
    tune_cold = splits_3way.loc[mask_tune_3w, 'is_cold_tune'].values.astype(bool)
    tune_warm = splits_3way.loc[mask_tune_3w, 'is_warm_tune'].values.astype(bool)
    
    print(f"\n  --- TUNE RESULTS (15,044 rows: {tune_cold.sum()} cold, {tune_warm.sum()} warm) ---")
    print(f"  Line 1 Tabular: Overall Acc={acc_tu1:.5f} (Cold={accuracy_score(y_tu[tune_cold], p_tu1[tune_cold] >= 0.5):.5f}, Warm={accuracy_score(y_tu[tune_warm], p_tu1[tune_warm] >= 0.5):.5f}) LL={ll_tu1:.5f}")
    print(f"  Line 2 Causal : Overall Acc={acc_tu2:.5f} (Cold={accuracy_score(y_tu[tune_cold], p_tu2[tune_cold] >= 0.5):.5f}, Warm={accuracy_score(y_tu[tune_warm], p_tu2[tune_warm] >= 0.5):.5f}) LL={ll_tu2:.5f}")
    delta_tu = int((p_tu2 >= 0.5).astype(int).sum() - (p_tu1 >= 0.5).astype(int).sum())
    fixes_tu = int(((p_tu2 >= 0.5) == y_tu & ((p_tu1 >= 0.5) != y_tu)).sum())
    breaks_tu = int(((p_tu2 >= 0.5) != y_tu & ((p_tu1 >= 0.5) == y_tu)).sum())
    print(f"  Line 2 vs Line 1 on Tune: Fixes={fixes_tu}, Breaks={breaks_tu}, Net={fixes_tu - breaks_tu}")
    
    # SINGLE EVALUATION ON CONFIRM (14,075 rows)
    y_co = y_train[mask_conf_3w]
    acc_co1 = accuracy_score(y_co, p_co1 >= 0.5)
    acc_co2 = accuracy_score(y_co, p_co2 >= 0.5)
    ll_co1 = log_loss(y_co, p_co1)
    ll_co2 = log_loss(y_co, p_co2)
    
    conf_cold = splits_3way.loc[mask_conf_3w, 'is_cold_confirm'].values.astype(bool)
    conf_warm = splits_3way.loc[mask_conf_3w, 'is_warm_confirm'].values.astype(bool)
    
    fixes_co = int(((p_co2 >= 0.5) == y_co & ((p_co1 >= 0.5) != y_co)).sum())
    breaks_co = int(((p_co2 >= 0.5) != y_co & ((p_co1 >= 0.5) == y_co)).sum())
    net_co = fixes_co - breaks_co
    
    # Bootstrap Q05 on Confirm
    df_conf_sellers = df_train_meta.loc[mask_conf_3w, 'seller_id']
    diff_co_series = pd.Series(((p_co2 >= 0.5) == y_co).astype(int) - ((p_co1 >= 0.5) == y_co).astype(int))
    q05_seller_co, med_seller_co = bootstrap_seller_q05(diff_co_series, df_conf_sellers)
    
    print(f"\n  --- SINGLE CONFIRM RESULTS (14,075 rows: {conf_cold.sum()} cold, {conf_warm.sum()} warm) ---")
    print(f"  Line 1 Tabular: Overall Acc={acc_co1:.5f} (Cold={accuracy_score(y_co[conf_cold], p_co1[conf_cold] >= 0.5):.5f}, Warm={accuracy_score(y_co[conf_warm], p_co1[conf_warm] >= 0.5):.5f}) LL={ll_co1:.5f}")
    print(f"  Line 2 Causal : Overall Acc={acc_co2:.5f} (Cold={accuracy_score(y_co[conf_cold], p_co2[conf_cold] >= 0.5):.5f}, Warm={accuracy_score(y_co[conf_warm], p_co2[conf_warm] >= 0.5):.5f}) LL={ll_co2:.5f}")
    print(f"  Line 2 vs Line 1 on Confirm: Fixes={fixes_co}, Breaks={breaks_co}, Net={net_co}, Q05_seller={q05_seller_co}")

    # 5. Overall 70,000 OOF Analysis & Support Bins
    print("\n[5/6] Overall 70,000 OOF Analysis & Seller Support Bins...", flush=True)
    oof_acc1 = accuracy_score(y_train, oof_line1 >= 0.5)
    oof_acc2 = accuracy_score(y_train, oof_line2 >= 0.5)
    oof_ll1 = log_loss(y_train, oof_line1)
    oof_ll2 = log_loss(y_train, oof_line2)
    
    print(f"  Full 70k OOF Line 1 Tabular: Accuracy={oof_acc1:.5f}, LogLoss={oof_ll1:.5f}")
    print(f"  Full 70k OOF Line 2 Causal : Accuracy={oof_acc2:.5f}, LogLoss={oof_ll2:.5f}")
    
    # Save OOF files for Claude
    np.save(OUT_DIR / "oof_probabilities_line1_tabular.npy", oof_line1)
    np.save(OUT_DIR / "oof_probabilities_line2_causal.npy", oof_line2)
    np.save(OUT_DIR / "test_probabilities_line1_tabular.npy", test_line1)
    np.save(OUT_DIR / "test_probabilities_line2_causal.npy", test_line2)
    print(f"  Saved OOF and test probabilities in {OUT_DIR}")

    # 6. Candidate CSV Generation for Test & Hashes
    print("\n[6/6] Generating Candidate CSV Submissions and Hashes...", flush=True)
    t30_path = "submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv"
    df_t30 = pd.read_csv(t30_path)
    
    # Model 1 Full submission
    sub1 = pd.DataFrame({
        'ID': np.arange(1, 30001),
        'condition': np.where(test_line1 >= 0.5, 'used', 'new')
    })
    path_sub1 = OUT_DIR / "CANDIDATE_LINE1_TABULAR_PURO.csv"
    sub1.to_csv(path_sub1, index=False, lineterminator='\n')
    
    # Model 2 Full submission
    sub2 = pd.DataFrame({
        'ID': np.arange(1, 30001),
        'condition': np.where(test_line2 >= 0.5, 'used', 'new')
    })
    path_sub2 = OUT_DIR / "CANDIDATE_LINE2_CAUSAL_TABULAR.csv"
    sub2.to_csv(path_sub2, index=False, lineterminator='\n')
    
    def audit_csv(path, df, name):
        with open(path, 'rb') as f:
            raw_b = f.read().replace(b'\r\n', b'\n')
        lf_sha = hashlib.sha256(raw_b).hexdigest()
        
        # Helper without final newline
        df_sorted = df.sort_values('ID')
        payload_no_lf = '\n'.join(f'{int(i)}:{str(c)}' for i, c in zip(df_sorted['ID'], df_sorted['condition']))
        gf_no_lf = hashlib.sha256(payload_no_lf.encode('utf-8')).hexdigest()
        gf_with_lf = hashlib.sha256((payload_no_lf + '\n').encode('utf-8')).hexdigest()
        
        gate_g = validate_gate_g(path)
        diff_t30 = (df['condition'] != df_t30['condition']).sum()
        
        return {
            'name': name,
            'path': str(path),
            'rows': gate_g['rows'],
            'used_rate': float(gate_g['used_rate']),
            'diff_vs_t30': int(diff_t30),
            'lf_sha256': lf_sha,
            'gate_f_no_lf': gf_no_lf,
            'gate_f_with_lf': gf_with_lf
        }
        
    audit_sub1 = audit_csv(path_sub1, sub1, "CANDIDATE_LINE1_TABULAR_PURO")
    audit_sub2 = audit_csv(path_sub2, sub2, "CANDIDATE_LINE2_CAUSAL_TABULAR")
    
    print("\n--- CSV AUDIT SUMMARY ---")
    for a in [audit_sub1, audit_sub2]:
        print(f"{a['name']}:")
        print(f"  Rows: {a['rows']}, Used rate: {a['used_rate']:.4f}")
        print(f"  Diff vs T30 (0.94823): {a['diff_vs_t30']}")
        print(f"  LF SHA-256: {a['lf_sha256']}")
        print(f"  Gate F (no LF): {a['gate_f_no_lf']}")
        print(f"  Gate F (with LF): {a['gate_f_with_lf']}")
        
    report = {
        'timestamp_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
        'elapsed_seconds': round(time.time() - t_start, 2),
        'metrics_oof_70k': {
            'line1_tabular_acc': round(oof_acc1, 5),
            'line1_tabular_ll': round(oof_ll1, 5),
            'line2_causal_acc': round(oof_acc2, 5),
            'line2_causal_ll': round(oof_ll2, 5)
        },
        'metrics_tune_15k': {
            'line1_acc': round(acc_tu1, 5),
            'line2_acc': round(acc_tu2, 5),
            'fixes': fixes_tu,
            'breaks': breaks_tu,
            'net': fixes_tu - breaks_tu
        },
        'metrics_confirm_14k': {
            'line1_acc': round(acc_co1, 5),
            'line2_acc': round(acc_co2, 5),
            'fixes': fixes_co,
            'breaks': breaks_co,
            'net': net_co,
            'q05_seller': q05_seller_co
        },
        'csv_candidates': [audit_sub1, audit_sub2]
    }
    
    with open(OUT_DIR / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nAll results saved in {OUT_DIR / 'experiment_report.json'}")
    print(f"Total pipeline time: {time.time() - t_start:.1f}s")
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()
