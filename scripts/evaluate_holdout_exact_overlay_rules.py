"""AP-2026: End-to-End Train-Only Holdout Evaluation of Exact Overlay Rules (A, B, C).

Compliance:
- Gate H: "## 0. Resumen negativo primero" in all reporting.
- Gate C: "NO_IDENTIFIABLE" for uncalibrated public confidence.
- Explicitly declares: Faithful OOF rule of S4 is impossible to reconstruct on train (S4 is a Kaggle test artifact).
- Clean end-to-end holdout on development pool (55,925 rows), NOT presented as confirm virgen (confirm 14,075 is strictly frozen).
- Evaluates the EXACT overlay rules that generate A (tau=0.80), B (A + cold tau=0.70), and C (tau=0.70) on top of the base anchor.
- Inner tune threshold selection frozen before inspecting outer holdout.
- Reports fixes, breaks, net delta, log loss, and disagreements across all five regimes.
- Computes three independent bootstraps: seller cluster, category cluster, and temporal block bootstrap.
- Coordinates with Claude (CL-0017) without duplicating cold signal computation.
- Zero Kaggle calls by agents.
"""

import os
import sys
import time
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from lightgbm import LGBMClassifier

sys.path.append(os.path.abspath("."))
from tests.test_preventive_gates import compute_id_target_hash, compute_file_sha256

def compute_causal_history(df_fit, df_eval, key='seller_id', m=5.0):
    prior = float(df_fit['y'].mean())
    sellers_fit = df_fit[key].values
    times_fit = df_fit['start_time'].values
    y_fit = df_fit['y'].values

    df_s = pd.DataFrame({
        'pos': np.arange(len(df_fit)),
        's': sellers_fit,
        't': times_fit,
        'y': y_fit
    }).sort_values(['s', 't'])

    grp = df_s.groupby('s', sort=False)
    cum_cnt = grp.cumcount().values.astype(np.float32)
    cum_y = grp['y'].cumsum().values.astype(np.float32)
    past_y = cum_y - df_s['y'].values.astype(np.float32)
    te_sorted = (past_y + m * prior) / (cum_cnt + m)

    pos_order = df_s['pos'].values
    te_fit = np.empty(len(df_fit), dtype=np.float32)
    cnt_fit = np.empty(len(df_fit), dtype=np.float32)
    te_fit[pos_order] = te_sorted
    cnt_fit[pos_order] = np.log1p(cum_cnt)

    seller_history_map = {}
    for s, sub in df_s.groupby('s'):
        seller_history_map[s] = (sub['t'].values, sub['y'].values)

    sellers_eval = df_eval[key].values
    times_eval = df_eval['start_time'].values
    te_eval = np.full(len(df_eval), prior, dtype=np.float32)
    cnt_eval = np.zeros(len(df_eval), dtype=np.float32)

    for i in range(len(df_eval)):
        s = sellers_eval[i]
        if s in seller_history_map:
            t_arr, y_arr = seller_history_map[s]
            past_mask = (t_arr < times_eval[i])
            n_past = past_mask.sum()
            if n_past > 0:
                te_eval[i] = (y_arr[past_mask].sum() + m * prior) / (n_past + m)
                cnt_eval[i] = np.log1p(n_past)

    return te_fit, cnt_fit, te_eval, cnt_eval

def compute_batch_history(df_fit, df_eval, m=5.0):
    prior = float(df_fit['y'].mean())
    sellers_fit = pd.Series(df_fit['seller_id'].values)
    y_fit = df_fit['y'].values

    grp_s = df_fit.groupby('seller_id')['y']
    s_cnt = grp_s.count()
    s_sum = grp_s.sum()

    row_cnt = sellers_fit.map(s_cnt).values
    row_sum = sellers_fit.map(s_sum).values
    loo_cnt = np.maximum(0, row_cnt - 1)
    loo_sum = np.maximum(0, row_sum - y_fit)
    te_s_fit = ((loo_sum + m * prior) / (loo_cnt + m)).astype(np.float32)
    cnt_s_fit = np.log1p(loo_cnt).astype(np.float32)

    s_map = (s_sum + m * prior) / (s_cnt + m)
    s_eval_series = pd.Series(df_eval['seller_id'].values)
    te_s_eval = s_eval_series.map(s_map).fillna(prior).values.astype(np.float32)
    cnt_s_eval = np.log1p(s_eval_series.map(s_cnt).fillna(0).values).astype(np.float32)

    return (
        np.column_stack([te_s_fit, cnt_s_fit]),
        np.column_stack([te_s_eval, cnt_s_eval])
    )

def run_cluster_bootstraps(diff, sellers, cats, timestamps, n_boot=2000, seed=20261114):
    """Computes cluster bootstrap by seller, cluster bootstrap by category, and temporal block bootstrap."""
    diff = np.asarray(diff, dtype=np.float64)
    sellers = np.asarray(sellers)
    cats = np.asarray(cats)
    timestamps = np.asarray(timestamps)
    n_items = len(diff)

    rng = np.random.default_rng(seed)

    # 1. Seller Cluster Bootstrap
    unique_s, s_idx = np.unique(sellers, return_inverse=True)
    n_s = len(unique_s)
    boot_seller = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w_s = rng.poisson(1.0, size=n_s)
        w_row = w_s[s_idx]
        norm = w_row.sum()
        boot_seller[b] = np.sum(diff * w_row) * (n_items / norm) if norm > 0 else 0.0

    # 2. Category Cluster Bootstrap
    unique_c, c_idx = np.unique(cats, return_inverse=True)
    n_c = len(unique_c)
    boot_cat = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w_c = rng.poisson(1.0, size=n_c)
        w_row = w_c[c_idx]
        norm = w_row.sum()
        boot_cat[b] = np.sum(diff * w_row) * (n_items / norm) if norm > 0 else 0.0

    # 3. Multiway Seller x Category
    boot_multiway = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w_s = rng.poisson(1.0, size=n_s)
        w_c = rng.poisson(1.0, size=n_c)
        w_row = w_s[s_idx] * w_c[c_idx]
        norm = w_row.sum()
        boot_multiway[b] = np.sum(diff * w_row) * (n_items / norm) if norm > 0 else 0.0

    # 4. Temporal Block Bootstrap
    # Divide time into 50 contiguous temporal blocks
    time_order = np.argsort(timestamps)
    block_assignments = np.empty(n_items, dtype=int)
    block_size = max(1, n_items // 50)
    for blk_idx in range(50):
        start = blk_idx * block_size
        end = (blk_idx + 1) * block_size if blk_idx < 49 else n_items
        block_assignments[time_order[start:end]] = blk_idx

    boot_temporal = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w_t = rng.poisson(1.0, size=50)
        w_row = w_t[block_assignments]
        norm = w_row.sum()
        boot_temporal[b] = np.sum(diff * w_row) * (n_items / norm) if norm > 0 else 0.0

    def get_stats(arr):
        return {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "q05": round(float(np.percentile(arr, 5)), 2),
            "median": round(float(np.median(arr)), 2),
            "q95": round(float(np.percentile(arr, 95)), 2),
        }

    return {
        "multiway": get_stats(boot_multiway),
        "seller_cluster": get_stats(boot_seller),
        "category_cluster": get_stats(boot_cat),
        "temporal_block": get_stats(boot_temporal),
        "worst_case_q05": min(
            get_stats(boot_multiway)["q05"],
            get_stats(boot_seller)["q05"],
            get_stats(boot_cat)["q05"],
            get_stats(boot_temporal)["q05"]
        )
    }

def main():
    t0 = time.time()
    print("=" * 80)
    print("EVALUATION OF EXACT OVERLAY RULES (A, B, C) IN TRAIN-ONLY HOLDOUT (CX-0201)")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/6] Loading data and splits (Confirm 14,075 strictly frozen)...")
    with open('data/splits/t98_parsed_train.pkl', 'rb') as f:
        df_train = pickle.load(f)
    df_split = pd.read_csv('data/splits/canonical_3way_splits.csv')
    df = df_train.merge(df_split[['id', 'role', 'regime']], on='id')

    confirm_mask = (df['role'] == 'confirm').values
    dev_mask = ~confirm_mask
    df_dev = df[dev_mask].copy().reset_index(drop=True)

    print(f"Total train: {len(df)} rows")
    print(f"Frozen Confirm: {confirm_mask.sum()} rows (NOT used, NOT presented as virgen)")
    print(f"Development pool: {len(df_dev)} rows")

    # 2. Honest End-to-End Holdout Setup on Development Pool
    # Outer-Fit (38,459 rows) vs Outer-Holdout (17,466 rows)
    # Inside Outer-Fit: Inner-Fit (31,214 rows) vs Inner-Tune (7,245 rows)
    print("\n[2/6] Setting up End-to-End Holdout and Inner-Tune partitions...")
    rng = np.random.default_rng(20261114)
    dev_sellers = np.array(df_dev['seller_id'].unique())
    rng.shuffle(dev_sellers)

    n_sellers = len(dev_sellers)
    n_cold_val = int(n_sellers * 0.15)
    n_pure_tr = int(n_sellers * 0.15)

    pure_cold_sellers = set(dev_sellers[:n_cold_val])
    pure_tr_sellers = set(dev_sellers[n_cold_val:n_cold_val + n_pure_tr])
    shared_sellers = set(dev_sellers[n_cold_val + n_pure_tr:])

    seller_idxs = df_dev.groupby('seller_id').indices
    ho_indices = []
    fit_indices = []

    for s in pure_cold_sellers:
        ho_indices.extend(seller_idxs[s])
    for s in pure_tr_sellers:
        fit_indices.extend(seller_idxs[s])

    for s in shared_sellers:
        idxs = seller_idxs[s]
        if len(idxs) == 1:
            fit_indices.extend(idxs)
        else:
            sub = df_dev.iloc[idxs]
            p70 = sub['start_time'].quantile(0.70)
            v_m = sub['start_time'] > p70
            if v_m.all() or (~v_m).all():
                v_m = np.zeros(len(idxs), dtype=bool)
                v_m[int(len(idxs) * 0.70):] = True
            for i, vm in zip(idxs, v_m):
                if vm:
                    ho_indices.append(i)
                else:
                    fit_indices.append(i)

    df_fit = df_dev.iloc[fit_indices].copy().reset_index(drop=True)
    df_ho = df_dev.iloc[ho_indices].copy().reset_index(drop=True)

    print(f"Fit pool rows: {len(df_fit)}")
    print(f"Holdout pool rows: {len(df_ho)}")

    # Regimes in Holdout relative to Fit pool
    fit_sellers_set = set(df_fit['seller_id'].unique())
    fit_times = df_fit.groupby('seller_id')['start_time'].agg(['min', 'max'])
    fit_stamps = df_fit.groupby('seller_id')['start_time'].apply(set).to_dict()

    s_ho = df_ho['seller_id'].values
    t_ho = df_ho['start_time'].values

    in_fit_ho = np.isin(s_ho, list(fit_sellers_set))
    bounds_ho = fit_times.reindex(s_ho)
    t_min_ho = bounds_ho['min'].values
    t_max_ho = bounds_ho['max'].values

    r_cold_ho = ~in_fit_ho
    has_same_time_ho = np.zeros(len(df_ho), dtype=bool)
    for k in range(len(df_ho)):
        s = s_ho[k]
        t = t_ho[k]
        if s in fit_stamps and t in fit_stamps[s]:
            has_same_time_ho[k] = True

    r_ties_ho = in_fit_ho & has_same_time_ho
    r_bracketed_ho = in_fit_ho & ~has_same_time_ho & (t_ho > t_min_ho) & (t_ho < t_max_ho)
    r_before_ho = in_fit_ho & ~has_same_time_ho & (t_ho < t_min_ho)
    r_after_ho = in_fit_ho & ~has_same_time_ho & (t_ho > t_max_ho)

    ho_regimes = {
        "bracketed (Warm)": r_bracketed_ho,
        "same_timestamp (Warm)": r_ties_ho,
        "before_train (Warm)": r_before_ho,
        "after_train (Warm)": r_after_ho,
        "new_seller (Cold)": r_cold_ho,
        "all_warm": in_fit_ho,
        "all": np.ones(len(df_ho), dtype=bool)
    }

    print("\nHoldout 5 Regimes Breakdown:")
    for rname, rmask in ho_regimes.items():
        print(f"  {rname:25}: {rmask.sum():5d} rows ({rmask.sum()/len(df_ho)*100:.1f}%)")

    # Within Fit pool, set aside Inner-Tune to select thresholds
    inner_fit_idxs = []
    inner_tune_idxs = []
    fit_seller_idxs = df_fit.groupby('seller_id').indices
    for s, idxs in fit_seller_idxs.items():
        if len(idxs) == 1:
            inner_fit_idxs.extend(idxs)
        else:
            sub = df_fit.iloc[idxs]
            p70_t = sub['start_time'].quantile(0.70)
            t_m = sub['start_time'] > p70_t
            if t_m.all() or (~t_m).all():
                t_m = np.zeros(len(idxs), dtype=bool)
                t_m[int(len(idxs) * 0.70):] = True
            for i, tm in zip(idxs, t_m):
                if tm:
                    inner_tune_idxs.append(i)
                else:
                    inner_fit_idxs.append(i)

    df_in_fit = df_fit.iloc[inner_fit_idxs].copy().reset_index(drop=True)
    df_in_tune = df_fit.iloc[inner_tune_idxs].copy().reset_index(drop=True)
    print(f"\nInner Fit rows: {len(df_in_fit)}, Inner Tune rows: {len(df_in_tune)}")

    # 3. Inner-Tune Threshold Selection
    print("\n[3/6] Running Inner-Tune Threshold Selection (tau grid)...")
    meta_cols = [
        'price', 'init_qty', 'avail_qty', 'sold_qty', 'free_shipping', 'local_pick_up',
        'accepts_mercadopago', 'automatic_relist', 'has_official_store', 'has_video',
        'has_catalog', 'num_attrs', 'num_pics', 'num_vars', 'num_deals', 'num_tags',
        'title_len', 'title_words', 'title_digits', 'title_uppers', 'warr_sin_garantia'
    ]

    vec_it = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=3, sublinear_tf=True)
    X_tx_if = vec_it.fit_transform(df_in_fit['title'].fillna(''))
    X_tx_it = vec_it.transform(df_in_tune['title'].fillna(''))

    clf_tx_it = LogisticRegression(C=1.5, max_iter=250, random_state=20261114)
    clf_tx_it.fit(X_tx_if, df_in_fit['y'])
    p_tx_if = clf_tx_it.predict_proba(X_tx_if)[:, 1, None]
    p_tx_it = clf_tx_it.predict_proba(X_tx_it)[:, 1, None]

    svd_it = TruncatedSVD(n_components=16, random_state=20261114)
    svd_if = svd_it.fit_transform(X_tx_if).astype(np.float32)
    svd_it_tune = svd_it.transform(X_tx_it).astype(np.float32)

    c_m_if, c_c_if, c_m_it, c_c_it = compute_causal_history(df_in_fit, df_in_tune)
    b_if, b_it = compute_batch_history(df_in_fit, df_in_tune)

    X_base_if = np.column_stack([p_tx_if, svd_if, df_in_fit[meta_cols].to_numpy(dtype=np.float32), c_m_if, c_c_if])
    X_base_it = np.column_stack([p_tx_it, svd_it_tune, df_in_tune[meta_cols].to_numpy(dtype=np.float32), c_m_it, c_c_it])

    X_batch_if = np.column_stack([X_base_if, b_if])
    X_batch_it = np.column_stack([X_base_it, b_it])

    lgb_params = dict(n_estimators=350, learning_rate=0.03, num_leaves=31, colsample_bytree=0.8, subsample=0.8, random_state=20261114, n_jobs=-1, verbose=-1)
    m_base_it = LGBMClassifier(**lgb_params).fit(X_base_if, df_in_fit['y'])
    m_batch_it = LGBMClassifier(**lgb_params).fit(X_batch_if, df_in_fit['y'])

    p_base_it = m_base_it.predict_proba(X_base_it)[:, 1]
    p_batch_it = m_batch_it.predict_proba(X_batch_it)[:, 1]

    # Inner-tune evaluation
    seen_if = set(df_in_fit['seller_id'].unique())
    warm_it_m = df_in_tune['seller_id'].isin(seen_if).values
    c_base_it = (p_base_it >= 0.5)
    y_it = df_in_tune['y'].to_numpy(dtype=int)

    tune_results = {}
    for tau in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        c_ov = c_base_it.copy()
        c_ov[(p_batch_it >= tau) & warm_it_m] = True
        c_ov[(p_batch_it <= 1.0 - tau) & warm_it_m] = False

        diff = (c_ov[warm_it_m] == y_it[warm_it_m]).astype(int) - (c_base_it[warm_it_m] == y_it[warm_it_m]).astype(int)
        fixes = int(np.sum(diff > 0))
        breaks = int(np.sum(diff < 0))
        net = fixes - breaks
        flips = int(np.sum(c_ov[warm_it_m] != c_base_it[warm_it_m]))
        prec = fixes / flips if flips > 0 else 0.0

        tune_results[f"{tau:.2f}"] = {
            "tau": tau, "flips": flips, "net": net, "fixes": fixes, "breaks": breaks, "precision": round(prec, 4)
        }
        print(f"  Inner Tune tau={tau:.2f}: Flips={flips:3d} | Net={net:+3d} (Fix={fixes:3d}, Brk={breaks:3d}) | Prec={prec:.1%}")

    # 4. Train Models on Full Fit Pool (38,459 rows) and Evaluate on Holdout Pool (17,466 rows)
    print("\n[4/6] Fitting models on Fit Pool (38,459 rows) and generating predictions on Holdout...")
    vec_fit = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=3, sublinear_tf=True)
    X_tx_fit = vec_fit.fit_transform(df_fit['title'].fillna(''))
    X_tx_ho = vec_fit.transform(df_ho['title'].fillna(''))

    clf_tx_fit = LogisticRegression(C=1.5, max_iter=250, random_state=20261114)
    clf_tx_fit.fit(X_tx_fit, df_fit['y'])
    p_tx_fit = clf_tx_fit.predict_proba(X_tx_fit)[:, 1, None]
    p_tx_ho = clf_tx_fit.predict_proba(X_tx_ho)[:, 1, None]

    svd_fit = TruncatedSVD(n_components=16, random_state=20261114)
    svd_fit_tr = svd_fit.fit_transform(X_tx_fit).astype(np.float32)
    svd_fit_ho = svd_fit.transform(X_tx_ho).astype(np.float32)

    c_m_f, c_c_f, c_m_h, c_c_h = compute_causal_history(df_fit, df_ho)
    b_f, b_h = compute_batch_history(df_fit, df_ho)

    X_base_fit = np.column_stack([p_tx_fit, svd_fit_tr, df_fit[meta_cols].to_numpy(dtype=np.float32), c_m_f, c_c_f])
    X_base_ho = np.column_stack([p_tx_ho, svd_fit_ho, df_ho[meta_cols].to_numpy(dtype=np.float32), c_m_h, c_c_h])

    X_batch_fit = np.column_stack([X_base_fit, b_f])
    X_batch_ho = np.column_stack([X_base_ho, b_h])

    # Fit Base and Batch
    m_base = LGBMClassifier(**lgb_params).fit(X_base_fit, df_fit['y'])
    m_batch = LGBMClassifier(**lgb_params).fit(X_batch_fit, df_fit['y'])

    p_base_ho = m_base.predict_proba(X_base_ho)[:, 1]
    p_batch_ho = m_batch.predict_proba(X_batch_ho)[:, 1]

    # Fit Cold Specialist on cold seller outer-CV inside fit pool (for Rule B simulation)
    # Using text + meta without seller history on cold sellers
    cold_sellers_fit = set(df_fit['seller_id'].value_counts()[df_fit['seller_id'].value_counts() == 1].index)
    cold_fit_mask = df_fit['seller_id'].isin(cold_sellers_fit).values
    m_cold = LGBMClassifier(**lgb_params).fit(X_base_fit[cold_fit_mask], df_fit['y'].values[cold_fit_mask])
    p_cold_ho = m_cold.predict_proba(X_base_ho)[:, 1]

    # Base predictions on Holdout (the pseudo-S4 anchor on holdout)
    c_base_ho = (p_base_ho >= 0.5)
    y_ho = df_ho['y'].to_numpy(dtype=int)

    ll_base_ho = float(log_loss(y_ho, p_base_ho))
    acc_base_ho = float(accuracy_score(y_ho, c_base_ho))
    print(f"\nBase Model on Holdout (N={len(df_ho)}): Accuracy={acc_base_ho:.4f}, LogLoss={ll_base_ho:.4f}")

    # 5. Evaluate Exact Overlay Rules A, B, C on Holdout
    print("\n[5/6] Evaluating Exact Overlay Rules A, B, C across the Five Regimes on Holdout...")
    warm_ho_m = ho_regimes["all_warm"]
    cold_ho_m = ho_regimes["new_seller (Cold)"]

    # Rule A: tau=0.80 on warm, 0 on cold
    c_rule_a = c_base_ho.copy()
    c_rule_a[(p_batch_ho >= 0.80) & warm_ho_m] = True
    c_rule_a[(p_batch_ho <= 0.20) & warm_ho_m] = False

    # Rule B: Rule A + cold tau=0.70
    c_rule_b = c_rule_a.copy()
    c_rule_b[(p_cold_ho >= 0.70) & cold_ho_m] = True
    c_rule_b[(p_cold_ho <= 0.30) & cold_ho_m] = False

    # Rule C: tau=0.70 on warm, 0 on cold
    c_rule_c = c_base_ho.copy()
    c_rule_c[(p_batch_ho >= 0.70) & warm_ho_m] = True
    c_rule_c[(p_batch_ho <= 0.30) & warm_ho_m] = False

    # Rule Clean (AG-0223): tau=0.90 on warm, 0 on cold
    c_rule_clean = c_base_ho.copy()
    c_rule_clean[(p_batch_ho >= 0.90) & warm_ho_m] = True
    c_rule_clean[(p_batch_ho <= 0.10) & warm_ho_m] = False

    rules = {
        "Rule_Clean (tau=0.90 Warm, 0 Cold) [Analog of Clean Candidate, 161 flips]": c_rule_clean,
        "Rule_A (tau=0.80 Warm, 0 Cold) [Analog of Cand A, 303 flips]": c_rule_a,
        "Rule_B (Warm + Cold tau=0.70) [Analog of Cand B, 379 flips]": c_rule_b,
        "Rule_C (tau=0.70 Warm, 0 Cold) [Analog of Cand C, 483 flips]": c_rule_c,
    }

    eval_rules_report = {}

    for rule_name, c_pred in rules.items():
        diff_all = (c_pred == y_ho).astype(int) - (c_base_ho == y_ho).astype(int)
        flips_all = (c_pred != c_base_ho)
        n_flips = int(flips_all.sum())
        fixes_all = int(np.sum(diff_all > 0))
        breaks_all = int(np.sum(diff_all < 0))
        net_all = fixes_all - breaks_all
        prec_all = fixes_all / n_flips if n_flips > 0 else 0.0

        # Construct coherent probability policy for overlay:
        # Probabilities are altered ONLY on rows where the discrete flip was triggered.
        p_eval_overlay = p_base_ho.copy()
        if "Rule_A" in rule_name or "Rule_C" in rule_name or "Rule_Clean" in rule_name:
            p_eval_overlay[flips_all] = p_batch_ho[flips_all]
        elif "Rule_B" in rule_name:
            p_eval_overlay[flips_all & warm_ho_m] = p_batch_ho[flips_all & warm_ho_m]
            p_eval_overlay[flips_all & cold_ho_m] = p_cold_ho[flips_all & cold_ho_m]

        ll_overlay = float(log_loss(y_ho, p_eval_overlay))
        brier_overlay = float(brier_score_loss(y_ho, p_eval_overlay))
        brier_base = float(brier_score_loss(y_ho, p_base_ho))

        # Conditional metrics on the flip support itself
        if n_flips > 0:
            ll_flips_base = float(log_loss(y_ho[flips_all], p_base_ho[flips_all], labels=[0, 1]))
            ll_flips_cand = float(log_loss(y_ho[flips_all], p_eval_overlay[flips_all], labels=[0, 1]))
            brier_flips_base = float(brier_score_loss(y_ho[flips_all], p_base_ho[flips_all]))
            brier_flips_cand = float(brier_score_loss(y_ho[flips_all], p_eval_overlay[flips_all]))
        else:
            ll_flips_base = 0.0
            ll_flips_cand = 0.0
            brier_flips_base = 0.0
            brier_flips_cand = 0.0

        # Also reference global continuous replacement of warm (which AG-0224 previously reported)
        p_batch_all_warm = p_base_ho.copy()
        p_batch_all_warm[warm_ho_m] = p_batch_ho[warm_ho_m]
        ll_continuous_warm = float(log_loss(y_ho, p_batch_all_warm, labels=[0, 1]))

        print(f"\n=======================================================")
        print(f"{rule_name}:")
        print(f"  Total Flips vs Base: {n_flips} | Net Delta: {net_all:+d} (Fixes={fixes_all}, Breaks={breaks_all}) | Prec={prec_all:.1%}")
        print(f"  Overlay LogLoss: {ll_overlay:.6f} vs Base {ll_base_ho:.6f} (Delta: {ll_overlay - ll_base_ho:+.6f})")
        print(f"  Overlay Brier:   {brier_overlay:.6f} vs Base {brier_base:.6f} (Delta: {brier_overlay - brier_base:+.6f})")
        if n_flips > 0:
            print(f"  Conditional on Flips (N={n_flips}): Brier={brier_flips_cand:.4f} vs Base {brier_flips_base:.4f} | LL={ll_flips_cand:.4f} vs {ll_flips_base:.4f}")
        print(f"  [Ref] Continuous Warm Model LogLoss: {ll_continuous_warm:.4f} (NOT the overlay)")

        # Breakdown by 5 regimes
        regime_eval = {}
        for rname in ["bracketed (Warm)", "same_timestamp (Warm)", "before_train (Warm)", "after_train (Warm)", "new_seller (Cold)"]:
            rmask = ho_regimes[rname]
            diff_r = diff_all[rmask]
            flips_r = flips_all[rmask]
            n_fl_r = int(flips_r.sum())
            fx_r = int(np.sum(diff_r > 0))
            br_r = int(np.sum(diff_r < 0))
            net_r = fx_r - br_r
            pr_r = fx_r / n_fl_r if n_fl_r > 0 else 0.0
            ll_r_base = float(log_loss(y_ho[rmask], p_base_ho[rmask], labels=[0, 1]))
            ll_r_cand = float(log_loss(y_ho[rmask], p_eval_overlay[rmask], labels=[0, 1]))
            eval_status = "EVALUADA" if n_fl_r > 0 else "NO_EVALUADA_CERO_FLIPS"
            regime_eval[rname] = {
                "eval_status": eval_status,
                "n_rows": int(rmask.sum()),
                "flips": n_fl_r,
                "net_delta": net_r,
                "fixes": fx_r,
                "breaks": br_r,
                "precision": round(pr_r, 4),
                "logloss_base": round(ll_r_base, 4),
                "logloss_cand": round(ll_r_cand, 4),
            }
            status_str = f"Flips={n_fl_r:3d}" if n_fl_r > 0 else "CERO FLIPS (NO EVALUADA)"
            print(f"    {rname:25}: {status_str:25} | Net={net_r:+3d} (Fix={fx_r:2d}, Brk={br_r:2d}) | Prec={pr_r:5.1%}")

        # Bootstraps
        print("  Running Bootstraps (Seller, Category, Multiway, Temporal)...")
        boot_res = run_cluster_bootstraps(
            diff_all,
            sellers=df_ho['seller_id'].values,
            cats=df_ho['category_id'].values,
            timestamps=df_ho['start_time'].values,
            n_boot=2000,
            seed=20261114
        )
        print(f"    Bootstrap Multiway [Q05, Med, Q95]: [{boot_res['multiway']['q05']:+5.1f}, {boot_res['multiway']['median']:+5.1f}, {boot_res['multiway']['q95']:+5.1f}]")
        print(f"    Marginal Seller    [Q05, Med, Q95]: [{boot_res['seller_cluster']['q05']:+5.1f}, {boot_res['seller_cluster']['median']:+5.1f}, {boot_res['seller_cluster']['q95']:+5.1f}]")
        print(f"    Marginal Category  [Q05, Med, Q95]: [{boot_res['category_cluster']['q05']:+5.1f}, {boot_res['category_cluster']['median']:+5.1f}, {boot_res['category_cluster']['q95']:+5.1f}]")
        print(f"    Temporal Block     [Q05, Med, Q95]: [{boot_res['temporal_block']['q05']:+5.1f}, {boot_res['temporal_block']['median']:+5.1f}, {boot_res['temporal_block']['q95']:+5.1f}]")
        print(f"    --> Worst-Case Q05: {boot_res['worst_case_q05']:+5.1f}")

        eval_rules_report[rule_name] = {
            "total_flips": n_flips,
            "net_delta": net_all,
            "fixes": fixes_all,
            "breaks": breaks_all,
            "precision": round(prec_all, 4),
            "logloss_overlay": round(ll_overlay, 6),
            "logloss_base": round(ll_base_ho, 6),
            "brier_overlay": round(brier_overlay, 6),
            "brier_base": round(brier_base, 6),
            "conditional_flips_brier_cand": round(brier_flips_cand, 6),
            "conditional_flips_brier_base": round(brier_flips_base, 6),
            "conditional_flips_ll_cand": round(ll_flips_cand, 6),
            "conditional_flips_ll_base": round(ll_flips_base, 6),
            "reference_continuous_warm_logloss": round(ll_continuous_warm, 4),
            "by_regime": regime_eval,
            "bootstraps": boot_res
        }

    # 6. Save JSON Report
    report = {
        "timestamp_iso": pd.Timestamp.now().isoformat(),
        "runner": "scripts/evaluate_holdout_exact_overlay_rules.py",
        "structural_declaration": "No es posible reconstruir una regla OOF fiel a S4 en train; S4 es un artefacto puntuado en test. Se evalúan las reglas exactas sobre holdout end-to-end.",
        "confirm_rows_frozen": int(confirm_mask.sum()),
        "dev_pool_rows": len(df_dev),
        "fit_pool_rows": len(df_fit),
        "holdout_pool_rows": len(df_ho),
        "inner_tune_selection": tune_results,
        "exact_rules_evaluation_holdout": eval_rules_report,
        "coordination_claude": "CL-0017 ACK registrado en 18d3bbc; no se duplica entrenamiento cold ni holdout externo de Claude.",
        "elapsed_seconds": round(time.time() - t0, 2)
    }

    out_json = 'results/holdout_exact_overlay_rules_report.json'
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport successfully saved to {out_json} in {time.time()-t0:.2f}s")
    print("=" * 80)

if __name__ == '__main__':
    main()
