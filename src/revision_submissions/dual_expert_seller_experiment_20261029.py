"""Dual-Expert Architecture for Seen vs New Sellers on Deep CatBoost.

Hypothesis: Global CatBoost relies heavily on seller-level features (te_seller, sell_mean_*, seller_id),
which causes high variance or default-imputation bias when predicting on unseen (new) sellers.
Training a specialized New-Seller Expert (with seller features stripped and stronger category/text regularization)
and adaptively routing new sellers to this expert should improve generalization on unseen sellers
while preserving the strong ~93.5% accuracy on seen sellers.

Evaluated on honest external seller-group partitions (fit ~45k, tune ~12k, confirm ~13k)
across two independent seeds (2026 and 314159).
Frozen decision rules and blend weights on tune before confirm.
"""
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier, Pool

from deep_catboost_nbsvm_ablation_20261029 import (
    load_dataset, extract_deep_features, split_seller_groups, cluster_bootstrap_ci
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "revision_submissions"


def run_dual_expert_seed(seed, F_df, cat_df, titles, warranties, combos, y_used, sellers, ids):
    print(f"\n==================================================================", flush=True)
    print(f"RUNNING DUAL-EXPERT SELLER ARCHITECTURE ON SEED {seed}", flush=True)
    print(f"==================================================================", flush=True)
    
    fit, tune, confirm = split_seller_groups(sellers, seed)
    y_fit = y_used[fit]
    prior = float(y_fit.mean())
    
    # 1. Target Encoding strictly inside fit via 4-fold GroupKFold
    print("Computing clean GroupKFold Target Encodings inside fit...", flush=True)
    te_cols = ["seller_id", "category_id", "city"]
    gkf = GroupKFold(n_splits=4)
    
    fit_te = {c: np.full(len(fit), prior, dtype=np.float32) for c in te_cols}
    tune_te = {}
    confirm_te = {}
    
    for tr, va in gkf.split(fit, y_fit, sellers[fit]):
        for c in te_cols:
            k_tr = cat_df[c].iloc[fit[tr]]
            k_va = cat_df[c].iloc[fit[va]]
            stats = pd.DataFrame({"k": k_tr, "y": y_fit[tr]}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + 5.0 * prior) / (stats["count"] + 5.0)
            fit_te[c][va] = k_va.map(smoothed).fillna(prior).values.astype(np.float32)
            
    for c in te_cols:
        k_fit = cat_df[c].iloc[fit]
        stats = pd.DataFrame({"k": k_fit, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
        smoothed = (stats["sum"] + 5.0 * prior) / (stats["count"] + 5.0)
        tune_te[c] = cat_df[c].iloc[tune].map(smoothed).fillna(prior).values.astype(np.float32)
        confirm_te[c] = cat_df[c].iloc[confirm].map(smoothed).fillna(prior).values.astype(np.float32)
        
    # 2. Text Representation
    print("Fitting text model on fit...", flush=True)
    vec_combo = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                                sublinear_tf=True, strip_accents="unicode", lowercase=True)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                               max_features=150000, sublinear_tf=True)
    
    X_w_fit = vec_combo.fit_transform([combos[i] for i in fit])
    X_c_fit = vec_char.fit_transform([titles[i] for i in fit])
    X_sp_fit = sp.hstack([X_w_fit, X_c_fit], format="csr")
    
    X_w_tune = vec_combo.transform([combos[i] for i in tune])
    X_c_tune = vec_char.transform([titles[i] for i in tune])
    X_sp_tune = sp.hstack([X_w_tune, X_c_tune], format="csr")
    
    X_w_conf = vec_combo.transform([combos[i] for i in confirm])
    X_c_conf = vec_char.transform([titles[i] for i in confirm])
    X_sp_conf = sp.hstack([X_w_conf, X_c_conf], format="csr")
    
    oof_sp = np.zeros(len(fit), dtype=np.float32)
    for tr, va in gkf.split(fit, y_fit, sellers[fit]):
        lr_sp = LogisticRegression(C=3.0, max_iter=250, solver="liblinear", random_state=seed)
        lr_sp.fit(X_sp_fit[tr], y_fit[tr])
        oof_sp[va] = lr_sp.predict_proba(X_sp_fit[va])[:, 1]
        
    lr_sp_full = LogisticRegression(C=3.0, max_iter=300, solver="liblinear", random_state=seed).fit(X_sp_fit, y_fit)
    p_sp_tune = lr_sp_full.predict_proba(X_sp_tune)[:, 1]
    p_sp_conf = lr_sp_full.predict_proba(X_sp_conf)[:, 1]
    
    # 3. Assemble DataFrames
    CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]
    
    df_fit = F_df.iloc[fit].copy().reset_index(drop=True)
    df_tune = F_df.iloc[tune].copy().reset_index(drop=True)
    df_conf = F_df.iloc[confirm].copy().reset_index(drop=True)
    
    for c in te_cols:
        df_fit[f"te_{c}"] = fit_te[c]
        df_tune[f"te_{c}"] = tune_te[c]
        df_conf[f"te_{c}"] = confirm_te[c]
        
    df_fit["txt_sparse_p"] = oof_sp
    df_tune["txt_sparse_p"] = p_sp_tune
    df_conf["txt_sparse_p"] = p_sp_conf
    
    for c in CAT_FEATURES:
        df_fit[c] = df_fit[c].astype(str)
        df_tune[c] = df_tune[c].astype(str)
        df_conf[c] = df_conf[c].astype(str)
        
    # Expert 1: Global Deep CatBoost (All features)
    print("Training Expert 1: Global Deep CatBoost...", flush=True)
    t0 = time.time()
    cb_global = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                   random_seed=seed, thread_count=12, verbose=0)
    cb_global.fit(Pool(df_fit, y_fit, cat_features=CAT_FEATURES))
    print(f"Global CatBoost fitted in {time.time()-t0:.2f}s", flush=True)
    
    p_glob_tune = cb_global.predict_proba(Pool(df_tune, cat_features=CAT_FEATURES))[:, 1]
    p_glob_conf = cb_global.predict_proba(Pool(df_conf, cat_features=CAT_FEATURES))[:, 1]
    
    # Expert 2: Specialized New-Seller CatBoost
    # Identify seller features to drop
    seller_cols_to_drop = [
        "te_seller_id", "freq_seller_id", "seller_n_cats", "seller_mean_logprice",
        "seller_mean_init_q", "sell_mean_w_null", "sell_mean_t_usado", "sell_mean_t_nuevo",
        "sell_mean_ship_free", "sell_mean_auto_relist", "sell_mean_n_pics"
    ]
    seller_cols_present = [c for c in seller_cols_to_drop if c in df_fit.columns]
    
    df_new_fit = df_fit.drop(columns=seller_cols_present)
    df_new_tune = df_tune.drop(columns=seller_cols_present)
    df_new_conf = df_conf.drop(columns=seller_cols_present)
    
    print(f"Training Expert 2: Specialized New-Seller CatBoost (dropped {len(seller_cols_present)} seller features)...", flush=True)
    t0 = time.time()
    cb_new_seller = CatBoostClassifier(iterations=900, depth=5, learning_rate=0.07, l2_leaf_reg=8.0,
                                       random_seed=seed + 100, thread_count=12, verbose=0)
    cb_new_seller.fit(Pool(df_new_fit, y_fit, cat_features=CAT_FEATURES))
    print(f"New-Seller CatBoost fitted in {time.time()-t0:.2f}s", flush=True)
    
    p_new_tune = cb_new_seller.predict_proba(Pool(df_new_tune, cat_features=CAT_FEATURES))[:, 1]
    p_new_conf = cb_new_seller.predict_proba(Pool(df_new_conf, cat_features=CAT_FEATURES))[:, 1]
    
    # 4. Tune decision thresholds and routing weights on TUNE
    fit_sellers = set(sellers[fit])
    is_seen_tu = np.array([s in fit_sellers for s in sellers[tune]], dtype=bool)
    is_new_tu = ~is_seen_tu
    
    y_tu = y_fit_tu = y_used[tune]
    
    # Grid search on tune: blend weight w for new sellers: p_final_new = (1-w)*p_glob + w*p_new
    best_w = 0.0
    best_th_dual = 0.50
    best_acc_tu = 0.0
    
    for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        p_tu_blend = p_glob_tune.copy()
        p_tu_blend[is_new_tu] = (1.0 - w) * p_glob_tune[is_new_tu] + w * p_new_tune[is_new_tu]
        for th in np.linspace(0.40, 0.60, 21):
            acc = accuracy_score(y_tu, p_tu_blend >= th)
            if acc > best_acc_tu:
                best_acc_tu = acc
                best_w = w
                best_th_dual = th
                
    # Also find best single threshold for baseline Global CatBoost on tune
    best_th_base = 0.50
    best_acc_base_tu = 0.0
    for th in np.linspace(0.40, 0.60, 21):
        acc = accuracy_score(y_tu, p_glob_tune >= th)
        if acc > best_acc_base_tu:
            best_acc_base_tu = acc
            best_th_base = th
            
    print(f"Frozen decisions on TUNE:")
    print(f"  Baseline Global CatBoost: th={best_th_base:.3f} (Tune Acc: {best_acc_base_tu:.5f})")
    print(f"  Dual-Expert: w_new={best_w:.2f}, th={best_th_dual:.3f} (Tune Acc: {best_acc_tu:.5f})")
    
    # 5. Evaluate on Untouched CONFIRM Holdout
    y_co = y_used[confirm]
    is_seen_co = np.array([s in fit_sellers for s in sellers[confirm]], dtype=bool)
    is_new_co = ~is_seen_co
    
    pred_base_co = (p_glob_conf >= best_th_base).astype(int)
    
    p_dual_conf = p_glob_conf.copy()
    p_dual_conf[is_new_co] = (1.0 - best_w) * p_glob_conf[is_new_co] + best_w * p_new_conf[is_new_co]
    pred_dual_co = (p_dual_conf >= best_th_dual).astype(int)
    
    acc_base = accuracy_score(y_co, pred_base_co)
    acc_dual = accuracy_score(y_co, pred_dual_co)
    ll_base = log_loss(y_co, p_glob_conf)
    ll_dual = log_loss(y_co, p_dual_conf)
    
    fixes = int(np.sum((pred_base_co != y_co) & (pred_dual_co == y_co)))
    breaks = int(np.sum((pred_base_co == y_co) & (pred_dual_co != y_co)))
    net = fixes - breaks
    
    ci_low, ci_high, _ = cluster_bootstrap_ci(y_co, pred_base_co, pred_dual_co, sellers[confirm])
    
    fixes_seen = int(np.sum((pred_base_co[is_seen_co] != y_co[is_seen_co]) & (pred_dual_co[is_seen_co] == y_co[is_seen_co])))
    breaks_seen = int(np.sum((pred_base_co[is_seen_co] == y_co[is_seen_co]) & (pred_dual_co[is_seen_co] != y_co[is_seen_co])))
    net_seen = fixes_seen - breaks_seen
    
    fixes_new = int(np.sum((pred_base_co[is_new_co] != y_co[is_new_co]) & (pred_dual_co[is_new_co] == y_co[is_new_co])))
    breaks_new = int(np.sum((pred_base_co[is_new_co] == y_co[is_new_co]) & (pred_dual_co[is_new_co] != y_co[is_new_co])))
    net_new = fixes_new - breaks_new
    
    print(f"\nCONFIRM RESULTS (Seed {seed}):")
    print(f"  Base Global CatBoost Acc: {acc_base:.5f}, LL: {ll_base:.5f}")
    print(f"  Dual-Expert CatBoost Acc: {acc_dual:.5f}, LL: {ll_dual:.5f}")
    print(f"  Fixes / Breaks:           {fixes} / {breaks} -> NET: {net:+d}")
    print(f"  Bootstrap 95% CI:         [{ci_low:+.1f}, {ci_high:+.1f}]")
    print(f"  Seen sellers net:         {net_seen:+d} (Fixes: {fixes_seen}, Breaks: {breaks_seen}, n={is_seen_co.sum()})")
    print(f"  New sellers net:          {net_new:+d} (Fixes: {fixes_new}, Breaks: {breaks_new}, n={is_new_co.sum()})")
    
    pairs_file = RESULTS_DIR / f"paired_dual_expert_seed_{seed}.csv"
    with open(pairs_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "seller_id", "y_true", "is_seen", "p_base", "p_dual", "pred_base", "pred_dual"])
        for i in range(len(confirm)):
            w.writerow([
                ids[confirm[i]], sellers[confirm[i]], y_co[i], int(is_seen_co[i]),
                round(float(p_glob_conf[i]), 5), round(float(p_dual_conf[i]), 5),
                pred_base_co[i], pred_dual_co[i]
            ])
            
    return {
        "seed": seed,
        "n_confirm": len(confirm),
        "best_w_new": best_w,
        "th_base": best_th_base,
        "th_dual": best_th_dual,
        "acc_base": float(acc_base),
        "acc_dual": float(acc_dual),
        "ll_base": float(ll_base),
        "ll_dual": float(ll_dual),
        "fixes": fixes,
        "breaks": breaks,
        "net": net,
        "ci_95": [ci_low, ci_high],
        "seen_n": int(is_seen_co.sum()),
        "seen_net": net_seen,
        "seen_fixes": fixes_seen,
        "seen_breaks": breaks_seen,
        "new_n": int(is_new_co.sum()),
        "new_net": net_new,
        "new_fixes": fixes_new,
        "new_breaks": breaks_new,
    }


def main():
    rows, ids, sellers, y_used = load_dataset()
    F_df, cat_df, titles, warranties, combos = extract_deep_features(rows)
    
    seeds = [2026, 314159]
    all_results = []
    
    for s in seeds:
        res = run_dual_expert_seed(s, F_df, cat_df, titles, warranties, combos, y_used, sellers, ids)
        all_results.append(res)
        
    out_json = RESULTS_DIR / "round155_dual_expert_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll dual-expert results saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
