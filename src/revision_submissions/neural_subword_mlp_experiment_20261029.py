"""Neural Subword Embedding (SVD + MLP) Feature Integration into Deep CatBoost.

Extracts deep tabular features (from Round 126 / feats.py: pic stats, dates,
ratios, regex flags, payment methods, seller unsupervised aggregations)
combined with:
1. Leak-free Target Encodings (seller, category, city)
2. Sparse TF-IDF linear probability (txt_sparse_p)
3. Neural Subword Embedding probability (txt_mlp_p):
   - Subword character n-grams (3-6) + word n-grams (1-2)
   - TruncatedSVD (64 continuous latent semantic components)
   - Multi-Layer Perceptron (64 -> 32 -> 1, ReLU, Adam, alpha=1e-3)
   - GroupKFold by seller internal OOF inside fit.

Evaluates on honest external seller-group partitions (fit ~45k, tune ~12k, confirm ~13k)
across two independent seeds (2026 and 314159).
Tests whether neural continuous subword representations complement Deep CatBoost.
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier, Pool

from deep_catboost_nbsvm_ablation_20261029 import (
    load_dataset, extract_deep_features, split_seller_groups, cluster_bootstrap_ci
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "revision_submissions"


def run_neural_ablation(seed, F_df, cat_df, titles, warranties, combos, y_used, sellers, ids):
    print(f"\n==================================================================", flush=True)
    print(f"RUNNING NEURAL SUBWORD (SVD+MLP) ABLATION ON SEED {seed}", flush=True)
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
        
    # 2. Text Models: Sparse TF-IDF & Neural Subword (SVD + MLP)
    print("Extracting text features on fit...", flush=True)
    vec_combo = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000,
                                sublinear_tf=True, strip_accents="unicode", lowercase=True)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                               max_features=120000, sublinear_tf=True)
    
    X_w_fit = vec_combo.fit_transform([combos[i] for i in fit])
    X_c_fit = vec_char.fit_transform([titles[i] for i in fit])
    X_sp_fit = sp.hstack([X_w_fit, X_c_fit], format="csr")
    
    X_w_tune = vec_combo.transform([combos[i] for i in tune])
    X_c_tune = vec_char.transform([titles[i] for i in tune])
    X_sp_tune = sp.hstack([X_w_tune, X_c_tune], format="csr")
    
    X_w_conf = vec_combo.transform([combos[i] for i in confirm])
    X_c_conf = vec_char.transform([titles[i] for i in confirm])
    X_sp_conf = sp.hstack([X_w_conf, X_c_conf], format="csr")
    
    # Subword representations for SVD + MLP
    print("Fitting TruncatedSVD subword embeddings on fit...", flush=True)
    vec_subword = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=5,
                                  max_features=50000, sublinear_tf=True)
    X_sub_fit = vec_subword.fit_transform([titles[i] for i in fit])
    X_sub_tune = vec_subword.transform([titles[i] for i in tune])
    X_sub_conf = vec_subword.transform([titles[i] for i in confirm])
    
    svd = TruncatedSVD(n_components=64, random_state=seed)
    X_emb_fit = svd.fit_transform(X_sub_fit)
    X_emb_tune = svd.transform(X_sub_tune)
    X_emb_conf = svd.transform(X_sub_conf)
    
    # OOF generation inside fit
    print("Computing clean GroupKFold OOF for Sparse LR and Neural MLP inside fit...", flush=True)
    oof_sp = np.zeros(len(fit), dtype=np.float32)
    oof_mlp = np.zeros(len(fit), dtype=np.float32)
    
    t0_mlp = time.time()
    for tr, va in gkf.split(fit, y_fit, sellers[fit]):
        # Sparse LR
        lr_sp = LogisticRegression(C=3.0, max_iter=250, solver="liblinear", random_state=seed)
        lr_sp.fit(X_sp_fit[tr], y_fit[tr])
        oof_sp[va] = lr_sp.predict_proba(X_sp_fit[va])[:, 1]
        
        # Neural Subword MLP
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                            alpha=1e-3, max_iter=25, random_state=seed, early_stopping=False)
        mlp.fit(X_emb_fit[tr], y_fit[tr])
        oof_mlp[va] = mlp.predict_proba(X_emb_fit[va])[:, 1]
        
    print(f"OOF generation completed in {time.time()-t0_mlp:.2f}s", flush=True)
    print(f"  Fit OOF Sparse Acc: {accuracy_score(y_fit, oof_sp > 0.5):.4f}, LL: {log_loss(y_fit, oof_sp):.4f}")
    print(f"  Fit OOF Neural Acc: {accuracy_score(y_fit, oof_mlp > 0.5):.4f}, LL: {log_loss(y_fit, oof_mlp):.4f}")
    
    # Refit full text models on entire fit
    lr_sp_full = LogisticRegression(C=3.0, max_iter=300, solver="liblinear", random_state=seed).fit(X_sp_fit, y_fit)
    p_sp_tune = lr_sp_full.predict_proba(X_sp_tune)[:, 1]
    p_sp_conf = lr_sp_full.predict_proba(X_sp_conf)[:, 1]
    
    mlp_full = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                             alpha=1e-3, max_iter=30, random_state=seed).fit(X_emb_fit, y_fit)
    p_mlp_tune = mlp_full.predict_proba(X_emb_tune)[:, 1]
    p_mlp_conf = mlp_full.predict_proba(X_emb_conf)[:, 1]
    
    # 3. Assemble Feature DataFrames for CatBoost
    CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]
    
    df_base_fit = F_df.iloc[fit].copy().reset_index(drop=True)
    df_base_tune = F_df.iloc[tune].copy().reset_index(drop=True)
    df_base_conf = F_df.iloc[confirm].copy().reset_index(drop=True)
    
    for c in te_cols:
        df_base_fit[f"te_{c}"] = fit_te[c]
        df_base_tune[f"te_{c}"] = tune_te[c]
        df_base_conf[f"te_{c}"] = confirm_te[c]
        
    df_base_fit["txt_sparse_p"] = oof_sp
    df_base_tune["txt_sparse_p"] = p_sp_tune
    df_base_conf["txt_sparse_p"] = p_sp_conf
    
    for c in CAT_FEATURES:
        df_base_fit[c] = df_base_fit[c].astype(str)
        df_base_tune[c] = df_base_tune[c].astype(str)
        df_base_conf[c] = df_base_conf[c].astype(str)
        
    # Neural DataFrame: Base + txt_mlp_p
    df_neural_fit = df_base_fit.copy()
    df_neural_tune = df_base_tune.copy()
    df_neural_conf = df_base_conf.copy()
    
    df_neural_fit["txt_mlp_p"] = oof_mlp
    df_neural_tune["txt_mlp_p"] = p_mlp_tune
    df_neural_conf["txt_mlp_p"] = p_mlp_conf
    
    # 4. Train Deep CatBoost Baseline
    print("Training Strong Deep CatBoost Baseline (iterations=900, depth=6)...", flush=True)
    t0 = time.time()
    cb_base = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                 random_seed=seed, thread_count=12, verbose=0)
    cb_base.fit(Pool(df_base_fit, y_fit, cat_features=CAT_FEATURES))
    print(f"Deep CatBoost Baseline fitted in {time.time()-t0:.2f}s", flush=True)
    
    p_base_tune = cb_base.predict_proba(Pool(df_base_tune, cat_features=CAT_FEATURES))[:, 1]
    p_base_conf = cb_base.predict_proba(Pool(df_base_conf, cat_features=CAT_FEATURES))[:, 1]
    
    # 5. Train Neural-Augmented Deep CatBoost
    print("Training Neural-Augmented Deep CatBoost (Base + txt_mlp_p)...", flush=True)
    t0 = time.time()
    cb_neural = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                   random_seed=seed, thread_count=12, verbose=0)
    cb_neural.fit(Pool(df_neural_fit, y_fit, cat_features=CAT_FEATURES))
    print(f"Neural Deep CatBoost fitted in {time.time()-t0:.2f}s", flush=True)
    
    p_neural_tune = cb_neural.predict_proba(Pool(df_neural_tune, cat_features=CAT_FEATURES))[:, 1]
    p_neural_conf = cb_neural.predict_proba(Pool(df_neural_conf, cat_features=CAT_FEATURES))[:, 1]
    
    # 6. Freeze Decision Threshold on TUNE before touching CONFIRM
    y_tu = y_used[tune]
    best_th_base, best_acc_base_tu = 0.50, 0.0
    best_th_neural, best_acc_neural_tu = 0.50, 0.0
    
    for th in np.linspace(0.40, 0.60, 41):
        acc_b = accuracy_score(y_tu, p_base_tune >= th)
        if acc_b > best_acc_base_tu:
            best_acc_base_tu = acc_b
            best_th_base = th
            
        acc_n = accuracy_score(y_tu, p_neural_tune >= th)
        if acc_n > best_acc_neural_tu:
            best_acc_neural_tu = acc_n
            best_th_neural = th
            
    print(f"Frozen Decision Thresholds on TUNE:")
    print(f"  Base Deep CatBoost: th={best_th_base:.3f} (Tune Acc: {best_acc_base_tu:.5f})")
    print(f"  Neural Deep CatBoost: th={best_th_neural:.3f} (Tune Acc: {best_acc_neural_tu:.5f})")
    
    # 7. Evaluate on Untouched CONFIRM Holdout
    y_co = y_used[confirm]
    pred_base_co = (p_base_conf >= best_th_base).astype(int)
    pred_neural_co = (p_neural_conf >= best_th_neural).astype(int)
    
    acc_base = accuracy_score(y_co, pred_base_co)
    acc_neural = accuracy_score(y_co, pred_neural_co)
    ll_base = log_loss(y_co, p_base_conf)
    ll_neural = log_loss(y_co, p_neural_conf)
    
    fixes = int(np.sum((pred_base_co != y_co) & (pred_neural_co == y_co)))
    breaks = int(np.sum((pred_base_co == y_co) & (pred_neural_co != y_co)))
    net = fixes - breaks
    
    ci_low, ci_high, _ = cluster_bootstrap_ci(y_co, pred_base_co, pred_neural_co, sellers[confirm])
    
    fit_sellers = set(sellers[fit])
    is_seen_co = np.array([s in fit_sellers for s in sellers[confirm]], dtype=bool)
    is_new_co = ~is_seen_co
    
    fixes_seen = int(np.sum((pred_base_co[is_seen_co] != y_co[is_seen_co]) & (pred_neural_co[is_seen_co] == y_co[is_seen_co])))
    breaks_seen = int(np.sum((pred_base_co[is_seen_co] == y_co[is_seen_co]) & (pred_neural_co[is_seen_co] != y_co[is_seen_co])))
    net_seen = fixes_seen - breaks_seen
    
    fixes_new = int(np.sum((pred_base_co[is_new_co] != y_co[is_new_co]) & (pred_neural_co[is_new_co] == y_co[is_new_co])))
    breaks_new = int(np.sum((pred_base_co[is_new_co] == y_co[is_new_co]) & (pred_neural_co[is_new_co] != y_co[is_new_co])))
    net_new = fixes_new - breaks_new
    
    print(f"\nCONFIRM RESULTS (Seed {seed}):")
    print(f"  Base Deep CatBoost Acc:   {acc_base:.5f}, LL: {ll_base:.5f}")
    print(f"  Neural Deep CatBoost Acc: {acc_neural:.5f}, LL: {ll_neural:.5f}")
    print(f"  Fixes / Breaks:           {fixes} / {breaks} -> NET: {net:+d}")
    print(f"  Bootstrap 95% CI:         [{ci_low:+.1f}, {ci_high:+.1f}]")
    print(f"  Seen sellers net:         {net_seen:+d} (Fixes: {fixes_seen}, Breaks: {breaks_seen}, n={is_seen_co.sum()})")
    print(f"  New sellers net:          {net_new:+d} (Fixes: {fixes_new}, Breaks: {breaks_new}, n={is_new_co.sum()})")
    
    # Save paired predictions
    pairs_file = RESULTS_DIR / f"paired_neural_ablation_seed_{seed}.csv"
    with open(pairs_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "seller_id", "y_true", "is_seen", "p_base", "p_neural", "pred_base", "pred_neural"])
        for i in range(len(confirm)):
            w.writerow([
                ids[confirm[i]], sellers[confirm[i]], y_co[i], int(is_seen_co[i]),
                round(float(p_base_conf[i]), 5), round(float(p_neural_conf[i]), 5),
                pred_base_co[i], pred_neural_co[i]
            ])
            
    return {
        "seed": seed,
        "n_confirm": len(confirm),
        "th_base": best_th_base,
        "th_neural": best_th_neural,
        "acc_base": float(acc_base),
        "acc_neural": float(acc_neural),
        "ll_base": float(ll_base),
        "ll_neural": float(ll_neural),
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
        res = run_neural_ablation(s, F_df, cat_df, titles, warranties, combos, y_used, sellers, ids)
        all_results.append(res)
        
    out_json = RESULTS_DIR / "round153_neural_subword_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll neural ablation results saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
