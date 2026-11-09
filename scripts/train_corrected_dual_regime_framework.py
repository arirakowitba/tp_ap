"""Corrected Dual-Regime Validation and Modeling Framework (AP-2026-Q-2).

Author: Antigravity
Date: 2026-11-09
Policy: Full compliance with CX-0138 and Ariel directives.

Key Methodological Guarantees:
1. Corrected Fold-Local Feature Engineering:
   - All transformations (TF-IDF word, TF-IDF char, TruncatedSVD, category TE, seller TE, category median price)
     are fitted strictly inside each inner training fold. Zero data leakage into validation or meta-features.
2. Deterministic Categorical Mapping:
   - Categorical maps learned strictly on fit, unseen categories mapped to -1.
3. Dual-Regime Frozen Holdout Partition:
   - Exactly 14,000 items (20% of 70k) reflecting the true test mixture:
     * 71.62% Seen Holdout (10,027 items from repeat sellers in fit).
     * 28.38% Unseen Holdout (3,973 items from completely disjoint sellers).
   - Zero overlap verified between fit pool and unseen holdout.
4. Complementary Diverse Architectures:
   - LightGBM (depth unconstrained, structured + ratios + text SVD).
   - CatBoost Symmetric (depth 6).
   - XGBoost Hist (depth 6).
   - Sparse Ridge Text Classifier (direct high-dimensional text without SVD compression).
5. Regime-Conditioned Dual-Head Meta-Stacker:
   - Specialized cross-fitted meta-stackers for Seen regime and Unseen regime.
6. High-Confidence Margin Curation:
   - Evaluates variance reduction via margin thresholding to maximize P5.
7. Clustered Bootstrap & Stress-Test Calibration.
"""

import os
import sys
import gc
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, KFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

ROOT = Path(".")
DATA_DIR = ROOT / "data/ap_2026_q_2/raw"
OUT_DIR = ROOT / "results/dual_regime_20261109"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_raw_data():
    print("[1/7] Loading raw train and test data...", flush=True)
    df_train = pd.read_json(DATA_DIR / "train_data.jsonlines", lines=True)
    df_test = pd.read_json(DATA_DIR / "test_data.jsonlines", lines=True)
    
    y = (df_train["condition"] == "used").astype(np.int8).values
    df_train["target"] = y
    
    print(f"  Train: {len(df_train)} items ({len(df_train['seller_id'].unique())} sellers)")
    print(f"  Test:  {len(df_test)} items ({len(df_test['seller_id'].unique())} sellers)")
    return df_train, df_test


def create_dual_regime_frozen_split(df_train, seed=20261109):
    print("\n[2/7] Constructing Dual-Regime Frozen Holdout Partition (71.62% Seen / 28.38% Unseen)...", flush=True)
    split_cache_p = OUT_DIR / "frozen_dual_regime_indices.json"
    
    if split_cache_p.exists():
        with open(split_cache_p, "r") as f:
            idx_data = json.load(f)
        fit_idx = np.array(idx_data["fit_idx"])
        seen_holdout_idx = np.array(idx_data["seen_holdout_idx"])
        unseen_holdout_idx = np.array(idx_data["unseen_holdout_idx"])
        print("  Loaded cached dual-regime split.")
    else:
        rng = np.random.RandomState(seed)
        sellers = np.array(df_train["seller_id"].unique(), copy=True)
        rng.shuffle(sellers)
        
        # 1. Unseen sellers: disjoint from fit pool, target ~3973 items
        unseen_sellers = set()
        unseen_items_count = 0
        for s in sellers:
            cnt = (df_train["seller_id"] == s).sum()
            if unseen_items_count + cnt <= 4000:
                unseen_sellers.add(s)
                unseen_items_count += cnt
            if unseen_items_count >= 3973:
                break
                
        # 2. Seen holdout: items from sellers with >1 item in remaining pool
        pool_seen = df_train[~df_train["seller_id"].isin(unseen_sellers)]
        multi_sellers = pool_seen["seller_id"].value_counts()
        multi_sellers = np.array(multi_sellers[multi_sellers > 1].index.values, copy=True)
        rng.shuffle(multi_sellers)
        
        seen_holdout_idx = []
        for s in multi_sellers:
            items = pool_seen[pool_seen["seller_id"] == s].index.values
            n_take = max(1, int(len(items) * 0.25))
            sampled = rng.choice(items, size=min(n_take, len(items) - 1), replace=False)
            seen_holdout_idx.extend(sampled)
            if len(seen_holdout_idx) >= 10027:
                break
                
        seen_holdout_idx = np.array(seen_holdout_idx[:10027])
        unseen_holdout_idx = df_train[df_train["seller_id"].isin(unseen_sellers)].index.values[:3973]
        all_holdout = np.concatenate([seen_holdout_idx, unseen_holdout_idx])
        fit_idx = np.setdiff1d(df_train.index.values, all_holdout)
        
        # Cache indices
        with open(split_cache_p, "w") as f:
            json.dump({
                "fit_idx": fit_idx.tolist(),
                "seen_holdout_idx": seen_holdout_idx.tolist(),
                "unseen_holdout_idx": unseen_holdout_idx.tolist()
            }, f)
        print("  Saved dual-regime split to disk.")

    # Integrity verification
    fit_sellers = set(df_train.iloc[fit_idx]["seller_id"])
    unseen_sellers = set(df_train.iloc[unseen_holdout_idx]["seller_id"])
    seen_sellers = set(df_train.iloc[seen_holdout_idx]["seller_id"])
    
    assert len(fit_sellers.intersection(unseen_sellers)) == 0, "ERROR: Unseen overlap!"
    assert seen_sellers.issubset(fit_sellers), "ERROR: Seen seller not in fit!"
    
    n_tot_holdout = len(seen_holdout_idx) + len(unseen_holdout_idx)
    pct_seen = len(seen_holdout_idx) / n_tot_holdout
    pct_unseen = len(unseen_holdout_idx) / n_tot_holdout
    
    print(f"  Fit pool size: {len(fit_idx)} items")
    print(f"  Holdout size:  {n_tot_holdout} items")
    print(f"    - Seen Holdout:   {len(seen_holdout_idx)} items ({pct_seen*100:.2f}%)")
    print(f"    - Unseen Holdout: {len(unseen_holdout_idx)} items ({pct_unseen*100:.2f}%)")
    print("  [INTEGRITY AUDIT]: PASS (0 unseen seller leakage, 100% seen seller presence in fit).")
    
    return fit_idx, seen_holdout_idx, unseen_holdout_idx


class FoldFeatureExtractor:
    """Strictly fold-local feature extractor to prevent any data leakage."""
    def __init__(self):
        self.cat_map = {}
        self.ship_map = {}
        self.buy_map = {}
        self.list_map = {}
        self.cat_median_price = {}
        self.global_median_price = 0.0
        self.cat_te = {}
        self.seller_te = {}
        self.global_prior = 0.5
        self.vec_word = None
        self.vec_char = None
        self.svd_word = None
        self.svd_char = None

    def fit(self, df_tr, y_tr):
        self.global_prior = float(y_tr.mean())
        
        # Categorical mappings
        for col, attr in [("category_id", "cat_map"), ("buying_mode", "buy_map"), ("listing_type_id", "list_map")]:
            vals = df_tr[col].dropna().unique()
            setattr(self, attr, {v: i for i, v in enumerate(vals)})
            
        ship_modes = df_tr["shipping"].apply(lambda s: s.get("mode") if isinstance(s, dict) else "unknown").unique()
        self.ship_map = {v: i for i, v in enumerate(ship_modes)}
        
        # Category median price
        prices = df_tr["price"].fillna(0.0).astype(np.float32)
        self.global_median_price = float(prices.median())
        self.cat_median_price = df_tr.groupby("category_id")["price"].median().to_dict()
        
        # Target Encodings with m-estimates
        # Category TE (m=15)
        cat_counts = df_tr.groupby("category_id").size()
        cat_means = df_tr.groupby("category_id")["target"].mean()
        self.cat_te = ((cat_counts * cat_means + 15.0 * self.global_prior) / (cat_counts + 15.0)).to_dict()
        
        # Seller TE (m=25)
        seller_counts = df_tr.groupby("seller_id").size()
        seller_means = df_tr.groupby("seller_id")["target"].mean()
        self.seller_te = ((seller_counts * seller_means + 25.0 * self.global_prior) / (seller_counts + 25.0)).to_dict()
        
        # Text TF-IDF + SVD
        titles = df_tr["title"].fillna("").astype(str).str.lower()
        self.vec_word = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
        self.svd_word = TruncatedSVD(n_components=32, random_state=42)
        w_mat = self.vec_word.fit_transform(titles)
        self.svd_word.fit(w_mat)
        
        self.vec_char = TfidfVectorizer(max_features=20000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.svd_char = TruncatedSVD(n_components=16, random_state=42)
        c_mat = self.vec_char.fit_transform(titles)
        self.svd_char.fit(c_mat)
        return self

    def transform_dense(self, df):
        feats = pd.DataFrame(index=df.index)
        
        # Numerical & generalizable ratios
        price = df["price"].fillna(0.0).astype(np.float32)
        feats["price"] = price
        feats["log_price"] = np.log1p(np.maximum(price, 0.0))
        
        cat_med = df["category_id"].map(self.cat_median_price).fillna(self.global_median_price).astype(np.float32)
        feats["log_price_ratio_cat"] = np.log((price + 1.0) / (cat_med + 1.0))
        
        feats["initial_quantity"] = df["initial_quantity"].fillna(1).astype(np.float32)
        feats["sold_quantity"] = df["sold_quantity"].fillna(0).astype(np.float32)
        feats["sold_ratio"] = feats["sold_quantity"] / (feats["initial_quantity"] + 1e-4)
        feats["available_quantity"] = df["available_quantity"].fillna(0).astype(np.float32)
        
        # Warranties & text metadata
        warranty_str = df["warranty"].fillna("").astype(str).str.lower()
        feats["has_warranty"] = df["warranty"].notna().astype(np.int8)
        feats["warranty_len"] = warranty_str.str.len().astype(np.int16)
        feats["warranty_is_factory"] = warranty_str.str.contains(r"fabrica|fabricante", regex=True).astype(np.int8)
        feats["warranty_is_seller"] = warranty_str.str.contains(r"vendedor|local", regex=True).astype(np.int8)
        feats["warranty_is_official"] = warranty_str.str.contains(r"oficial", regex=True).astype(np.int8)
        feats["warranty_is_none"] = warranty_str.str.contains(r"sin garant|no.*garant", regex=True).astype(np.int8)
        
        titles = df["title"].fillna("").astype(str).str.lower()
        feats["title_len"] = titles.str.len().astype(np.int16)
        feats["title_words"] = titles.str.split().apply(len).astype(np.int8)
        feats["num_pictures"] = df["pictures"].apply(lambda p: len(p) if isinstance(p, list) else 0).astype(np.int8)
        feats["accepts_mercadopago"] = df["accepts_mercadopago"].astype(np.int8)
        feats["automatic_relist"] = df["automatic_relist"].astype(np.int8)
        
        # Learned categorical codes
        feats["cat_code"] = df["category_id"].map(self.cat_map).fillna(-1).astype(np.int32)
        feats["buy_code"] = df["buying_mode"].map(self.buy_map).fillna(-1).astype(np.int32)
        feats["list_code"] = df["listing_type_id"].map(self.list_map).fillna(-1).astype(np.int32)
        
        ship_mode_series = df["shipping"].apply(lambda s: s.get("mode") if isinstance(s, dict) else "unknown")
        feats["ship_code"] = ship_mode_series.map(self.ship_map).fillna(-1).astype(np.int32)
        
        # Target Encodings
        feats["cat_te"] = df["category_id"].map(self.cat_te).fillna(self.global_prior).astype(np.float32)
        feats["seller_te"] = df["seller_id"].map(self.seller_te).fillna(self.global_prior).astype(np.float32)
        
        # Text SVD
        w_mat = self.vec_word.transform(titles)
        w_svd = self.svd_word.transform(w_mat)
        c_mat = self.vec_char.transform(titles)
        c_svd = self.svd_char.transform(c_mat)
        
        dense_features = np.hstack([feats.values, w_svd, c_svd])
        return dense_features

    def transform_sparse_text(self, df):
        titles = df["title"].fillna("").astype(str).str.lower()
        w_mat = self.vec_word.transform(titles)
        c_mat = self.vec_char.transform(titles)
        return sparse.hstack([w_mat, c_mat], format="csr")


def run_honest_cross_fitting(df_fit):
    print("\n[3/7] Executing Honest 5-Fold Cross-Fitting on Fit Pool (56k items)...", flush=True)
    y_fit = df_fit["target"].values
    sellers_fit = df_fit["seller_id"].values
    
    # StratifiedGroupKFold on sellers in fit pool
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(df_fit), dtype=np.float32)
    oof_xgb = np.zeros(len(df_fit), dtype=np.float32)
    oof_cat = np.zeros(len(df_fit), dtype=np.float32)
    oof_rdg = np.zeros(len(df_fit), dtype=np.float32)
    
    t_start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df_fit, y_fit, sellers_fit)):
        f_t0 = time.time()
        print(f"  Fold {fold+1}/5...", end="", flush=True)
        
        df_tr = df_fit.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_fit[tr_idx]
        df_va = df_fit.iloc[va_idx].reset_index(drop=True)
        y_va = y_fit[va_idx]
        
        # 1. Fit fold-local feature extractor
        fe = FoldFeatureExtractor()
        fe.fit(df_tr, y_tr)
        
        X_tr_dense = fe.transform_dense(df_tr)
        X_va_dense = fe.transform_dense(df_va)
        
        X_tr_sparse = fe.transform_sparse_text(df_tr)
        X_va_sparse = fe.transform_sparse_text(df_va)
        
        # 2. Train LightGBM
        m_lgb = lgb.LGBMClassifier(n_estimators=380, learning_rate=0.06, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        m_lgb.fit(X_tr_dense, y_tr)
        oof_lgb[va_idx] = m_lgb.predict_proba(X_va_dense)[:, 1]
        
        # 3. Train XGBoost
        m_xgb = xgb.XGBClassifier(n_estimators=280, learning_rate=0.07, max_depth=6, random_state=42, eval_metric="logloss", n_jobs=-1)
        m_xgb.fit(X_tr_dense, y_tr)
        oof_xgb[va_idx] = m_xgb.predict_proba(X_va_dense)[:, 1]
        
        # 4. Train CatBoost
        m_cat = CatBoostClassifier(iterations=320, learning_rate=0.07, depth=6, random_seed=42, verbose=0, thread_count=-1)
        m_cat.fit(X_tr_dense, y_tr)
        oof_cat[va_idx] = m_cat.predict_proba(X_va_dense)[:, 1]
        
        # 5. Train Sparse Ridge Text Classifier
        m_rdg = RidgeClassifier(alpha=2.0, random_state=42)
        m_rdg.fit(X_tr_sparse, y_tr)
        # Convert decision function to pseudo-probability via sigmoid
        df_val_scores = m_rdg.decision_function(X_va_sparse)
        oof_rdg[va_idx] = 1.0 / (1.0 + np.exp(-df_val_scores))
        
        fold_acc = accuracy_score(y_va, (oof_lgb[va_idx] >= 0.5))
        print(f" Done ({time.time()-f_t0:.1f}s) | LGBM Acc: {fold_acc:.4f}", flush=True)

    print(f"  Cross-fitting completed in {time.time()-t_start:.1f}s.")
    
    # Summary of honest OOF
    print("\n  --- HONEST OOF BASE MODEL METRICS ON FIT POOL (56k items) ---")
    for name, p in [("LightGBM", oof_lgb), ("XGBoost", oof_xgb), ("CatBoost", oof_cat), ("Sparse Ridge", oof_rdg)]:
        acc = accuracy_score(y_fit, (p >= 0.5))
        loss = log_loss(y_fit, p)
        auc = roc_auc_score(y_fit, p)
        print(f"  {name:15s} | Acc: {acc:.5f} | LogLoss: {loss:.5f} | ROC-AUC: {auc:.5f}")
        
    return oof_lgb, oof_xgb, oof_cat, oof_rdg


def fit_dual_regime_meta_models(df_fit, oof_lgb, oof_xgb, oof_cat, oof_rdg):
    print("\n[4/7] Fitting Regime-Conditioned Meta-Models (Seen vs Unseen)...", flush=True)
    y_fit = df_fit["target"].values
    
    # Mark which rows in fit pool come from multi-item sellers (Seen-like proxy) vs single-item sellers (Unseen-like proxy)
    seller_counts = df_fit["seller_id"].value_counts()
    is_multi_seller = df_fit["seller_id"].map(seller_counts > 1).values
    
    meta_features = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_rdg])
    
    # Meta model for seen regime (multi-item sellers)
    meta_seen = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    meta_seen.fit(meta_features[is_multi_seller], y_fit[is_multi_seller])
    print(f"  Meta-Seen Weights [LGBM, XGB, CatB, Ridge]: {meta_seen.coef_[0].round(3)} | Intercept: {meta_seen.intercept_[0]:.3f}")
    
    # Meta model for unseen regime (single-item sellers proxy)
    meta_unseen = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    meta_unseen.fit(meta_features[~is_multi_seller], y_fit[~is_multi_seller])
    print(f"  Meta-Unseen Weights [LGBM, XGB, CatB, Ridge]: {meta_unseen.coef_[0].round(3)} | Intercept: {meta_unseen.intercept_[0]:.3f}")
    
    # Compute combined meta OOF
    oof_meta = np.zeros(len(df_fit), dtype=np.float32)
    oof_meta[is_multi_seller] = meta_seen.predict_proba(meta_features[is_multi_seller])[:, 1]
    oof_meta[~is_multi_seller] = meta_unseen.predict_proba(meta_features[~is_multi_seller])[:, 1]
    
    acc_meta = accuracy_score(y_fit, (oof_meta >= 0.5))
    loss_meta = log_loss(y_fit, oof_meta)
    auc_meta = roc_auc_score(y_fit, oof_meta)
    print(f"\n  Regime-Conditioned Meta-Stacker OOF Acc: {acc_meta:.5f} | LogLoss: {loss_meta:.5f} | ROC-AUC: {auc_meta:.5f}")
    
    # Compare with LGBM baseline on fit pool
    pred_base = (oof_lgb >= 0.5).astype(np.int8)
    pred_meta = (oof_meta >= 0.5).astype(np.int8)
    fixes = int(np.sum((pred_base != y_fit) & (pred_meta == y_fit)))
    breaks = int(np.sum((pred_base == y_fit) & (pred_meta != y_fit)))
    net = fixes - breaks
    print(f"  Pairwise vs LGBM Baseline: Fixes={fixes} | Breaks={breaks} | Net Gain={net:+d} hits on 56k")
    
    return meta_seen, meta_unseen, oof_meta


def evaluate_high_confidence_margin_thresholds(y_fit, oof_base, oof_meta):
    print("\n[5/7] High-Confidence Margin Sweep on OOF to Maximize P5...", flush=True)
    # The margin is how far from 0.5 the meta prediction must be before overriding baseline
    margins = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25]
    
    results = []
    base_pred = (oof_base >= 0.5).astype(np.int8)
    
    print(f"  {'Margin':8s} | {'N Flips':8s} | {'Fixes':6s} | {'Breaks':6s} | {'Net 56k':8s} | {'Precision':10s} | {'Est P5':8s}")
    print("  " + "-" * 70)
    
    for m in margins:
        # Flip rule: keep baseline unless meta is confident (|p_meta - 0.5| >= m) and points in opposite direction
        candidate_pred = base_pred.copy()
        flip_mask = (np.abs(oof_meta - 0.5) >= m) & ((oof_meta >= 0.5).astype(np.int8) != base_pred)
        candidate_pred[flip_mask] = (oof_meta[flip_mask] >= 0.5).astype(np.int8)
        
        n_flips = int(np.sum(flip_mask))
        fixes = int(np.sum((base_pred != y_fit) & (candidate_pred == y_fit)))
        breaks = int(np.sum((base_pred == y_fit) & (candidate_pred != y_fit)))
        net = fixes - breaks
        prec = (fixes / max(fixes + breaks, 1)) * 100.0
        
        # Approximate public net gain scaled to N_pub=20998
        scale = 20998.0 / len(y_fit)
        mu_gain = net * scale
        sigma_gain = np.sqrt(max(n_flips * scale * 4.0 * (prec/100.0) * (1.0 - prec/100.0), 1.0))
        p5_est = mu_gain - 1.645 * sigma_gain
        
        print(f"  {m:8.2f} | {n_flips:8d} | {fixes:6d} | {breaks:6d} | {net:+8d} | {prec:9.1f}% | {p5_est:+8.1f}")
        results.append({
            "margin": m, "n_flips": n_flips, "fixes": fixes, "breaks": breaks,
            "net": net, "precision": prec, "p5_est": p5_est
        })
        
    # Choose optimal margin that maximizes p5_est
    best_opt = max(results, key=lambda x: x["p5_est"])
    best_margin = best_opt["margin"]
    print(f"\n  Optimal Margin Selected strictly on OOF: delta = {best_margin:.2f} (Est P5: {best_opt['p5_est']:+.1f})")
    return best_margin, results


def train_full_fit_and_evaluate_frozen_holdout(df_fit, df_train, seen_holdout_idx, unseen_holdout_idx, meta_seen, meta_unseen, best_margin, df_test):
    print("\n[6/7] Final Model Training on Fit Pool & Single Frozen Holdout Evaluation...", flush=True)
    y_fit = df_fit["target"].values
    
    # 1. Fit complete feature extractor on all 56k fit pool
    print("  Fitting complete feature extractor on 56k fit pool...")
    fe_full = FoldFeatureExtractor()
    fe_full.fit(df_fit, y_fit)
    
    X_fit_dense = fe_full.transform_dense(df_fit)
    X_fit_sparse = fe_full.transform_sparse_text(df_fit)
    
    # Train full base models on 56k fit pool
    print("  Fitting base models on 56k fit pool...")
    m_lgb = lgb.LGBMClassifier(n_estimators=420, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    m_lgb.fit(X_fit_dense, y_fit)
    
    m_xgb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.06, max_depth=6, random_state=42, eval_metric="logloss", n_jobs=-1)
    m_xgb.fit(X_fit_dense, y_fit)
    
    m_cat = CatBoostClassifier(iterations=350, learning_rate=0.06, depth=6, random_seed=42, verbose=0, thread_count=-1)
    m_cat.fit(X_fit_dense, y_fit)
    
    m_rdg = RidgeClassifier(alpha=2.0, random_state=42)
    m_rdg.fit(X_fit_sparse, y_fit)
    
    # 2. Transform the 14k Frozen Dual-Regime Holdout
    print("  Transforming 14k Frozen Dual-Regime Holdout...")
    df_seen_ho = df_train.iloc[seen_holdout_idx].reset_index(drop=True)
    y_seen_ho = df_seen_ho["target"].values
    
    df_unseen_ho = df_train.iloc[unseen_holdout_idx].reset_index(drop=True)
    y_unseen_ho = df_unseen_ho["target"].values
    
    # Dense & sparse transforms
    X_seen_dense = fe_full.transform_dense(df_seen_ho)
    X_seen_sparse = fe_full.transform_sparse_text(df_seen_ho)
    
    X_unseen_dense = fe_full.transform_dense(df_unseen_ho)
    X_unseen_sparse = fe_full.transform_sparse_text(df_unseen_ho)
    
    # Predict base models on seen holdout
    p_seen_lgb = m_lgb.predict_proba(X_seen_dense)[:, 1]
    p_seen_xgb = m_xgb.predict_proba(X_seen_dense)[:, 1]
    p_seen_cat = m_cat.predict_proba(X_seen_dense)[:, 1]
    p_seen_rdg = 1.0 / (1.0 + np.exp(-m_rdg.decision_function(X_seen_sparse)))
    
    meta_in_seen = np.column_stack([p_seen_lgb, p_seen_xgb, p_seen_cat, p_seen_rdg])
    p_seen_meta = meta_seen.predict_proba(meta_in_seen)[:, 1]
    
    # Predict base models on unseen holdout
    p_unseen_lgb = m_lgb.predict_proba(X_unseen_dense)[:, 1]
    p_unseen_xgb = m_xgb.predict_proba(X_unseen_dense)[:, 1]
    p_unseen_cat = m_cat.predict_proba(X_unseen_dense)[:, 1]
    p_unseen_rdg = 1.0 / (1.0 + np.exp(-m_rdg.decision_function(X_unseen_sparse)))
    
    meta_in_unseen = np.column_stack([p_unseen_lgb, p_unseen_xgb, p_unseen_cat, p_unseen_rdg])
    p_unseen_meta = meta_unseen.predict_proba(meta_in_unseen)[:, 1]
    
    # Apply optimal margin shrinkage
    pred_seen_base = (p_seen_lgb >= 0.5).astype(np.int8)
    pred_seen_cand = pred_seen_base.copy()
    flip_seen = (np.abs(p_seen_meta - 0.5) >= best_margin) & ((p_seen_meta >= 0.5).astype(np.int8) != pred_seen_base)
    pred_seen_cand[flip_seen] = (p_seen_meta[flip_seen] >= 0.5).astype(np.int8)
    
    pred_unseen_base = (p_unseen_lgb >= 0.5).astype(np.int8)
    pred_unseen_cand = pred_unseen_base.copy()
    flip_unseen = (np.abs(p_unseen_meta - 0.5) >= best_margin) & ((p_unseen_meta >= 0.5).astype(np.int8) != pred_unseen_base)
    pred_unseen_cand[flip_unseen] = (p_unseen_meta[flip_unseen] >= 0.5).astype(np.int8)
    
    # Metrics Seen Holdout (10,027 items)
    acc_seen_base = accuracy_score(y_seen_ho, pred_seen_base)
    acc_seen_cand = accuracy_score(y_seen_ho, pred_seen_cand)
    fixes_seen = int(np.sum((pred_seen_base != y_seen_ho) & (pred_seen_cand == y_seen_ho)))
    breaks_seen = int(np.sum((pred_seen_base == y_seen_ho) & (pred_seen_cand != y_seen_ho)))
    net_seen = fixes_seen - breaks_seen
    
    # Metrics Unseen Holdout (3,973 items)
    acc_unseen_base = accuracy_score(y_unseen_ho, pred_unseen_base)
    acc_unseen_cand = accuracy_score(y_unseen_ho, pred_unseen_cand)
    fixes_unseen = int(np.sum((pred_unseen_base != y_unseen_ho) & (pred_unseen_cand == y_unseen_ho)))
    breaks_unseen = int(np.sum((pred_unseen_base == y_unseen_ho) & (pred_unseen_cand != y_unseen_ho)))
    net_unseen = fixes_unseen - breaks_unseen
    
    # Composite Holdout Metrics (Exact 71.62% seen / 28.38% unseen)
    tot_fixes = fixes_seen + fixes_unseen
    tot_breaks = breaks_seen + breaks_unseen
    tot_net = net_seen + net_unseen
    acc_comp_base = (acc_seen_base * 0.7162) + (acc_unseen_base * 0.2838)
    acc_comp_cand = (acc_seen_cand * 0.7162) + (acc_unseen_cand * 0.2838)
    
    print("\n  --- FROZEN DUAL-REGIME HOLDOUT EVALUATION (14,000 items) ---")
    print(f"  SEEN REGIME   (10,027 items): Base Acc: {acc_seen_base:.5f} -> Cand Acc: {acc_seen_cand:.5f} | Fixes: {fixes_seen}, Breaks: {breaks_seen}, Net: {net_seen:+d}")
    print(f"  UNSEEN REGIME  (3,973 items): Base Acc: {acc_unseen_base:.5f} -> Cand Acc: {acc_unseen_cand:.5f} | Fixes: {fixes_unseen}, Breaks: {breaks_unseen}, Net: {net_unseen:+d}")
    print(f"  COMPOSITE MIX (71.62/28.38): Base Acc: {acc_comp_base:.5f} -> Cand Acc: {acc_comp_cand:.5f} | Total Net: {tot_net:+d} hits")
    
    # 3. Predict on Test Set (30,000 items)
    print("\n  Generating Candidate Test Submission File...")
    X_test_dense = fe_full.transform_dense(df_test)
    X_test_sparse = fe_full.transform_sparse_text(df_test)
    
    p_test_lgb = m_lgb.predict_proba(X_test_dense)[:, 1]
    p_test_xgb = m_xgb.predict_proba(X_test_dense)[:, 1]
    p_test_cat = m_cat.predict_proba(X_test_dense)[:, 1]
    p_test_rdg = 1.0 / (1.0 + np.exp(-m_rdg.decision_function(X_test_sparse)))
    
    meta_in_test = np.column_stack([p_test_lgb, p_test_xgb, p_test_cat, p_test_rdg])
    
    # Route test predictions: is_seen in fit pool
    test_sellers = df_test["seller_id"].values
    is_seen_in_fit = np.isin(test_sellers, list(fe_full.seller_te.keys()))
    
    p_test_meta = np.zeros(len(df_test), dtype=np.float32)
    p_test_meta[is_seen_in_fit] = meta_seen.predict_proba(meta_in_test[is_seen_in_fit])[:, 1]
    p_test_meta[~is_seen_in_fit] = meta_unseen.predict_proba(meta_in_test[~is_seen_in_fit])[:, 1]
    
    test_pred_base = (p_test_lgb >= 0.5).astype(np.int8)
    test_pred_cand = test_pred_base.copy()
    flip_test = (np.abs(p_test_meta - 0.5) >= best_margin) & ((p_test_meta >= 0.5).astype(np.int8) != test_pred_base)
    test_pred_cand[flip_test] = (p_test_meta[flip_test] >= 0.5).astype(np.int8)
    
    test_condition = np.where(test_pred_cand == 1, "used", "new")
    sub_df = pd.DataFrame({"ID": np.arange(1, len(df_test) + 1), "condition": test_condition})
    
    cand_csv_p = OUT_DIR / "candidate_c6_dual_regime_curated_margin.csv"
    sub_df.to_csv(cand_csv_p, index=False)
    cand_hash = sha256_file(cand_csv_p)
    print(f"  Test Submission saved: {cand_csv_p}")
    print(f"  SHA-256: {cand_hash}")
    print(f"  Test Flips from Base: {int(np.sum(flip_test))} items ({np.mean(flip_test)*100:.2f}%)")
    print(f"  Class Distribution: Used: {(test_condition=='used').mean()*100:.2f}% | New: {(test_condition=='new').mean()*100:.2f}%")
    
    return {
        "acc_seen_base": acc_seen_base, "acc_seen_cand": acc_seen_cand, "net_seen": net_seen,
        "acc_unseen_base": acc_unseen_base, "acc_unseen_cand": acc_unseen_cand, "net_unseen": net_unseen,
        "acc_comp_base": acc_comp_base, "acc_comp_cand": acc_comp_cand, "tot_net": tot_net,
        "tot_fixes": tot_fixes, "tot_breaks": tot_breaks,
        "test_csv_path": str(cand_csv_p), "test_csv_hash": cand_hash,
        "test_n_flips": int(np.sum(flip_test)),
        "cand_pred_holdout": np.concatenate([pred_seen_cand, pred_unseen_cand]),
        "base_pred_holdout": np.concatenate([pred_seen_base, pred_unseen_base]),
        "y_holdout": np.concatenate([y_seen_ho, y_unseen_ho]),
        "sellers_holdout": np.concatenate([df_seen_ho["seller_id"].values, df_unseen_ho["seller_id"].values])
    }


def compute_clustered_uncertainty(holdout_data):
    print("\n[7/7] Computing Clustered Bootstrap & Prequential Stress-Test Quantiles...", flush=True)
    y_true = holdout_data["y_holdout"]
    p_cand = holdout_data["cand_pred_holdout"]
    p_base = holdout_data["base_pred_holdout"]
    sellers = holdout_data["sellers_holdout"]
    
    # Net gain indicator per item
    # Fix = +1, Break = -1, Unchanged = 0
    item_gains = ((p_base != y_true) & (p_cand == y_true)).astype(np.int8) - ((p_base == y_true) & (p_cand != y_true)).astype(np.int8)
    
    # Cluster bootstrap by seller (B=1000)
    unique_sellers = np.unique(sellers)
    seller_to_indices = {s: np.where(sellers == s)[0] for s in unique_sellers}
    
    N_pub = 20998
    scale = N_pub / len(y_true)
    
    rng = np.random.RandomState(42)
    B = 1000
    boot_gains = np.zeros(B, dtype=np.float32)
    
    for b in range(B):
        sample_sellers = rng.choice(unique_sellers, size=len(unique_sellers), replace=True)
        sampled_indices = np.concatenate([seller_to_indices[s] for s in sample_sellers])
        boot_net = np.sum(item_gains[sampled_indices])
        boot_gains[b] = boot_net * (N_pub / len(sampled_indices))
        
    p5_raw = float(np.percentile(boot_gains, 5))
    p50_raw = float(np.percentile(boot_gains, 50))
    p95_raw = float(np.percentile(boot_gains, 95))
    prob_raw = float(np.mean(boot_gains >= 53.0))
    
    # Stress-Test Widening with gamma=2.956
    gamma = 2.9557
    half_width = (p95_raw - p5_raw) / 2.0
    half_width_adj = half_width * gamma
    
    p5_adj = p50_raw - half_width_adj
    p95_adj = p50_raw + half_width_adj
    
    sigma_adj = half_width_adj / 1.645
    z_to_53 = (53.0 - p50_raw) / max(sigma_adj, 0.01)
    prob_adj = float(1.0 - norm.cdf(z_to_53))
    
    H_anchor = 19896
    s_p5_raw = (H_anchor + p5_raw) / N_pub
    s_p50_raw = (H_anchor + p50_raw) / N_pub
    s_p95_raw = (H_anchor + p95_raw) / N_pub
    
    s_p5_adj = (H_anchor + p5_adj) / N_pub
    s_p50_adj = (H_anchor + p50_raw) / N_pub
    s_p95_adj = (H_anchor + p95_adj) / N_pub
    
    verdict = "QUALIFIED" if p5_adj >= 53.0 else "NO_QUALIFY"
    
    print("\n  --- CLUSTERED BOOTSTRAP GAIN QUANTILE ESTIMATION ---")
    print(f"  Raw P5  (95% Conf Lower Bound): {p5_raw:+6.1f} hits | Public Score: {s_p5_raw:.5f}")
    print(f"  Raw P50 (Expected Median Gain): {p50_raw:+6.1f} hits | Public Score: {s_p50_raw:.5f}")
    print(f"  Raw P95 (Optimistic Upper 5%):  {p95_raw:+6.1f} hits | Public Score: {s_p95_raw:.5f}")
    print(f"  Raw P(Gain >= +53 hits / S >= 0.95000): {prob_raw*100:.2f}%")
    
    print(f"\n  --- PREQUENTIAL STRESS TEST (WIDENED BY gamma={gamma:.2f}) ---")
    print(f"  Adjusted P5:  {p5_adj:+6.1f} hits | Public Score P5:  {s_p5_adj:.5f}")
    print(f"  Adjusted P50: {p50_raw:+6.1f} hits | Public Score P50: {s_p50_adj:.5f}")
    print(f"  Adjusted P95: {p95_adj:+6.1f} hits | Public Score P95: {s_p95_adj:.5f}")
    print(f"  Stress-Tested P(Gain >= +53): {prob_adj*100:.2f}%")
    print(f"  Formal Qualification Status: {verdict} (Gate requirement: P5_adj >= +53)")
    
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "local_time_art": "2026-11-09 16:30 ART",
        "methodology": "Corrected Dual-Regime Frozen Holdout with Fold-Local Encodings and Margin Curation",
        "raw_bootstrap": {
            "p5_hits": p5_raw, "p50_hits": p50_raw, "p95_hits": p95_raw,
            "score_p5": s_p5_raw, "score_p50": s_p50_raw, "score_p95": s_p95_raw,
            "prob_ge_53": prob_raw
        },
        "stress_test_prequential": {
            "gamma": gamma,
            "p5_hits": p5_adj, "p50_hits": p50_raw, "p95_hits": p95_adj,
            "score_p5": s_p5_adj, "score_p50": s_p50_adj, "score_p95": s_p95_adj,
            "prob_ge_53": prob_adj
        },
        "holdout_performance": {
            "seen_net": holdout_data["net_seen"],
            "unseen_net": holdout_data["net_unseen"],
            "total_net": holdout_data["tot_net"],
            "fixes": holdout_data["tot_fixes"],
            "breaks": holdout_data["tot_breaks"],
            "comp_acc_base": holdout_data["acc_comp_base"],
            "comp_acc_cand": holdout_data["acc_comp_cand"]
        },
        "test_submission": {
            "path": holdout_data["test_csv_path"],
            "sha256": holdout_data["test_csv_hash"],
            "n_flips": holdout_data["test_n_flips"]
        },
        "verdict": verdict
    }
    
    with open(OUT_DIR / "dual_regime_audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Audit Report saved: {OUT_DIR / 'dual_regime_audit_report.json'}")
    return report


def main():
    print("=" * 80)
    print("CORRECTED DUAL-REGIME FRAMEWORK: ZERO-LEAKAGE & HIGH-CONFIDENCE MARGIN AUDIT")
    print("Policy: Full compliance with CX-0138 and Ariel directives")
    print("=" * 80, flush=True)
    
    t0 = time.time()
    df_train, df_test = load_raw_data()
    
    fit_idx, seen_ho_idx, unseen_ho_idx = create_dual_regime_frozen_split(df_train)
    df_fit = df_train.iloc[fit_idx].reset_index(drop=True)
    
    oof_lgb, oof_xgb, oof_cat, oof_rdg = run_honest_cross_fitting(df_fit)
    meta_seen, meta_unseen, oof_meta = fit_dual_regime_meta_models(df_fit, oof_lgb, oof_xgb, oof_cat, oof_rdg)
    
    best_margin, margin_results = evaluate_high_confidence_margin_thresholds(df_fit["target"].values, oof_lgb, oof_meta)
    
    holdout_data = train_full_fit_and_evaluate_frozen_holdout(
        df_fit, df_train, seen_ho_idx, unseen_ho_idx, meta_seen, meta_unseen, best_margin, df_test
    )
    
    report = compute_clustered_uncertainty(holdout_data)
    
    print("\n" + "=" * 80)
    print(f"Corrected Dual-Regime Audit completed in {time.time()-t0:.1f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
