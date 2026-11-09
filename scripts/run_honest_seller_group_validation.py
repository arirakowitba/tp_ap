"""Honest Seller-Group Cross-Validation and Nested Stacking Pipeline.

Author: Antigravity
Date: 2026-11-09
Policy: Strict compliance with CX-0134, CX-0135, CX-0136, CX-0137, and Ariel's P0 Directives.

Key Scientific Guarantees:
1. Outer CV by Seller-Group:
   - 5-Fold GroupKFold strictly on seller_id. Zero seller overlap across folds.
2. Leak-Free Feature Pipelines:
   - TF-IDF and SVD (Word & Char) fitted strictly inside each outer train fold.
   - Category and geographic target encodings fitted strictly inside outer train.
   - Unseen sellers in outer val naturally default to global smoothed prior.
3. Inner Split for Early Stopping:
   - Outer train split into inner-train (85%) and inner-val (15%) for early stopping.
   - Outer test fold is 100% virgin and never touched during training or round selection.
4. Nested Meta-Stacking:
   - 5-Fold cross-fitted meta learner (L2 Logistic Regression).
   - Zero in-sample evaluation: every meta probability is purely out-of-fold.
5. Clustered Bootstrap & P5 Gain vs Baseline:
   - 1,000 seller-grouped bootstrap resamples simulating N_pub = 20,998.
   - Empirical percentiles P5, P50, P95 of net gain over strong baseline.
   - Qualification criterion: P5 >= +53 net hits.
"""

import os
import re
import sys
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

ROOT = Path(".")
OUT_DIR = ROOT / "results/honest_seller_group_validation_20261109"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)


def sha256_file(filepath):
    p = Path(filepath)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_base_tabular(rows):
    feats = {}
    prices = [float(r.get("price") or 0.0) for r in rows]
    init_prices = [float(r.get("initial_price") or 0.0) for r in rows]
    base_prices = [float(r.get("base_price") or 0.0) for r in rows]
    p_arr = np.array(prices, dtype=np.float32)
    feats["price"] = p_arr
    feats["log_price"] = np.log1p(np.maximum(p_arr, 0.0))
    feats["init_price"] = np.array(init_prices, dtype=np.float32)
    feats["base_price"] = np.array(base_prices, dtype=np.float32)
    feats["price_diff_init"] = feats["init_price"] - feats["price"]
    feats["price_ratio_init"] = feats["price"] / np.maximum(feats["init_price"], 1.0)
    feats["price_diff_base"] = feats["base_price"] - feats["price"]

    currencies = [str(r.get("currency_id") or "") for r in rows]
    feats["is_usd"] = np.array([1.0 if c == "USD" else 0.0 for c in currencies], dtype=np.float32)
    feats["is_ars"] = np.array([1.0 if c == "ARS" else 0.0 for c in currencies], dtype=np.float32)

    feats["init_qty"] = np.array([float(r.get("initial_quantity") or 0.0) for r in rows], dtype=np.float32)
    feats["avail_qty"] = np.array([float(r.get("available_quantity") or 0.0) for r in rows], dtype=np.float32)
    feats["sold_qty"] = np.array([float(r.get("sold_quantity") or 0.0) for r in rows], dtype=np.float32)
    feats["log_init_qty"] = np.log1p(np.maximum(feats["init_qty"], 0.0))
    feats["log_sold_qty"] = np.log1p(np.maximum(feats["sold_qty"], 0.0))
    feats["qty_sold_ratio"] = feats["sold_qty"] / np.maximum(feats["init_qty"], 1.0)
    feats["is_sold_zero"] = (feats["sold_qty"] == 0.0).astype(np.float32)
    feats["is_avail_one"] = (feats["avail_qty"] == 1.0).astype(np.float32)
    feats["is_init_one"] = (feats["init_qty"] == 1.0).astype(np.float32)

    sellers = [r.get("seller_address") or {} for r in rows]
    feats["has_city"] = np.array([1.0 if (s.get("city") or {}).get("name") else 0.0 for s in sellers], dtype=np.float32)
    feats["has_state"] = np.array([1.0 if (s.get("state") or {}).get("name") else 0.0 for s in sellers], dtype=np.float32)
    feats["is_capital"] = np.array([1.0 if "capital" in ((s.get("state") or {}).get("name") or "").lower() else 0.0 for s in sellers], dtype=np.float32)
    feats["is_buenos_aires"] = np.array([1.0 if "buenos aires" in ((s.get("state") or {}).get("name") or "").lower() else 0.0 for s in sellers], dtype=np.float32)

    non_mp = [r.get("non_mercado_pago_payment_methods") or [] for r in rows]
    feats["n_non_mp"] = np.array([float(len(m)) for m in non_mp], dtype=np.float32)
    feats["has_non_mp"] = (feats["n_non_mp"] > 0).astype(np.float32)

    ship = [r.get("shipping") or {} for r in rows]
    feats["free_shipping"] = np.array([1.0 if s.get("free_shipping") else 0.0 for s in ship], dtype=np.float32)
    feats["local_pick_up"] = np.array([1.0 if s.get("local_pick_up") else 0.0 for s in ship], dtype=np.float32)

    pics = [r.get("pictures") or [] for r in rows]
    feats["n_pics"] = np.array([float(len(p)) for p in pics], dtype=np.float32)
    feats["has_multiple_pics"] = (feats["n_pics"] > 1).astype(np.float32)

    tags = [r.get("tags") or [] for r in rows]
    feats["n_tags"] = np.array([float(len(t)) for t in tags], dtype=np.float32)
    feats["has_good_quality_thumb"] = np.array([1.0 if "good_quality_thumbnail" in t else 0.0 for t in tags], dtype=np.float32)
    feats["is_dragged_bids_gold"] = np.array([1.0 if "dragged_bids_and_visits" in t else 0.0 for t in tags], dtype=np.float32)

    # Text features
    titles = [str(r.get("title") or "") for r in rows]
    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_word_count"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["title_has_numbers"] = np.array([1.0 if any(c.isdigit() for c in t) else 0.0 for t in titles], dtype=np.float32)
    feats["title_has_exclamation"] = np.array([1.0 if "!" in t else 0.0 for t in titles], dtype=np.float32)

    # Categories
    cats = {
        "category_id": [str(r.get("category_id") or "MISSING") for r in rows],
        "listing_type_id": [str(r.get("listing_type_id") or "MISSING") for r in rows],
        "buying_mode": [str(r.get("buying_mode") or "MISSING") for r in rows],
        "status": [str(r.get("status") or "MISSING") for r in rows],
        "city": [str((s.get("city") or {}).get("name") or "MISSING") for s in sellers],
        "state": [str((s.get("state") or {}).get("name") or "MISSING") for s in sellers],
        "seller_id": [str(r.get("seller_id") or "MISSING") for r in rows],
    }

    texts = [f"{t} {' '.join(tg)}" for t, tg in zip(titles, tags)]
    return pd.DataFrame(feats), pd.DataFrame(cats), texts


def compute_target_encoding(train_s, train_y, test_s, weight=20.0):
    global_mean = float(np.mean(train_y))
    df = pd.DataFrame({"cat": train_s, "y": train_y})
    stats = df.groupby("cat")["y"].agg(["count", "mean"])
    smooth = (stats["count"] * stats["mean"] + weight * global_mean) / (stats["count"] + weight)
    smooth_dict = smooth.to_dict()
    te_test = test_s.map(smooth_dict).fillna(global_mean).values.astype(np.float32)
    return smooth_dict, global_mean, te_test


def main():
    log("=" * 75)
    log("HONEST SELLER-GROUP VALIDATION & NESTED STACKING PIPELINE")
    log("Policy: CX-0136 / CX-0137 P0 Gate (Promotion requires P5 >= +53 hits vs EVT-132)")
    log("=" * 75)

    t_start = time.time()
    train_path = ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines"
    log(f"Loading raw train dataset from {train_path}...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    N_tr = len(train_rows)
    y_hard = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=np.int32)
    seller_ids = np.array([r.get("seller_id") or -1 for r in train_rows])
    log(f"Loaded {N_tr} training items. Base used rate: {np.mean(y_hard):.4f}")

    # Extract base tabular and categorical
    df_base, df_cat, text_all = extract_base_tabular(train_rows)
    log(f"Extracted {df_base.shape[1]} tabular features across {N_tr} records.")

    # 5-Fold GroupKFold strictly by seller_id
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df_base, y_hard, groups=seller_ids))

    # OOF prediction matrices
    oof_lgb = np.zeros(N_tr, dtype=np.float64)
    oof_xgb = np.zeros(N_tr, dtype=np.float64)
    oof_cat = np.zeros(N_tr, dtype=np.float64)
    oof_ridge = np.zeros(N_tr, dtype=np.float64)

    # Base models specifications
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "n_estimators": 1500,
        "random_state": 42,
        "n_jobs": 6,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "n_estimators": 1500,
        "random_state": 42,
        "n_jobs": 6,
    }

    cat_params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 1200,
        "learning_rate": 0.06,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "thread_count": 6,
        "verbose": False,
    }

    fold_metrics = []

    for fold_num, (tr_idx, va_idx) in enumerate(folds, 1):
        log(f"\n{'='*30} OUTER SELLER-GROUP FOLD {fold_num} / 5 {'='*30}")
        t_fold = time.time()
        n_tr_sellers = len(np.unique(seller_ids[tr_idx]))
        n_va_sellers = len(np.unique(seller_ids[va_idx]))
        overlap = len(set(seller_ids[tr_idx]).intersection(set(seller_ids[va_idx])))
        log(f"Train items: {len(tr_idx)} ({n_tr_sellers} sellers) | Val items: {len(va_idx)} ({n_va_sellers} sellers) | Seller Overlap: {overlap}")
        assert overlap == 0, f"Critical error: seller overlap detected in fold {fold_num}!"

        # 1. Feature Engineering INSIDE outer train fold
        # TF-IDF Word (32 SVD) and Char (16 SVD)
        log("Fitting TF-IDF Word & Char Vectorizers inside outer train fold...")
        vec_word = TfidfVectorizer(max_features=25000, min_df=3, ngram_range=(1, 2), sublinear_tf=True)
        X_w_tr = vec_word.fit_transform([text_all[i] for i in tr_idx])
        X_w_va = vec_word.transform([text_all[i] for i in va_idx])

        svd_w = TruncatedSVD(n_components=32, random_state=42 + fold_num)
        w_svd_tr = svd_w.fit_transform(X_w_tr)
        w_svd_va = svd_w.transform(X_w_va)

        vec_char = TfidfVectorizer(max_features=20000, min_df=5, analyzer="char", ngram_range=(3, 5), sublinear_tf=True)
        X_c_tr = vec_char.fit_transform([text_all[i] for i in tr_idx])
        X_c_va = vec_char.transform([text_all[i] for i in va_idx])

        svd_c = TruncatedSVD(n_components=16, random_state=42 + fold_num)
        c_svd_tr = svd_c.fit_transform(X_c_tr)
        c_svd_va = svd_c.transform(X_c_va)

        # Assemble fold dataframes
        X_tr = df_base.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = df_base.iloc[va_idx].copy().reset_index(drop=True)

        for i in range(32):
            X_tr[f"svd_w_{i}"] = w_svd_tr[:, i].astype(np.float32)
            X_va[f"svd_w_{i}"] = w_svd_va[:, i].astype(np.float32)
        for i in range(16):
            X_tr[f"svd_c_{i}"] = c_svd_tr[:, i].astype(np.float32)
            X_va[f"svd_c_{i}"] = c_svd_va[:, i].astype(np.float32)

        # Target encodings strictly from outer train
        for col_name, wt in [("category_id", 20.0), ("city", 10.0), ("state", 10.0)]:
            _, g_mean, te_va = compute_target_encoding(
                df_cat[col_name].iloc[tr_idx], y_hard[tr_idx], df_cat[col_name].iloc[va_idx], weight=wt
            )
            # Inner OOF target encoding for outer train
            te_tr_inner = np.zeros(len(tr_idx), dtype=np.float32)
            inner_gkf = GroupKFold(n_splits=3)
            inner_sellers = seller_ids[tr_idx]
            for in_tr, in_va in inner_gkf.split(X_tr, y_hard[tr_idx], groups=inner_sellers):
                _, _, te_in_va = compute_target_encoding(
                    df_cat[col_name].iloc[tr_idx].iloc[in_tr], y_hard[tr_idx][in_tr], df_cat[col_name].iloc[tr_idx].iloc[in_va], weight=wt
                )
                te_tr_inner[in_va] = te_in_va

            X_tr[f"te_{col_name}"] = te_tr_inner
            X_va[f"te_{col_name}"] = te_va

        # Categorical columns for LightGBM
        cat_cols = ["listing_type_id", "buying_mode", "status"]
        for c in cat_cols:
            X_tr[c] = df_cat[c].iloc[tr_idx].astype("category").values
            X_va[c] = df_cat[c].iloc[va_idx].astype("category").values

        y_tr = y_hard[tr_idx]
        y_va = y_hard[va_idx]

        # 2. Split outer train into inner-train (85%) and inner-val (15%) for honest early stopping
        # Grouped by seller so inner early stopping also respects seller clusters!
        inner_sellers_fold = seller_ids[tr_idx]
        unique_in_sellers = np.unique(inner_sellers_fold)
        np.random.seed(42 + fold_num)
        np.random.shuffle(unique_in_sellers)
        split_point = int(len(unique_in_sellers) * 0.85)
        in_tr_sellers = set(unique_in_sellers[:split_point])
        
        in_tr_mask = np.array([s in in_tr_sellers for s in inner_sellers_fold])
        in_va_mask = ~in_tr_mask

        log(f"Inner Early Stopping Split: Train={in_tr_mask.sum()}, Val={in_va_mask.sum()} | Outer Val={len(va_idx)} VIRGIN")

        # MODEL 1: LightGBM
        log("Training Model 1: LightGBM Leaf-Wise...")
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(
            X_tr[in_tr_mask], y_tr[in_tr_mask],
            eval_set=[(X_tr[in_va_mask], y_tr[in_va_mask])],
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)]
        )
        p_lgb_va = model_lgb.predict_proba(X_va)[:, 1]
        oof_lgb[va_idx] = p_lgb_va
        acc_lgb = accuracy_score(y_va, (p_lgb_va >= 0.5).astype(int))
        log(f"LGBM Fold {fold_num} Acc: {acc_lgb:.5f} | LogLoss: {log_loss(y_va, p_lgb_va):.5f}")

        # MODEL 2: XGBoost
        log("Training Model 2: XGBoost Hist...")
        # Drop category dtypes for XGBoost
        X_tr_num = X_tr.drop(columns=cat_cols)
        X_va_num = X_va.drop(columns=cat_cols)
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(
            X_tr_num[in_tr_mask], y_tr[in_tr_mask],
            eval_set=[(X_tr_num[in_va_mask], y_tr[in_va_mask])],
            verbose=False
        )
        p_xgb_va = model_xgb.predict_proba(X_va_num)[:, 1]
        oof_xgb[va_idx] = p_xgb_va
        acc_xgb = accuracy_score(y_va, (p_xgb_va >= 0.5).astype(int))
        log(f"XGB  Fold {fold_num} Acc: {acc_xgb:.5f} | LogLoss: {log_loss(y_va, p_xgb_va):.5f}")

        # MODEL 3: CatBoost
        log("Training Model 3: CatBoost Symmetric...")
        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(
            X_tr_num[in_tr_mask], y_tr[in_tr_mask],
            eval_set=[(X_tr_num[in_va_mask], y_tr[in_va_mask])],
            early_stopping_rounds=60,
            verbose=False
        )
        p_cat_va = model_cat.predict_proba(X_va_num)[:, 1]
        oof_cat[va_idx] = p_cat_va
        acc_cat = accuracy_score(y_va, (p_cat_va >= 0.5).astype(int))
        log(f"CatB Fold {fold_num} Acc: {acc_cat:.5f} | LogLoss: {log_loss(y_va, p_cat_va):.5f}")

        # MODEL 4: Ridge Linear on Text Word SVD + Char SVD
        log("Training Model 4: Ridge Text Specialist...")
        svd_cols = [c for c in X_tr.columns if c.startswith("svd_")]
        ridge = RidgeClassifier(alpha=10.0, random_state=42)
        ridge.fit(X_tr[svd_cols], y_tr)
        # Decision function calibrated via sigmoid
        d_val = ridge.decision_function(X_va[svd_cols])
        p_ridge_va = expit(d_val)
        oof_ridge[va_idx] = p_ridge_va
        acc_ridge = accuracy_score(y_va, (p_ridge_va >= 0.5).astype(int))
        log(f"Ridge Fold {fold_num} Acc: {acc_ridge:.5f} | LogLoss: {log_loss(y_va, p_ridge_va):.5f}")

        fold_metrics.append({
            "fold": fold_num,
            "acc_lgb": acc_lgb,
            "acc_xgb": acc_xgb,
            "acc_cat": acc_cat,
            "acc_ridge": acc_ridge,
            "duration_s": time.time() - t_fold
        })

    # Global Base Models Evaluation
    log("\n" + "=" * 60)
    log("OUTER SELLER-GROUP BASE MODELS SUMMARY (HONEST OOF):")
    for name, p_arr in [("LightGBM", oof_lgb), ("XGBoost", oof_xgb), ("CatBoost", oof_cat), ("Ridge Text", oof_ridge)]:
        acc = accuracy_score(y_hard, (p_arr >= 0.5).astype(int))
        ll = log_loss(y_hard, np.clip(p_arr, 1e-6, 1.0 - 1e-6))
        auc = roc_auc_score(y_hard, p_arr)
        hits = int(round(acc * N_tr))
        log(f"{name:<15} | Acc: {acc:.5f} ({hits}/{N_tr}) | LogLoss: {ll:.5f} | AUC: {auc:.5f}")
    log("=" * 60)

    # 3. Nested Meta-Stacking (Cross-Fitting on Level-1)
    log("\nExecuting Nested 5-Fold Cross-Fitting for Meta-Stacker...")
    X_meta = np.column_stack([
        logit(np.clip(oof_lgb, 1e-6, 1.0 - 1e-6)),
        logit(np.clip(oof_xgb, 1e-6, 1.0 - 1e-6)),
        logit(np.clip(oof_cat, 1e-6, 1.0 - 1e-6)),
        logit(np.clip(oof_ridge, 1e-6, 1.0 - 1e-6)),
    ])

    oof_meta = np.zeros(N_tr, dtype=np.float64)
    # Cross-fitting meta model by seller-group as well!
    for fold_num, (tr_idx, va_idx) in enumerate(folds, 1):
        meta_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        meta_lr.fit(X_meta[tr_idx], y_hard[tr_idx])
        oof_meta[va_idx] = meta_lr.predict_proba(X_meta[va_idx])[:, 1]

    meta_acc = accuracy_score(y_hard, (oof_meta >= 0.5).astype(int))
    meta_ll = log_loss(y_hard, np.clip(oof_meta, 1e-6, 1.0 - 1e-6))
    meta_auc = roc_auc_score(y_hard, oof_meta)
    meta_hits = int(round(meta_acc * N_tr))
    log(f"NESTED META-STACKER | Acc: {meta_acc:.5f} ({meta_hits}/{N_tr}) | LogLoss: {meta_ll:.5f} | AUC: {meta_auc:.5f}")

    # Baseline strong model analogue (LightGBM on outer seller-group)
    base_pred = (oof_lgb >= 0.5).astype(int)
    meta_pred = (oof_meta >= 0.5).astype(int)

    fixes = np.sum((meta_pred == y_hard) & (base_pred != y_hard))
    breaks = np.sum((meta_pred != y_hard) & (base_pred == y_hard))
    net_gain = fixes - breaks
    log(f"\nPairwise Comparison (Meta vs LGBM Base) on 70k Train Items:")
    log(f"Fixes: {fixes} | Breaks: {breaks} | Net Gain: {net_gain:+d} hits")

    # 4. Seller-Grouped Clustered Bootstrap Uncertainty (1,000 resamples)
    log("\nComputing Seller-Grouped Cluster Bootstrap Uncertainty (B=1,000)...")
    unique_sellers, seller_inverse = np.unique(seller_ids, return_inverse=True)
    n_unique_sellers = len(unique_sellers)
    
    seller_to_items = [[] for _ in range(n_unique_sellers)]
    for i, s_idx in enumerate(seller_inverse):
        seller_to_items[s_idx].append(i)

    np.random.seed(20261109)
    N_pub = 20998
    B = 1000
    delta_hits_samples = []

    for b in range(B):
        sampled_sellers = np.random.choice(n_unique_sellers, size=int(n_unique_sellers * (N_pub / N_tr) * 1.05), replace=True)
        sampled_items = []
        for s in sampled_sellers:
            sampled_items.extend(seller_to_items[s])
            if len(sampled_items) >= N_pub:
                break
        sampled_idx = np.array(sampled_items[:N_pub])
        
        y_b = y_hard[sampled_idx]
        p_meta_b = oof_meta[sampled_idx]
        p_base_b = oof_lgb[sampled_idx]
        
        hits_meta = np.sum((p_meta_b >= 0.5).astype(int) == y_b)
        hits_base = np.sum((p_base_b >= 0.5).astype(int) == y_b)
        delta_hits_samples.append(hits_meta - hits_base)

    delta_hits_samples = np.array(delta_hits_samples)
    p5 = float(np.percentile(delta_hits_samples, 5))
    p50 = float(np.percentile(delta_hits_samples, 50))
    p95 = float(np.percentile(delta_hits_samples, 95))
    prob_ge_53 = float(np.mean(delta_hits_samples >= 53))

    H_anchor = 19896
    score_p5 = (H_anchor + p5) / N_pub
    score_p50 = (H_anchor + p50) / N_pub
    score_p95 = (H_anchor + p95) / N_pub

    gate_verdict = "QUALIFIED" if p5 >= 53 else "NO_QUALIFY"

    log("\n" + "=" * 60)
    log("CALIBRATED CLUSTERED UNCERTAINTY RESULTS:")
    log(f"Public Net Gain P5  (95% Conf Lower Bound): {p5:+.1f} hits | Score P5:  {score_p5:.5f}")
    log(f"Public Net Gain P50 (Median Expected):       {p50:+.1f} hits | Score P50: {score_p50:.5f}")
    log(f"Public Net Gain P95 (Top 5% Optimistic):     {p95:+.1f} hits | Score P95: {score_p95:.5f}")
    log(f"P(gain >= +53 hits / S >= 0.95000): {prob_ge_53*100:.2f}%")
    log(f"Operational Verdict: {gate_verdict} (Requirement: P5 >= +53)")
    log("=" * 60)

    # 5. Save all outputs
    log("\nSaving validation outputs and hashes...")
    df_oof = pd.DataFrame({
        "ID": range(N_tr),
        "seller_id": seller_ids,
        "true_condition": y_hard,
        "prob_lgb": oof_lgb,
        "prob_xgb": oof_xgb,
        "prob_cat": oof_cat,
        "prob_ridge": oof_ridge,
        "prob_nested_meta": oof_meta,
    })
    oof_csv_path = OUT_DIR / "honest_seller_group_oof_predictions.csv"
    df_oof.to_csv(oof_csv_path, index=False)

    np.save(OUT_DIR / "bootstrap_gain_distribution.npy", delta_hits_samples)

    cert = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "policy": "CX-0137",
        "validation_scheme": "5-Fold GroupKFold by seller_id (Zero seller overlap)",
        "inner_early_stopping": "85/15 seller-grouped split inside outer train (Virgin outer val)",
        "meta_cross_fitting": "5-Fold GroupKFold nested L2 Logistic Regression",
        "n_train": N_tr,
        "n_public_simulated": N_pub,
        "bootstrap_iterations": B,
        "base_models": {
            "lgbm": {"acc": float(accuracy_score(y_hard, (oof_lgb >= 0.5).astype(int))), "logloss": float(log_loss(y_hard, oof_lgb))},
            "xgb": {"acc": float(accuracy_score(y_hard, (oof_xgb >= 0.5).astype(int))), "logloss": float(log_loss(y_hard, oof_xgb))},
            "catboost": {"acc": float(accuracy_score(y_hard, (oof_cat >= 0.5).astype(int))), "logloss": float(log_loss(y_hard, oof_cat))},
            "ridge": {"acc": float(accuracy_score(y_hard, (oof_ridge >= 0.5).astype(int))), "logloss": float(log_loss(y_hard, oof_ridge))},
        },
        "nested_meta_model": {
            "acc": float(meta_acc),
            "logloss": float(meta_ll),
            "auc": float(meta_auc),
            "hits": int(meta_hits),
        },
        "pairwise_vs_lgbm_base": {
            "fixes": int(fixes),
            "breaks": int(breaks),
            "net_gain": int(net_gain),
        },
        "calibrated_uncertainty": {
            "p5_hits": p5,
            "p50_hits": p50,
            "p95_hits": p95,
            "score_p5": score_p5,
            "score_p50": score_p50,
            "score_p95": score_p95,
            "prob_public_score_ge_095": prob_ge_53,
        },
        "verdict": gate_verdict,
        "files_sha256": {
            "train_data": sha256_file(train_path),
            "oof_predictions": sha256_file(oof_csv_path),
        }
    }

    cert_path = OUT_DIR / "confidence_certificate_preliminary.json"
    with open(cert_path, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)

    log(f"Certificate saved to {cert_path}")
    log(f"Pipeline finished in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
