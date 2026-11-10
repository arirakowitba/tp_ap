#!/usr/bin/env python3
"""Canonical Nested Dual-Regime Training & Paired Baseline Verification.

Implements Agent 2, 3, 4, 5, 6 mandates under the Canonical Split Manifest:
- Loads data/splits/canonical_nested_dual_regime_splits.csv (Agent 1).
- In each fold k:
  * fit_set (~49.580 rows): strictly used for feature fitting and model training.
  * seen_val (~6.415 rows): true out-of-fold holdout from seen sellers (fit_set ∩ seen_val = ∅).
  * unseen_val (14.000 rows): true out-of-fold holdout from 100% unseen sellers (fit_set ∩ unseen_val = ∅).
- Multi-channel strong text: Word (1-2) + Char-WB (3-5) TF-IDF (45.000 max_features) + 120 TruncatedSVD components.
- Zero leakage: TF-IDF, SVD, category medians, and leave-one-out seller TE fitted strictly on fit_set.
- Inner seller-group split on fit_set solely to determine best_iteration and inner meta-blend weights.
- Outer models trained on 100% of fit_set without eval_set.
- Evaluates:
  * CatBoost Depth 8 (Agent 2)
  * LightGBM 127 Leaves (Agent 3)
  * Asymmetric Lossguide XGBoost (Agent 3)
  * Strong Multi-Channel Text Logistic SAGA (Agent 4)
  * Multi-Model Cross-Fitted Meta-Ensemble (Agent 6)
- Paired comparison row-by-row vs verified strong baseline (oof_predictions_podium_engine.csv, 0.94230 OOF).
- Reports fixes, breaks, net hits, seller-clustered bootstrap 95% CI, and category-clustered bootstrap 95% CI.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.train_strictly_nested_dual_regime_cv import extract_raw_fields


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_fold_features_canonical(
    fit_idx: np.ndarray,
    eval_indices: list[np.ndarray],
    corpus_all: list[str],
    df_base_all: pd.DataFrame,
    cats_all: dict[str, np.ndarray],
    y_all: np.ndarray,
    n_svd: int = 120,
) -> tuple[pd.DataFrame, list[pd.DataFrame], sparse.csr_matrix, list[sparse.csr_matrix]]:
    """Compute features strictly fitted on fit_idx and transform eval_indices."""
    # 1. High-Capacity Text Representations: Word (1-2) + Char-WB (3-5)
    fit_corpus = [corpus_all[i] for i in fit_idx]
    
    vec_word = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        min_df=3,
    )
    vec_char = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char_wb",
        max_features=25000,
        sublinear_tf=True,
        min_df=5,
    )

    X_fit_w = vec_word.fit_transform(fit_corpus)
    X_fit_c = vec_char.fit_transform(fit_corpus)
    X_fit_tfidf = sparse.hstack([X_fit_w, X_fit_c]).tocsr()

    X_eval_tfidfs = []
    for idx in eval_indices:
        ev_corpus = [corpus_all[i] for i in idx]
        ev_w = vec_word.transform(ev_corpus)
        ev_c = vec_char.transform(ev_corpus)
        X_eval_tfidfs.append(sparse.hstack([ev_w, ev_c]).tocsr())

    # TruncatedSVD on fit_set TF-IDF
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    X_fit_svd = svd.fit_transform(X_fit_tfidf).astype(np.float32)
    X_eval_svds = [svd.transform(X_eval).astype(np.float32) for X_eval in X_eval_tfidfs]

    svd_cols = [f"svd_{i}" for i in range(n_svd)]
    df_fit_svd = pd.DataFrame(X_fit_svd, columns=svd_cols, index=fit_idx)
    df_eval_svds = [pd.DataFrame(X_svd, columns=svd_cols, index=idx) for X_svd, idx in zip(X_eval_svds, eval_indices)]

    # 2. Category Intrinsic Statistics (Price ratios, z-scores, frequencies)
    df_fit_base = df_base_all.iloc[fit_idx].copy()
    df_eval_bases = [df_base_all.iloc[idx].copy() for idx in eval_indices]

    cat_fit = cats_all["category_id"][fit_idx]
    global_med_price = float(df_fit_base["log_price"].median())
    global_std_price = float(df_fit_base["log_price"].std()) if float(df_fit_base["log_price"].std()) > 0 else 1.0

    cat_med_price = df_fit_base.groupby(cat_fit)["log_price"].median().to_dict()
    cat_std_price = df_fit_base.groupby(cat_fit)["log_price"].std().fillna(global_std_price).to_dict()

    df_fit_base["rel_price_cat"] = (df_fit_base["log_price"] - pd.Series(cat_fit, index=fit_idx).map(cat_med_price).fillna(global_med_price)).astype(np.float32)
    df_fit_base["cat_price_zscore"] = np.clip(
        (df_fit_base["log_price"] - pd.Series(cat_fit, index=fit_idx).map(cat_med_price).fillna(global_med_price)) /
        pd.Series(cat_fit, index=fit_idx).map(cat_std_price).fillna(global_std_price),
        -8.0, 8.0
    ).astype(np.float32)

    for df_eval_b, idx in zip(df_eval_bases, eval_indices):
        cat_ev = cats_all["category_id"][idx]
        df_eval_b["rel_price_cat"] = (df_eval_b["log_price"] - pd.Series(cat_ev, index=idx).map(cat_med_price).fillna(global_med_price)).astype(np.float32)
        df_eval_b["cat_price_zscore"] = np.clip(
            (df_eval_b["log_price"] - pd.Series(cat_ev, index=idx).map(cat_med_price).fillna(global_med_price)) /
            pd.Series(cat_ev, index=idx).map(cat_std_price).fillna(global_std_price),
            -8.0, 8.0
        ).astype(np.float32)

    # Frequency encodings
    for cat_name in ["category_id", "listing_type", "buying_mode", "state", "ship_mode"]:
        counts = Counter(cats_all[cat_name][fit_idx])
        df_fit_base[f"freq_{cat_name}"] = pd.Series(cats_all[cat_name][fit_idx], index=fit_idx).map(counts).fillna(0).astype(np.float32)
        for df_eval_b, idx in zip(df_eval_bases, eval_indices):
            df_eval_b[f"freq_{cat_name}"] = pd.Series(cats_all[cat_name][idx], index=idx).map(counts).fillna(0).astype(np.float32)

    # 3. Seller TE with Leave-One-Out (Agent 5 requirement)
    # Fit seller TE on fit_idx
    seller_fit = cats_all["seller_id"][fit_idx]
    y_fit = y_all[fit_idx]
    global_prior = float(np.mean(y_fit))
    m = 10.0

    seller_stats = pd.DataFrame({"seller": seller_fit, "y": y_fit}).groupby("seller")["y"].agg(["count", "sum"])
    seller_map = ((seller_stats["sum"] + m * global_prior) / (seller_stats["count"] + m)).to_dict()

    # Leave-One-Out for fit_idx
    s_series = pd.Series(seller_fit, index=fit_idx)
    c_counts = s_series.map(seller_stats["count"]).fillna(0.0).values
    c_sums = s_series.map(seller_stats["sum"]).fillna(0.0).values
    df_fit_base["seller_te_loo"] = np.where(
        c_counts > 1.0,
        (c_sums - y_fit + m * global_prior) / (c_counts - 1.0 + m),
        global_prior
    ).astype(np.float32)

    # Smooth mapping for eval indices
    for df_eval_b, idx in zip(df_eval_bases, eval_indices):
        s_ev = pd.Series(cats_all["seller_id"][idx], index=idx)
        df_eval_b["seller_te_loo"] = s_ev.map(seller_map).fillna(global_prior).astype(np.float32)

    X_fit_dense = pd.concat([df_fit_base, df_fit_svd], axis=1)
    X_eval_denses = [pd.concat([df_b, df_s], axis=1) for df_b, df_s in zip(df_eval_bases, df_eval_svds)]

    return X_fit_dense, X_eval_denses, X_fit_tfidf, X_eval_tfidfs


def clustered_bootstrap_ci(
    diff: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261110,
) -> tuple[float, float, float, float]:
    """Cluster bootstrap for delta accuracy grouped by cluster_ids."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    cluster_to_indices = {}
    for idx, c in enumerate(cluster_ids):
        cluster_to_indices.setdefault(c, []).append(idx)
    cluster_to_indices = {c: np.array(ixs) for c, ixs in cluster_to_indices.items()}

    boot_deltas = np.zeros(n_boot)
    for b in range(n_boot):
        sample_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = np.concatenate([cluster_to_indices[c] for c in sample_clusters])
        boot_deltas[b] = np.mean(diff[sampled_indices])

    mean_delta = float(np.mean(boot_deltas))
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    p_gt_zero = float(np.mean(boot_deltas > 0))
    return mean_delta, ci_lower, ci_upper, p_gt_zero


def main():
    print("=" * 80)
    print("CANONICAL NESTED DUAL-REGIME TRAINING & VERIFICATION (AGENTS 2-6)")
    print("Strict Out-of-Fold Evaluation for Both Seen and Unseen Seller Regimes")
    print("=" * 80)

    start_total = time.time()
    out_dir = Path("results/canonical_nested_models_20261110")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Canonical Split Manifest
    splits_csv_path = Path("data/splits/canonical_nested_dual_regime_splits.csv")
    assert splits_csv_path.exists(), "Missing canonical splits! Run build_canonical_nested_splits.py first."
    df_splits = pd.read_csv(splits_csv_path)
    n_train = len(df_splits)
    print(f"Loaded Canonical Splits Manifest: {len(df_splits)} items", flush=True)

    # 2. Load Raw Datasets
    train_path = Path("data/ap_2026_q_2/raw/train_data.jsonlines")
    test_path = Path("data/ap_2026_q_2/raw/test_data.jsonlines")
    baseline_path = Path("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")

    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    n_test = len(test_rows)
    y_train = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=int)

    # Baseline verification
    df_baseline = pd.read_csv(baseline_path)
    p_baseline = df_baseline["oof_blend"].values
    pred_baseline = (p_baseline >= 0.5).astype(int)
    acc_baseline = accuracy_score(y_train, pred_baseline)
    ll_baseline = log_loss(y_train, p_baseline)
    print(f"Strong Baseline (Podium Engine): Acc = {acc_baseline:.5f} ({int(acc_baseline*n_train)} hits), LogLoss = {ll_baseline:.5f}")

    # Extract raw features
    print("Extracting raw intrinsic tabular features and text corpus...", flush=True)
    train_ids, train_corpus, df_train_base, train_cats = extract_raw_fields(train_rows)
    test_ids, test_corpus, df_test_base, test_cats = extract_raw_fields(test_rows)
    sellers_train = train_cats["seller_id"]
    categories_train = train_cats["category_id"]

    # Storage for Honest OOF Predictions
    # Unseen-sellers OOF (full 70k: exactly one prediction per item from its outer fold)
    oof_unseen_p_cb = np.zeros(n_train)
    oof_unseen_p_lgb = np.zeros(n_train)
    oof_unseen_p_xgb = np.zeros(n_train)
    oof_unseen_p_text = np.zeros(n_train)
    oof_unseen_p_quad = np.zeros(n_train)

    # Seen-sellers OOF (all seen_val holdout rows across 5 folds, exactly 0 fit rows)
    seen_val_indices_all = []
    seen_val_p_cb_all = []
    seen_val_p_lgb_all = []
    seen_val_p_xgb_all = []
    seen_val_p_text_all = []
    seen_val_p_quad_all = []

    # Test predictions (average across 5 outer folds)
    test_p_cb = np.zeros(n_test)
    test_p_lgb = np.zeros(n_test)
    test_p_xgb = np.zeros(n_test)
    test_p_text = np.zeros(n_test)
    test_p_quad = np.zeros(n_test)

    n_splits = 5
    fold_reports = []

    for fold in range(n_splits):
        fold_start = time.time()
        print(f"\n{'='*30} FOLD {fold+1}/{n_splits} {'='*30}", flush=True)

        fit_idx = np.where(df_splits[f"role_in_fold_{fold}"] == "fit")[0]
        seen_val_idx = np.where(df_splits[f"role_in_fold_{fold}"] == "seen_val")[0]
        unseen_val_idx = np.where(df_splits[f"role_in_fold_{fold}"] == "unseen_val")[0]

        print(f"Role Counts: Fit={len(fit_idx)}, SeenVal={len(seen_val_idx)} (0% fit overlap!), UnseenVal={len(unseen_val_idx)} (0% fit overlap!)", flush=True)
        assert len(set(fit_idx) & set(seen_val_idx)) == 0, "FATAL: fit_idx and seen_val_idx overlap!"
        assert len(set(fit_idx) & set(unseen_val_idx)) == 0, "FATAL: fit_idx and unseen_val_idx overlap!"

        # Inner Split on fit_idx for early stopping iteration freezing
        fit_sellers = sellers_train[fit_idx]
        fit_targets = y_train[fit_idx]

        inner_sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42 + fold)
        in_tr_pos, in_va_pos = next(inner_sgkf.split(fit_idx, fit_targets, groups=fit_sellers))
        inner_tr_idx = fit_idx[in_tr_pos]
        inner_va_idx = fit_idx[in_va_pos]
        assert len(set(sellers_train[inner_tr_idx]) & set(sellers_train[inner_va_idx])) == 0, "Inner seller overlap!"

        # Stage 1: Feature Extraction on Inner Split
        X_in_tr_dense, [X_in_va_dense], X_in_tr_tfidf, [X_in_va_tfidf] = compute_fold_features_canonical(
            inner_tr_idx,
            [inner_va_idx],
            train_corpus,
            df_train_base,
            train_cats,
            y_train,
            n_svd=120,
        )

        # Stage 2: Inner Early Stopping Iteration Freezing
        print("Tuning iterations on inner validation set...", flush=True)

        # CatBoost Inner
        cb_in = CatBoostClassifier(iterations=750, learning_rate=0.04, depth=8, l2_leaf_reg=5.0, random_seed=42+fold, verbose=0, thread_count=4)
        cb_in.fit(X_in_tr_dense, y_train[inner_tr_idx], eval_set=(X_in_va_dense, y_train[inner_va_idx]), early_stopping_rounds=60, verbose=False)
        best_iter_cb = cb_in.get_best_iteration() or 500
        p_in_va_cb = cb_in.predict_proba(X_in_va_dense)[:, 1]

        # LightGBM Inner
        lgb_in = lgb.LGBMClassifier(n_estimators=750, learning_rate=0.04, num_leaves=127, max_depth=9, random_state=42+fold, n_jobs=4, verbose=-1)
        lgb_in.fit(X_in_tr_dense, y_train[inner_tr_idx], eval_set=[(X_in_va_dense, y_train[inner_va_idx])], callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)])
        best_iter_lgb = lgb_in.best_iteration_ or 350
        p_in_va_lgb = lgb_in.predict_proba(X_in_va_dense)[:, 1]

        # Asymmetric XGBoost Inner
        xgb_in = xgb.XGBClassifier(tree_method="hist", grow_policy="lossguide", max_leaves=120, max_depth=11, learning_rate=0.035, n_estimators=750, random_state=42+fold, n_jobs=4, eval_metric="logloss", early_stopping_rounds=60)
        xgb_in.fit(X_in_tr_dense, y_train[inner_tr_idx], eval_set=[(X_in_va_dense, y_train[inner_va_idx])], verbose=False)
        best_iter_xgb = xgb_in.best_iteration or 400
        p_in_va_xgb = xgb_in.predict_proba(X_in_va_dense)[:, 1]

        # Strong Text Logistic Inner
        lr_in = LogisticRegression(C=1.5, max_iter=400, solver="saga", random_state=42+fold, n_jobs=4)
        lr_in.fit(X_in_tr_tfidf, y_train[inner_tr_idx])
        p_in_va_text = lr_in.predict_proba(X_in_va_tfidf)[:, 1]

        # Inner Meta-Weights Optimization
        def inner_loss(w):
            w = np.maximum(0, w)
            w = w / np.sum(w)
            p = w[0]*p_in_va_cb + w[1]*p_in_va_lgb + w[2]*p_in_va_xgb + w[3]*p_in_va_text
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return log_loss(y_train[inner_va_idx], p)

        res_w = minimize(inner_loss, [0.25, 0.35, 0.35, 0.05], bounds=[(0, 1)]*4, method="L-BFGS-B")
        w_inner = res_w.x / np.sum(res_w.x)
        print(f"Frozen Hyperparameters Fold {fold+1}: CB={best_iter_cb}, LGB={best_iter_lgb}, XGB={best_iter_xgb}", flush=True)
        print(f"Inner Optimal Weights: CB={w_inner[0]:.4f}, LGB={w_inner[1]:.4f}, XGB={w_inner[2]:.4f}, Text={w_inner[3]:.4f}", flush=True)

        del cb_in, lgb_in, xgb_in, lr_in, X_in_tr_dense, X_in_va_dense, X_in_tr_tfidf, X_in_va_tfidf
        gc.collect()

        # Stage 3: Fit Features on 100% of fit_idx and transform seen_val, unseen_val, test
        print(f"Refitting features strictly on fit_set ({len(fit_idx)} items)...", flush=True)
        test_eval_indices = n_train + np.arange(n_test)
        X_fit_dense, [X_seen_dense, X_unseen_dense, X_test_dense], X_fit_tfidf, [X_seen_tfidf, X_unseen_tfidf, X_test_tfidf] = compute_fold_features_canonical(
            fit_idx,
            [seen_val_idx, unseen_val_idx, test_eval_indices],
            train_corpus + test_corpus,
            pd.concat([df_train_base, df_test_base], ignore_index=True),
            {k: np.concatenate([train_cats[k], test_cats[k]]) for k in train_cats},
            np.concatenate([y_train, np.zeros(n_test, dtype=int)]),
            n_svd=120,
        )

        # Stage 4: Train Outer Models on fit_set WITHOUT eval_set
        print(f"Training Outer Models on fit_set (NO eval_set, frozen iterations)...", flush=True)

        cb_outer = CatBoostClassifier(iterations=best_iter_cb, learning_rate=0.04, depth=8, l2_leaf_reg=5.0, random_seed=42+fold, verbose=0, thread_count=4)
        cb_outer.fit(X_fit_dense, y_train[fit_idx], verbose=False)
        p_seen_cb = cb_outer.predict_proba(X_seen_dense)[:, 1]
        p_unseen_cb = cb_outer.predict_proba(X_unseen_dense)[:, 1]
        p_test_cb_f = cb_outer.predict_proba(X_test_dense)[:, 1]

        lgb_outer = lgb.LGBMClassifier(n_estimators=best_iter_lgb, learning_rate=0.04, num_leaves=127, max_depth=9, random_state=42+fold, n_jobs=4, verbose=-1)
        lgb_outer.fit(X_fit_dense, y_train[fit_idx])
        p_seen_lgb = lgb_outer.predict_proba(X_seen_dense)[:, 1]
        p_unseen_lgb = lgb_outer.predict_proba(X_unseen_dense)[:, 1]
        p_test_lgb_f = lgb_outer.predict_proba(X_test_dense)[:, 1]

        xgb_outer = xgb.XGBClassifier(tree_method="hist", grow_policy="lossguide", max_leaves=120, max_depth=11, learning_rate=0.035, n_estimators=best_iter_xgb, random_state=42+fold, n_jobs=4, eval_metric="logloss")
        xgb_outer.fit(X_fit_dense, y_train[fit_idx], verbose=False)
        p_seen_xgb = xgb_outer.predict_proba(X_seen_dense)[:, 1]
        p_unseen_xgb = xgb_outer.predict_proba(X_unseen_dense)[:, 1]
        p_test_xgb_f = xgb_outer.predict_proba(X_test_dense)[:, 1]

        lr_outer = LogisticRegression(C=1.5, max_iter=400, solver="saga", random_state=42+fold, n_jobs=4)
        lr_outer.fit(X_fit_tfidf, y_train[fit_idx])
        p_seen_text = lr_outer.predict_proba(X_seen_tfidf)[:, 1]
        p_unseen_text = lr_outer.predict_proba(X_unseen_tfidf)[:, 1]
        p_test_text_f = lr_outer.predict_proba(X_test_tfidf)[:, 1]

        # Apply frozen inner weights
        p_seen_quad = w_inner[0]*p_seen_cb + w_inner[1]*p_seen_lgb + w_inner[2]*p_seen_xgb + w_inner[3]*p_seen_text
        p_unseen_quad = w_inner[0]*p_unseen_cb + w_inner[1]*p_unseen_lgb + w_inner[2]*p_unseen_xgb + w_inner[3]*p_unseen_text
        p_test_quad_f = w_inner[0]*p_test_cb_f + w_inner[1]*p_test_lgb_f + w_inner[2]*p_test_xgb_f + w_inner[3]*p_test_text_f

        # Record Unseen OOF
        oof_unseen_p_cb[unseen_val_idx] = p_unseen_cb
        oof_unseen_p_lgb[unseen_val_idx] = p_unseen_lgb
        oof_unseen_p_xgb[unseen_val_idx] = p_unseen_xgb
        oof_unseen_p_text[unseen_val_idx] = p_unseen_text
        oof_unseen_p_quad[unseen_val_idx] = p_unseen_quad

        # Record Seen OOF
        seen_val_indices_all.extend(seen_val_idx.tolist())
        seen_val_p_cb_all.extend(p_seen_cb.tolist())
        seen_val_p_lgb_all.extend(p_seen_lgb.tolist())
        seen_val_p_xgb_all.extend(p_seen_xgb.tolist())
        seen_val_p_text_all.extend(p_seen_text.tolist())
        seen_val_p_quad_all.extend(p_seen_quad.tolist())

        # Test average
        test_p_cb += p_test_cb_f / n_splits
        test_p_lgb += p_test_lgb_f / n_splits
        test_p_xgb += p_test_xgb_f / n_splits
        test_p_text += p_test_text_f / n_splits
        test_p_quad += p_test_quad_f / n_splits

        # Metrics for fold
        acc_seen_fold = accuracy_score(y_train[seen_val_idx], (p_seen_quad >= 0.5).astype(int))
        acc_unseen_fold = accuracy_score(y_train[unseen_val_idx], (p_unseen_quad >= 0.5).astype(int))
        ll_seen_fold = log_loss(y_train[seen_val_idx], p_seen_quad)
        ll_unseen_fold = log_loss(y_train[unseen_val_idx], p_unseen_quad)

        fold_elapsed = time.time() - fold_start
        print(f"Fold {fold+1} FINISHED in {fold_elapsed:.1f}s | Honest Seen Acc={acc_seen_fold:.5f}, LL={ll_seen_fold:.5f} | Unseen Acc={acc_unseen_fold:.5f}, LL={ll_unseen_fold:.5f}", flush=True)

        fold_reports.append({
            "fold": fold + 1,
            "best_iter_cb": int(best_iter_cb),
            "best_iter_lgb": int(best_iter_lgb),
            "best_iter_xgb": int(best_iter_xgb),
            "weights": [float(w) for w in w_inner],
            "honest_seen_accuracy": float(acc_seen_fold),
            "honest_seen_logloss": float(ll_seen_fold),
            "honest_unseen_accuracy": float(acc_unseen_fold),
            "honest_unseen_logloss": float(ll_unseen_fold),
            "elapsed_seconds": float(fold_elapsed),
        })

        del cb_outer, lgb_outer, xgb_outer, lr_outer, X_fit_dense, X_seen_dense, X_unseen_dense, X_test_dense, X_fit_tfidf, X_seen_tfidf, X_unseen_tfidf, X_test_tfidf
        gc.collect()

    # 4. Global Out-of-Fold Metrics & Verification
    print(f"\n{'='*80}\n[4/5] Computing Global Honest Dual-Regime Metrics\n{'='*80}", flush=True)

    # A. Unseen Sellers (70.000 items)
    acc_unseen_global = accuracy_score(y_train, (oof_unseen_p_quad >= 0.5).astype(int))
    ll_unseen_global = log_loss(y_train, oof_unseen_p_quad)
    auc_unseen_global = roc_auc_score(y_train, oof_unseen_p_quad)

    # B. Seen Sellers Holdout (32.092 items, 100% out-of-fold!)
    seen_val_indices_arr = np.array(seen_val_indices_all)
    seen_y_arr = y_train[seen_val_indices_arr]
    seen_p_quad_arr = np.array(seen_val_p_quad_all)

    acc_seen_honest = accuracy_score(seen_y_arr, (seen_p_quad_arr >= 0.5).astype(int))
    ll_seen_honest = log_loss(seen_y_arr, seen_p_quad_arr)
    auc_seen_honest = roc_auc_score(seen_y_arr, seen_p_quad_arr)

    # C. Realistic Mix Sensitivity (71.62% seen / 28.38% unseen)
    acc_mix_honest = 0.7162 * acc_seen_honest + 0.2838 * acc_unseen_global

    print(f"HONEST RÉGIMEN UNSEEN-SELLERS (100% outer holdout, 70k items):")
    print(f"  Acc = {acc_unseen_global:.5f} ({int(acc_unseen_global*n_train)}/70000), LogLoss = {ll_unseen_global:.5f}, ROC-AUC = {auc_unseen_global:.5f}")
    print(f"HONEST RÉGIMEN SEEN-SELLERS (100% out-of-fold holdout, {len(seen_y_arr)} items, fit ∩ seen_val = ∅):")
    print(f"  Acc = {acc_seen_honest:.5f} ({int(acc_seen_honest*len(seen_y_arr))}/{len(seen_y_arr)}), LogLoss = {ll_seen_honest:.5f}, ROC-AUC = {auc_seen_honest:.5f}")
    print(f"HONEST REALISTIC MIX TEST SENSITIVITY (71.62% seen / 28.38% unseen):")
    print(f"  Acc = {acc_mix_honest:.5f}")

    # Paired Item-by-Item Comparison vs Strong Baseline
    hit_cand = ((oof_unseen_p_quad >= 0.5).astype(int) == y_train)
    hit_base = (pred_baseline == y_train)
    diff = hit_cand.astype(int) - hit_base.astype(int)

    fixes = int(np.sum(~hit_base & hit_cand))
    breaks = int(np.sum(hit_base & ~hit_cand))
    net_hits = fixes - breaks
    delta_acc = float(acc_unseen_global - acc_baseline)
    delta_ll = float(ll_unseen_global - ll_baseline)

    print("\nPAIRED ITEM-BY-ITEM COMPARISON VS STRONG BASELINE (Podium Engine):")
    print(f"  Candidate Unseen Acc: {acc_unseen_global:.5f} vs Baseline {acc_baseline:.5f}")
    print(f"  Paired Delta Acc:     {delta_acc:+.5f} (Net: {net_hits:+d} | Fixes: {fixes}, Breaks: {breaks})")
    print(f"  Paired Delta LL:      {delta_ll:+.5f}")

    # Paired Bootstrap by Seller Cluster (1000 resamples)
    print("\nComputing Seller-Clustered Bootstrap CI (1000 resamples)...", flush=True)
    boot_seller_mean, boot_s_lower, boot_s_upper, boot_s_p = clustered_bootstrap_ci(
        diff, sellers_train, n_boot=1000, seed=20261110
    )
    print(f"  Seller-Clustered Bootstrap 95% CI: [{boot_s_lower:+.5f}, {boot_s_upper:+.5f}] (Mean: {boot_seller_mean:+.5f})")

    # Paired Bootstrap by Category Cluster (1000 resamples)
    print("Computing Category-Clustered Bootstrap CI (1000 resamples)...", flush=True)
    boot_cat_mean, boot_c_lower, boot_c_upper, boot_c_p = clustered_bootstrap_ci(
        diff, categories_train, n_boot=1000, seed=20261110
    )
    print(f"  Category-Clustered Bootstrap 95% CI: [{boot_c_lower:+.5f}, {boot_c_upper:+.5f}] (Mean: {boot_cat_mean:+.5f})")

    # Evaluate the two pre-declared hypotheses w=0.10 and w=0.35 (CX-0144 / CX-0146)
    print("\nEVALUATING PRE-DECLARED HYPOTHESES (w=0.10 and w=0.35) VS BASELINE:", flush=True)
    predeclared_results = {}
    for w_val in [0.10, 0.35]:
        p_b = (1 - w_val) * p_baseline + w_val * oof_unseen_p_quad
        pred_b = (p_b >= 0.5).astype(int)
        acc_b = accuracy_score(y_train, pred_b)
        ll_b = log_loss(y_train, p_b)

        hit_b = (pred_b == y_train)
        diff_b = hit_b.astype(int) - hit_base.astype(int)
        fixes_b = int(np.sum(~hit_base & hit_b))
        breaks_b = int(np.sum(hit_base & ~hit_b))
        net_b = fixes_b - breaks_b

        _, s_lo, s_hi, s_p = clustered_bootstrap_ci(diff_b, sellers_train, n_boot=1000, seed=20261110)
        _, c_lo, c_hi, c_p = clustered_bootstrap_ci(diff_b, categories_train, n_boot=1000, seed=20261110)

        print(f"Hypothesis w={w_val:.2f}: Acc={acc_b:.5f} (Net: {net_b:+d} | Fixes: {fixes_b}, Breaks: {breaks_b}), LL={ll_b:.5f}")
        print(f"  Seller-Cluster 95% CI:   [{s_lo:+.5f}, {s_hi:+.5f}] | P(Delta>0)={s_p:.4f}")
        print(f"  Category-Cluster 95% CI: [{c_lo:+.5f}, {c_hi:+.5f}] | P(Delta>0)={c_p:.4f}")

        predeclared_results[f"w_{w_val}"] = {
            "weight": w_val,
            "accuracy": float(acc_b),
            "logloss": float(ll_b),
            "fixes": fixes_b,
            "breaks": breaks_b,
            "net_hits": net_b,
            "seller_clustered_95ci": [s_lo, s_hi],
            "category_clustered_95ci": [c_lo, c_hi],
            "probability_delta_gt_zero": s_p,
        }

    # 5. Save Artifacts
    print(f"\n[5/5] Saving Canonical Nested OOF, Test Predictions, and JSON Report...", flush=True)

    oof_unseen_df = pd.DataFrame({
        "id": train_ids,
        "seller_id": sellers_train,
        "category_id": categories_train,
        "y_true": y_train,
        "p_pred_cb": oof_unseen_p_cb,
        "p_pred_lgb": oof_unseen_p_lgb,
        "p_pred_xgb": oof_unseen_p_xgb,
        "p_pred_text": oof_unseen_p_text,
        "p_pred_quad": oof_unseen_p_quad,
        "outer_unseen_fold": df_splits["outer_unseen_fold"].values,
    })
    oof_csv_path = out_dir / "oof_canonical_nested_dual_regime.csv"
    oof_unseen_df.to_csv(oof_csv_path, index=False)
    oof_hash = compute_sha256(oof_csv_path)
    print(f"Saved Unseen OOF DataFrame: {oof_csv_path} | SHA-256: {oof_hash}")

    seen_df = pd.DataFrame({
        "id": train_ids[seen_val_indices_arr],
        "seller_id": sellers_train[seen_val_indices_arr],
        "category_id": categories_train[seen_val_indices_arr],
        "y_true": seen_y_arr,
        "p_pred_quad": seen_p_quad_arr,
    })
    seen_csv_path = out_dir / "seen_val_honest_oof.csv"
    seen_df.to_csv(seen_csv_path, index=False)
    seen_hash = compute_sha256(seen_csv_path)
    print(f"Saved Honest Seen Val OOF DataFrame: {seen_csv_path} | SHA-256: {seen_hash}")

    test_df = pd.DataFrame({
        "id": test_ids,
        "seller_id": test_cats["seller_id"],
        "category_id": test_cats["category_id"],
        "p_used_cb": test_p_cb,
        "p_used_lgb": test_p_lgb,
        "p_used_xgb": test_p_xgb,
        "p_used_text": test_p_text,
        "p_used_quad": test_p_quad,
    })
    test_csv_path = out_dir / "test_probabilities_canonical_nested.csv"
    test_df.to_csv(test_csv_path, index=False)
    test_hash = compute_sha256(test_csv_path)
    print(f"Saved Test Probabilities: {test_csv_path} | SHA-256: {test_hash}")

    # Summary Report
    summary = {
        "title": "Canonical Nested Dual-Regime Training & Paired Baseline Verification Report",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits_manifest": "data/splits/canonical_nested_dual_regime_splits_manifest.json",
        "splits_csv_sha256": compute_sha256(splits_csv_path),
        "total_train_rows": n_train,
        "total_test_rows": n_test,
        "outer_splits": n_splits,
        "models_trained": [
            "CatBoost Depth 8 (Agent 2)",
            "LightGBM 127 Leaves (Agent 3)",
            "Asymmetric Lossguide XGBoost (Agent 3)",
            "Strong Multi-Channel Text Logistic SAGA (Agent 4)",
            "Seller TE LOO & Category Stats (Agent 5)",
            "Convex Multi-Model Meta-Ensemble (Agent 6)",
        ],
        "baseline_comparison": {
            "baseline_name": "Podium Engine (results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv)",
            "baseline_accuracy": float(acc_baseline),
            "baseline_logloss": float(ll_baseline),
            "baseline_hits": int(acc_baseline * n_train),
        },
        "honest_unseen_seller_regime": {
            "sample_size": n_train,
            "accuracy": float(acc_unseen_global),
            "logloss": float(ll_unseen_global),
            "roc_auc": float(auc_unseen_global),
            "fixes_vs_baseline": fixes,
            "breaks_vs_baseline": breaks,
            "net_hits_vs_baseline": net_hits,
            "seller_clustered_95ci": [boot_s_lower, boot_s_upper],
            "category_clustered_95ci": [boot_c_lower, boot_c_upper],
        },
        "honest_seen_seller_regime": {
            "sample_size": len(seen_y_arr),
            "accuracy": float(acc_seen_honest),
            "logloss": float(ll_seen_honest),
            "roc_auc": float(auc_seen_honest),
            "audit_check": "fit_set ∩ seen_val = ∅ (True Holdout, Zero In-Sample Reuse)",
        },
        "honest_mix_sensitivity": {
            "mix_formula": "0.7162 * honest_seen_acc + 0.2838 * honest_unseen_acc",
            "accuracy": float(acc_mix_honest),
        },
        "predeclared_hypotheses": predeclared_results,
        "fold_details": fold_reports,
        "artifacts_sha256": {
            "oof_unseen_csv": oof_hash,
            "seen_val_honest_csv": seen_hash,
            "test_probabilities_csv": test_hash,
        },
        "total_elapsed_seconds": float(time.time() - start_total),
    }

    report_json_path = out_dir / "paired_canonical_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_hash = compute_sha256(report_json_path)
    print(f"Saved JSON Report: {report_json_path} | SHA-256: {report_hash}")
    print(f"\nCOMPLETED CANONICAL NESTED VERIFICATION in {(time.time() - start_total)/60:.2f} minutes!")


if __name__ == "__main__":
    main()
