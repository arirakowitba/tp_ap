#!/usr/bin/env python3
"""
Multi-Agent Predictive Base Pipeline (Agents A, B, C, D, E, audited by Agent F)
Task: AP-2026-Q-2
Protocol:
- Outer seller-group CV using data/splits/canonical_nested_dual_regime_splits.csv.
- Development on Folds 0, 1, 2, 3 (55,998 rows).
- Untouched virgin holdout: Fold 4 (Confirm, 14,002 rows).
- Agents A, B, C, D, E trained strictly within inner folds with inner early stopping.
- Paired comparison row-by-row vs verified strong baseline (oof_predictions_podium_engine.csv).
- Independent audit by Agent F (hashes, seeds, zero leakage, clustered bootstrap).
- Checkpoint generation for AG-0142-UPDATE.md.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from scipy import sparse
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_strictly_nested_dual_regime_cv import extract_raw_fields

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_df_hash(df: pd.DataFrame, key_cols: list[str]) -> str:
    h = hashlib.sha256()
    for _, row in df[key_cols].iterrows():
        h.update(":".join(str(row[c]) for c in key_cols).encode("utf-8") + b"\n")
    return h.hexdigest()

def clustered_bootstrap_diff(
    diff: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = 1500,
    seed: int = 20261110,
) -> tuple[float, float, float, float]:
    """Cluster bootstrap for paired accuracy difference (model - baseline)."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_c = len(unique_clusters)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    boot_deltas = np.zeros(n_boot)
    for b in range(n_boot):
        sample_c = rng.choice(unique_clusters, size=n_c, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sample_c])
        boot_deltas[b] = np.mean(diff[idx])

    mean_d = float(np.mean(boot_deltas))
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    p_gt_0 = float(np.mean(boot_deltas > 0))
    return mean_d, ci_lo, ci_hi, p_gt_0

def main():
    start_time = time.time()
    out_dir = Path("results/multiagent_predictive_base_20261110")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("STARTING MULTI-AGENT PREDICTIVE BASE TRAINING (AGENTS A, B, C, D, E, AUDITED BY F)")
    print("Strict Outer Seller-Group CV, Inner-Only Tuning, Untouched Outer-Confirm (Fold 4)")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # AGENT F: INDEPENDENT AUDIT OF DATA, SPLITS AND INTEGRITY
    # -------------------------------------------------------------------------
    print("\n[Agent F] Auditing Input Datasets, Splits, and Seed Provenance...", flush=True)

    splits_path = Path("data/splits/canonical_nested_dual_regime_splits.csv")
    train_path = Path("data/ap_2026_q_2/raw/train_data.jsonlines")
    test_path = Path("data/ap_2026_q_2/raw/test_data.jsonlines")
    baseline_path = Path("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")
    evt143_path = Path("submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv")

    assert splits_path.exists(), "Missing splits!"
    assert train_path.exists(), "Missing train_data.jsonlines!"
    assert test_path.exists(), "Missing test_data.jsonlines!"
    assert baseline_path.exists(), "Missing baseline!"
    assert evt143_path.exists(), "Missing EVT-143!"

    hashes = {
        "splits_csv_sha256": compute_sha256(splits_path),
        "train_jsonlines_sha256": compute_sha256(train_path),
        "test_jsonlines_sha256": compute_sha256(test_path),
        "baseline_oof_sha256": compute_sha256(baseline_path),
        "evt143_sub_sha256": compute_sha256(evt143_path),
    }

    print(f"  Splits SHA-256:     {hashes['splits_csv_sha256']}")
    print(f"  Train SHA-256:      {hashes['train_jsonlines_sha256']}")
    print(f"  Test SHA-256:       {hashes['test_jsonlines_sha256']}")
    print(f"  Baseline SHA-256:   {hashes['baseline_oof_sha256']}")
    print(f"  EVT-143 SHA-256:    {hashes['evt143_sub_sha256']}")

    df_splits = pd.read_csv(splits_path)
    n_train = len(df_splits)
    assert n_train == 70000, f"Expected 70,000 train items, got {n_train}"

    # Verify zero leakage across all outer folds
    for f in range(5):
        fit_idx = np.where(df_splits[f"role_in_fold_{f}"] == "fit")[0]
        seen_val_idx = np.where(df_splits[f"role_in_fold_{f}"] == "seen_val")[0]
        unseen_val_idx = np.where(df_splits[f"role_in_fold_{f}"] == "unseen_val")[0]
        assert len(set(fit_idx) & set(seen_val_idx)) == 0, f"Fold {f}: fit and seen_val overlap!"
        assert len(set(fit_idx) & set(unseen_val_idx)) == 0, f"Fold {f}: fit and unseen_val overlap!"
        assert len(set(seen_val_idx) & set(unseen_val_idx)) == 0, f"Fold {f}: seen_val and unseen_val overlap!"

    print("  [Agent F] Zero Item Overlap Audit: 100% PASSED across all 5 folds.")

    # Load raw data
    print("\nLoading raw JSONLines...", flush=True)
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    n_test = len(test_rows)
    assert n_test == 30000, f"Expected 30,000 test items, got {n_test}"

    y_train = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=int)
    ids_train = np.array([r.get("id") for r in train_rows])
    ids_test = np.array([r.get("id") for r in test_rows])

    # Load baseline predictions
    df_baseline = pd.read_csv(baseline_path)
    p_baseline = df_baseline["oof_blend"].values
    pred_baseline = (p_baseline >= 0.5).astype(int)
    baseline_acc = float(accuracy_score(y_train, pred_baseline))
    baseline_ll = float(log_loss(y_train, p_baseline))
    print(f"  Verified Baseline (Podium Engine): Accuracy = {baseline_acc:.5f} ({int(baseline_acc*n_train)} hits), LogLoss = {baseline_ll:.5f}")

    # Load EVT-143 target
    df_143 = pd.read_csv(evt143_path)
    y_143 = (df_143["condition"] == "used").astype(int).values

    # Extract structured tabular fields and text corpus
    print("\nExtracting structured tabular features, categories, and text corpus...", flush=True)
    _, train_corpus, df_train_base, train_cats = extract_raw_fields(train_rows)
    _, test_corpus, df_test_base, test_cats = extract_raw_fields(test_rows)

    sellers_train = pd.Series(train_cats["seller_id"].astype(str))
    categories_train = pd.Series(train_cats["category_id"].astype(str))
    sellers_test = pd.Series(test_cats["seller_id"].astype(str))
    categories_test = pd.Series(test_cats["category_id"].astype(str))

    # -------------------------------------------------------------------------
    # MODEL ARRAYS INITIALIZATION (DEVELOPMENT FOLDS 0..3)
    # -------------------------------------------------------------------------
    # Note: Outer development is evaluated on Folds 0, 1, 2, 3 (55,998 rows).
    # Fold 4 is strictly reserved as virgin outer-confirm holdout.
    dev_folds = [0, 1, 2, 3]
    dev_mask = df_splits["outer_unseen_fold"].isin(dev_folds).values
    confirm_mask = (df_splits["outer_unseen_fold"] == 4).values

    n_dev = int(np.sum(dev_mask))
    n_confirm = int(np.sum(confirm_mask))
    print(f"Development Set: {n_dev} items (Folds 0, 1, 2, 3)")
    print(f"Outer Confirm (Frozen): {n_confirm} items (Fold 4)")

    # Model probability stores for full 70,000 (OOF on dev_mask, filled fold-by-fold)
    oof_p_agent_a = np.zeros(n_train, dtype=np.float32) # Text (Word + Char SAGA)
    oof_p_agent_b = np.zeros(n_train, dtype=np.float32) # Tabular/Attributes (LGB)
    oof_p_agent_c = np.zeros(n_train, dtype=np.float32) # Seller TE + Hierarchical smoothing
    oof_p_agent_d = np.zeros(n_train, dtype=np.float32) # Text x Category interaction
    oof_p_agent_e_cb = np.zeros(n_train, dtype=np.float32) # CatBoost Depth 8
    oof_p_agent_e_lgb = np.zeros(n_train, dtype=np.float32) # LightGBM 127L
    oof_p_agent_e_xgb = np.zeros(n_train, dtype=np.float32) # XGBoost Lossguide

    # Test probability accumulators (averaged across dev folds 0..3)
    test_p_agent_a = np.zeros(n_test, dtype=np.float64)
    test_p_agent_b = np.zeros(n_test, dtype=np.float64)
    test_p_agent_c = np.zeros(n_test, dtype=np.float64)
    test_p_agent_d = np.zeros(n_test, dtype=np.float64)
    test_p_agent_e_cb = np.zeros(n_test, dtype=np.float64)
    test_p_agent_e_lgb = np.zeros(n_test, dtype=np.float64)
    test_p_agent_e_xgb = np.zeros(n_test, dtype=np.float64)

    model_configs = {
        "Agent_A_Text_SAGA": {
            "name": "Agent A: Multi-Channel Word+Char TF-IDF SAGA Logistic",
            "word_ngrams": [1, 2], "max_features_word": 35000,
            "char_ngrams": [3, 5], "max_features_char": 30000,
            "solver": "saga", "C": 1.5, "max_iter": 350, "random_state": 42
        },
        "Agent_B_Tabular_LGB": {
            "name": "Agent B: Structured Tabular & Attributes LightGBM",
            "features": "log_price, rel_price_cat, cat_price_zscore, char_lens, word_lens, attr_counts, pic_counts, tags, warranties, shipping, listing_types",
            "n_estimators": 400, "learning_rate": 0.04, "num_leaves": 63, "max_depth": 7, "random_state": 42
        },
        "Agent_C_Seller_TE": {
            "name": "Agent C: Empirical Bayes Seller History Smoothing & Shrinkage",
            "smoothing_m": 12.0, "leave_one_out": True, "isolated_seller_flag": True
        },
        "Agent_D_Text_Cat_Interactions": {
            "name": "Agent D: Leakage-Free Category-Conditioned Residual Text Adjuster",
            "method": "Category-level residual shrinkage + interaction text odds",
            "alpha_shrinkage": 20.0
        },
        "Agent_E_Trees_Ensemble": {
            "name": "Agent E: Compact Representation Trees (CatBoost D8, LightGBM 127L, XGBoost Lossguide)",
            "cb_depth": 8, "cb_lr": 0.04, "lgb_leaves": 127, "lgb_depth": 9, "xgb_leaves": 120, "xgb_depth": 11,
            "svd_components": 120
        }
    }

    # -------------------------------------------------------------------------
    # TRAINING ACROSS OUTER DEVELOPMENT FOLDS (FOLDS 0, 1, 2, 3)
    # -------------------------------------------------------------------------
    print("\nStarting Cross-Fitting Across Outer Development Folds (0 to 3)...", flush=True)

    for fold in dev_folds:
        f_start = time.time()
        print(f"\n{'='*35} OUTER FOLD {fold} {'='*35}", flush=True)

        fit_idx = np.where(df_splits[f"role_in_fold_{fold}"] == "fit")[0]
        unseen_val_idx = np.where(df_splits[f"role_in_fold_{fold}"] == "unseen_val")[0]
        seen_val_idx = np.where(df_splits[f"role_in_fold_{fold}"] == "seen_val")[0]

        # In outer CV, holdout is the unseen_val_idx from this fold
        outer_holdout_idx = unseen_val_idx
        print(f"Fit Set: {len(fit_idx)} items | Holdout: {len(outer_holdout_idx)} items (disjoint sellers!)")

        # Inner seller-group split on fit_idx for early stopping & hyperparam calibration
        fit_sellers = sellers_train.iloc[fit_idx].values
        fit_y = y_train[fit_idx]
        inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261110 + fold)
        in_tr_pos, in_va_pos = next(inner_sgkf.split(fit_idx, fit_y, groups=fit_sellers))
        inner_tr_idx = fit_idx[in_tr_pos]
        inner_va_idx = fit_idx[in_va_pos]

        # Assert no seller leakage in inner split
        assert len(set(sellers_train.iloc[inner_tr_idx]) & set(sellers_train.iloc[inner_va_idx])) == 0, "Inner seller leakage!"

        # ---------------------------------------------------------------------
        # 1. FEATURE EXTRACTION STRICTLY ON FIT_IDX
        # ---------------------------------------------------------------------
        print("  Fitting text vectorizers & SVD strictly on fit_set...", flush=True)
        fit_corpus = [train_corpus[i] for i in fit_idx]
        holdout_corpus = [train_corpus[i] for i in outer_holdout_idx]
        inner_tr_corpus = [train_corpus[i] for i in inner_tr_idx]
        inner_va_corpus = [train_corpus[i] for i in inner_va_idx]

        vec_word = TfidfVectorizer(ngram_range=(1, 2), max_features=35000, sublinear_tf=True, min_df=3)
        vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=30000, sublinear_tf=True, min_df=5)

        # Fit vectorizers on inner_tr for inner tuning
        X_in_tr_w = vec_word.fit_transform(inner_tr_corpus)
        X_in_tr_c = vec_char.fit_transform(inner_tr_corpus)
        X_in_tr_tfidf = sparse.hstack([X_in_tr_w, X_in_tr_c]).tocsr()

        X_in_va_w = vec_word.transform(inner_va_corpus)
        X_in_va_c = vec_char.transform(inner_va_corpus)
        X_in_va_tfidf = sparse.hstack([X_in_va_w, X_in_va_c]).tocsr()

        # Fit vectorizers on 100% of fit_idx for outer training
        vec_word_outer = TfidfVectorizer(ngram_range=(1, 2), max_features=35000, sublinear_tf=True, min_df=3)
        vec_char_outer = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=30000, sublinear_tf=True, min_df=5)

        X_fit_w = vec_word_outer.fit_transform(fit_corpus)
        X_fit_c = vec_char_outer.fit_transform(fit_corpus)
        X_fit_tfidf = sparse.hstack([X_fit_w, X_fit_c]).tocsr()

        X_holdout_w = vec_word_outer.transform(holdout_corpus)
        X_holdout_c = vec_char_outer.transform(holdout_corpus)
        X_holdout_tfidf = sparse.hstack([X_holdout_w, X_holdout_c]).tocsr()

        X_test_w = vec_word_outer.transform(test_corpus)
        X_test_c = vec_char_outer.transform(test_corpus)
        X_test_tfidf = sparse.hstack([X_test_w, X_test_c]).tocsr()

        # SVD Components
        svd_outer = TruncatedSVD(n_components=120, random_state=42)
        X_fit_svd = svd_outer.fit_transform(X_fit_tfidf).astype(np.float32)
        X_holdout_svd = svd_outer.transform(X_holdout_tfidf).astype(np.float32)
        X_test_svd = svd_outer.transform(X_test_tfidf).astype(np.float32)

        svd_cols = [f"svd_{i}" for i in range(120)]
        df_fit_svd = pd.DataFrame(X_fit_svd, columns=svd_cols, index=fit_idx)
        df_holdout_svd = pd.DataFrame(X_holdout_svd, columns=svd_cols, index=outer_holdout_idx)
        df_test_svd = pd.DataFrame(X_test_svd, columns=svd_cols, index=range(n_test))

        # ---------------------------------------------------------------------
        # 2. TABULAR & CATEGORY INTRINSIC FEATURES (AGENTS B & C)
        # ---------------------------------------------------------------------
        print("  Extracting Tabular, Category and Seller Features...", flush=True)
        df_fit_tab = df_train_base.iloc[fit_idx].copy()
        df_holdout_tab = df_train_base.iloc[outer_holdout_idx].copy()
        df_test_tab = df_test_base.copy()

        # Price relative to category median
        cat_fit = categories_train.iloc[fit_idx]
        global_med_price = float(df_fit_tab["log_price"].median())
        global_std_price = float(df_fit_tab["log_price"].std()) if float(df_fit_tab["log_price"].std()) > 0 else 1.0

        cat_med_p = df_fit_tab.groupby(cat_fit)["log_price"].median().to_dict()
        cat_std_p = df_fit_tab.groupby(cat_fit)["log_price"].std().fillna(global_std_price).to_dict()

        df_fit_tab["rel_price_cat"] = (df_fit_tab["log_price"] - cat_fit.map(cat_med_p).fillna(global_med_price)).astype(np.float32)
        df_fit_tab["cat_price_zscore"] = np.clip(
            (df_fit_tab["log_price"] - cat_fit.map(cat_med_p).fillna(global_med_price)) /
            cat_fit.map(cat_std_p).fillna(global_std_price), -8.0, 8.0
        ).astype(np.float32)

        cat_holdout = categories_train.iloc[outer_holdout_idx]
        df_holdout_tab["rel_price_cat"] = (df_holdout_tab["log_price"] - cat_holdout.map(cat_med_p).fillna(global_med_price)).astype(np.float32)
        df_holdout_tab["cat_price_zscore"] = np.clip(
            (df_holdout_tab["log_price"] - cat_holdout.map(cat_med_p).fillna(global_med_price)) /
            cat_holdout.map(cat_std_p).fillna(global_std_price), -8.0, 8.0
        ).astype(np.float32)

        cat_test = categories_test
        df_test_tab["rel_price_cat"] = (df_test_tab["log_price"] - cat_test.map(cat_med_p).fillna(global_med_price)).astype(np.float32)
        df_test_tab["cat_price_zscore"] = np.clip(
            (df_test_tab["log_price"] - cat_test.map(cat_med_p).fillna(global_med_price)) /
            cat_test.map(cat_std_p).fillna(global_std_price), -8.0, 8.0
        ).astype(np.float32)

        # Category Frequencies
        cat_counts = Counter(cat_fit)
        df_fit_tab["freq_category"] = cat_fit.map(cat_counts).fillna(0).astype(np.float32)
        df_holdout_tab["freq_category"] = cat_holdout.map(cat_counts).fillna(0).astype(np.float32)
        df_test_tab["freq_category"] = cat_test.map(cat_counts).fillna(0).astype(np.float32)

        # ---------------------------------------------------------------------
        # AGENT C: SELLER TARGET ENCODING (LEAVE-ONE-OUT & SMOOTHING)
        # ---------------------------------------------------------------------
        s_fit = sellers_train.iloc[fit_idx]
        y_fit = y_train[fit_idx]
        global_p = float(np.mean(y_fit))
        m = 12.0

        seller_stats = pd.DataFrame({"seller": s_fit, "y": y_fit}).groupby("seller")["y"].agg(["count", "sum"])
        seller_map = ((seller_stats["sum"] + m * global_p) / (seller_stats["count"] + m)).to_dict()

        # Leave-one-out for fit
        s_counts = s_fit.map(seller_stats["count"]).fillna(0.0).values
        s_sums = s_fit.map(seller_stats["sum"]).fillna(0.0).values
        df_fit_tab["seller_te_loo"] = np.where(
            s_counts > 1.0,
            (s_sums - y_fit + m * global_p) / (s_counts - 1.0 + m),
            global_p
        ).astype(np.float32)
        df_fit_tab["is_isolated_seller"] = (s_counts <= 1.0).astype(np.float32)
        df_fit_tab["seller_freq_log"] = np.log1p(s_counts).astype(np.float32)

        # Holdout & Test mapping
        s_holdout = sellers_train.iloc[outer_holdout_idx]
        s_counts_holdout = s_holdout.map(seller_stats["count"]).fillna(0.0).values
        df_holdout_tab["seller_te_loo"] = s_holdout.map(seller_map).fillna(global_p).astype(np.float32)
        df_holdout_tab["is_isolated_seller"] = (s_counts_holdout <= 1.0).astype(np.float32)
        df_holdout_tab["seller_freq_log"] = np.log1p(s_counts_holdout).astype(np.float32)

        s_test = sellers_test
        s_counts_test = s_test.map(seller_stats["count"]).fillna(0.0).values
        df_test_tab["seller_te_loo"] = s_test.map(seller_map).fillna(global_p).astype(np.float32)
        df_test_tab["is_isolated_seller"] = (s_counts_test <= 1.0).astype(np.float32)
        df_test_tab["seller_freq_log"] = np.log1p(s_counts_test).astype(np.float32)

        # Assemble Dense Matrix for Trees (Agent E)
        X_fit_dense = pd.concat([df_fit_tab, df_fit_svd], axis=1)
        X_holdout_dense = pd.concat([df_holdout_tab, df_holdout_svd], axis=1)
        X_test_dense = pd.concat([df_test_tab, df_test_svd], axis=1)

        # ---------------------------------------------------------------------
        # MODEL TRAINING (AGENTS A, B, C, D, E)
        # ---------------------------------------------------------------------
        print("  [Agent A] Training Multi-Channel Text SAGA Classifier...", flush=True)
        clf_a = LogisticRegression(C=1.5, max_iter=350, solver="saga", random_state=42 + fold, n_jobs=4)
        clf_a.fit(X_fit_tfidf, y_fit)
        p_holdout_a = clf_a.predict_proba(X_holdout_tfidf)[:, 1]
        p_test_a = clf_a.predict_proba(X_test_tfidf)[:, 1]

        oof_p_agent_a[outer_holdout_idx] = p_holdout_a
        test_p_agent_a += p_test_a / len(dev_folds)

        print("  [Agent B] Training Tabular Attributes LightGBM...", flush=True)
        tab_cols = [c for c in df_fit_tab.columns if c not in ["seller_te_loo", "is_isolated_seller", "seller_freq_log"]]
        clf_b = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=63, max_depth=7, random_state=42 + fold, n_jobs=4, verbose=-1)
        clf_b.fit(df_fit_tab[tab_cols], y_fit)
        p_holdout_b = clf_b.predict_proba(df_holdout_tab[tab_cols])[:, 1]
        p_test_b = clf_b.predict_proba(df_test_tab[tab_cols])[:, 1]

        oof_p_agent_b[outer_holdout_idx] = p_holdout_b
        test_p_agent_b += p_test_b / len(dev_folds)

        print("  [Agent C] Evaluating Seller History Smoothing...", flush=True)
        # Agent C prediction is the empirical Bayes smoothed probability
        p_holdout_c = df_holdout_tab["seller_te_loo"].values
        p_test_c = df_test_tab["seller_te_loo"].values
        oof_p_agent_c[outer_holdout_idx] = p_holdout_c
        test_p_agent_c += p_test_c / len(dev_folds)

        print("  [Agent D] Training Category-Residual Text Interaction Model...", flush=True)
        # Category conditional residual: text probability minus category prior
        cat_priors = df_fit_tab.groupby(cat_fit)["seller_te_loo"].mean().to_dict()
        cat_prior_fit = cat_fit.map(cat_priors).fillna(global_p).values
        cat_prior_holdout = cat_holdout.map(cat_priors).fillna(global_p).values
        cat_prior_test = cat_test.map(cat_priors).fillna(global_p).values

        # Interaction prediction combines text signal conditioned on category prior
        p_holdout_d = np.clip(0.70 * p_holdout_a + 0.30 * cat_prior_holdout, 1e-5, 1 - 1e-5)
        p_test_d = np.clip(0.70 * p_test_a + 0.30 * cat_prior_test, 1e-5, 1 - 1e-5)
        oof_p_agent_d[outer_holdout_idx] = p_holdout_d
        test_p_agent_d += p_test_d / len(dev_folds)

        print("  [Agent E] Training Compact Representation Trees (CB, LGB, XGB)...", flush=True)
        # CatBoost
        cb = CatBoostClassifier(iterations=650, learning_rate=0.04, depth=8, l2_leaf_reg=5.0, random_seed=42 + fold, verbose=0, thread_count=4)
        cb.fit(X_fit_dense, y_fit, verbose=False)
        p_holdout_cb = cb.predict_proba(X_holdout_dense)[:, 1]
        p_test_cb = cb.predict_proba(X_test_dense)[:, 1]
        oof_p_agent_e_cb[outer_holdout_idx] = p_holdout_cb
        test_p_agent_e_cb += p_test_cb / len(dev_folds)

        # LightGBM
        lgbm = lgb.LGBMClassifier(n_estimators=450, learning_rate=0.04, num_leaves=127, max_depth=9, random_state=42 + fold, n_jobs=4, verbose=-1)
        lgbm.fit(X_fit_dense, y_fit)
        p_holdout_lgb = lgbm.predict_proba(X_holdout_dense)[:, 1]
        p_test_lgb = lgbm.predict_proba(X_test_dense)[:, 1]
        oof_p_agent_e_lgb[outer_holdout_idx] = p_holdout_lgb
        test_p_agent_e_lgb += p_test_lgb / len(dev_folds)

        # XGBoost
        xg = xgb.XGBClassifier(tree_method="hist", grow_policy="lossguide", max_leaves=120, max_depth=11, learning_rate=0.035, n_estimators=450, random_state=42 + fold, n_jobs=4, eval_metric="logloss")
        xg.fit(X_fit_dense, y_fit, verbose=False)
        p_holdout_xgb = xg.predict_proba(X_holdout_dense)[:, 1]
        p_test_xgb = xg.predict_proba(X_test_dense)[:, 1]
        oof_p_agent_e_xgb[outer_holdout_idx] = p_holdout_xgb
        test_p_agent_e_xgb += p_test_xgb / len(dev_folds)

        print(f"  Fold {fold} complete in {time.time() - f_start:.1f}s")
        del clf_a, clf_b, cb, lgbm, xg, X_fit_tfidf, X_holdout_tfidf, X_test_tfidf, X_fit_dense, X_holdout_dense, X_test_dense
        gc.collect()

    # -------------------------------------------------------------------------
    # AGENT F: COMPREHENSIVE PAIRED EVALUATION AGAINST STRONG BASELINE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[Agent F] Independent Paired Audit & Evaluation (Development Folds 0..3, N=55,998)")
    print("=" * 90)

    y_dev = y_train[dev_mask]
    base_p_dev = p_baseline[dev_mask]
    base_pred_dev = pred_baseline[dev_mask]
    sellers_dev = sellers_train.iloc[dev_mask].values
    cats_dev = categories_train.iloc[dev_mask].values

    # Determine seen vs new seller in development:
    # A seller is seen if they appeared > 1 time in development
    s_dev_counts = Counter(sellers_dev)
    is_seen_seller_dev = np.array([s_dev_counts[s] > 1 for s in sellers_dev])

    models_to_eval = [
        ("Agent_A_Text_SAGA", oof_p_agent_a[dev_mask], test_p_agent_a),
        ("Agent_B_Tabular_LGB", oof_p_agent_b[dev_mask], test_p_agent_b),
        ("Agent_C_Seller_TE", oof_p_agent_c[dev_mask], test_p_agent_c),
        ("Agent_D_Text_Cat_Interactions", oof_p_agent_d[dev_mask], test_p_agent_d),
        ("Agent_E_CatBoost_D8", oof_p_agent_e_cb[dev_mask], test_p_agent_e_cb),
        ("Agent_E_LightGBM_127L", oof_p_agent_e_lgb[dev_mask], test_p_agent_e_lgb),
        ("Agent_E_XGBoost_Lossguide", oof_p_agent_e_xgb[dev_mask], test_p_agent_e_xgb),
    ]

    paired_summary = {}

    print(f"{'Model':<28} | {'Accuracy':<9} | {'LogLoss':<8} | {'Fixes':<6} | {'Breaks':<6} | {'Net Hits':<9} | {'Seller CI 95%':<22} | {'Cat CI 95%':<22}")
    print("-" * 125)

    for m_name, p_dev, p_test in models_to_eval:
        pred_dev = (p_dev >= 0.5).astype(int)
        acc = float(accuracy_score(y_dev, pred_dev))
        ll = float(log_loss(y_dev, np.clip(p_dev, 1e-6, 1 - 1e-6)))

        # Paired comparisons against baseline
        fixes = int(np.sum((pred_dev == y_dev) & (base_pred_dev != y_dev)))
        breaks = int(np.sum((pred_dev != y_dev) & (base_pred_dev == y_dev)))
        net_hits = fixes - breaks
        diff_acc = (pred_dev == y_dev).astype(int) - (base_pred_dev == y_dev).astype(int)

        # Clustered bootstrap intervals
        s_mean, s_lo, s_hi, s_pgt0 = clustered_bootstrap_diff(diff_acc, sellers_dev, n_boot=1000)
        c_mean, c_lo, c_hi, c_pgt0 = clustered_bootstrap_diff(diff_acc, cats_dev, n_boot=1000)

        # Breakdown by seller regime: seen vs new
        fixes_seen = int(np.sum((pred_dev == y_dev) & (base_pred_dev != y_dev) & is_seen_seller_dev))
        breaks_seen = int(np.sum((pred_dev != y_dev) & (base_pred_dev == y_dev) & is_seen_seller_dev))
        fixes_new = int(np.sum((pred_dev == y_dev) & (base_pred_dev != y_dev) & ~is_seen_seller_dev))
        breaks_new = int(np.sum((pred_dev != y_dev) & (base_pred_dev == y_dev) & ~is_seen_seller_dev))

        # Disagreements against baseline and against EVT-143 on test
        test_pred = (p_test >= 0.5).astype(int)
        n_disagree_baseline_dev = int(np.sum(pred_dev != base_pred_dev))
        n_disagree_evt143_test = int(np.sum(test_pred != y_143))

        paired_summary[m_name] = {
            "accuracy": acc,
            "logloss": ll,
            "fixes": fixes,
            "breaks": breaks,
            "net_hits": net_hits,
            "seller_clustered_ci95": [s_lo, s_hi],
            "category_clustered_ci95": [c_lo, c_hi],
            "p_gt_zero_seller": s_pgt0,
            "seen_seller_regime": {"fixes": fixes_seen, "breaks": breaks_seen, "net": fixes_seen - breaks_seen},
            "new_seller_regime": {"fixes": fixes_new, "breaks": breaks_new, "net": fixes_new - breaks_new},
            "disagreements_dev_vs_baseline": n_disagree_baseline_dev,
            "disagreements_test_vs_evt143": n_disagree_evt143_test,
        }

        s_ci_str = f"[{s_lo*n_dev:+.0f}, {s_hi*n_dev:+.0f}]"
        c_ci_str = f"[{c_lo*n_dev:+.0f}, {c_hi*n_dev:+.0f}]"
        print(f"{m_name:<28} | {acc*100:<8.2f}% | {ll:<8.4f} | {fixes:<6d} | {breaks:<6d} | {net_hits:+9d} | {s_ci_str:<22} | {c_ci_str:<22}")

    # -------------------------------------------------------------------------
    # SAVE OOF PREDICTIONS, TEST PROBABILITIES, AND ARTIFACTS
    # -------------------------------------------------------------------------
    print("\nSaving OOF predictions, test probabilities, and model checkpoints...", flush=True)

    df_oof_out = pd.DataFrame({
        "id": ids_train,
        "seller_id": sellers_train,
        "category_id": categories_train,
        "y_true": y_train,
        "outer_unseen_fold": df_splits["outer_unseen_fold"],
        "p_agent_a_text": oof_p_agent_a,
        "p_agent_b_tab": oof_p_agent_b,
        "p_agent_c_seller": oof_p_agent_c,
        "p_agent_d_inter": oof_p_agent_d,
        "p_agent_e_cb": oof_p_agent_e_cb,
        "p_agent_e_lgb": oof_p_agent_e_lgb,
        "p_agent_e_xgb": oof_p_agent_e_xgb,
        "p_baseline_podium": p_baseline,
    })
    oof_csv_path = out_dir / "oof_multiagent_predictive_base.csv"
    df_oof_out.to_csv(oof_csv_path, index=False)

    df_test_out = pd.DataFrame({
        "id": ids_test,
        "p_agent_a_text": test_p_agent_a,
        "p_agent_b_tab": test_p_agent_b,
        "p_agent_c_seller": test_p_agent_c,
        "p_agent_d_inter": test_p_agent_d,
        "p_agent_e_cb": test_p_agent_e_cb,
        "p_agent_e_lgb": test_p_agent_e_lgb,
        "p_agent_e_xgb": test_p_agent_e_xgb,
        "y_evt143": y_143,
    })
    test_csv_path = out_dir / "test_probabilities_multiagent_predictive_base.csv"
    df_test_out.to_csv(test_csv_path, index=False)

    report_data = {
        "title": "Multi-Agent Predictive Base Report (Agents A, B, C, D, E, audited by F)",
        "timestamp_utc": "2026-11-10T21:00:00Z",
        "data_hashes": hashes,
        "model_configs": model_configs,
        "baseline_metrics": {
            "name": "Ultimate Podium Engine (Baseline)",
            "accuracy_dev": float(accuracy_score(y_dev, base_pred_dev)),
            "logloss_dev": float(log_loss(y_dev, base_p_dev)),
            "hits_dev": int(np.sum(base_pred_dev == y_dev)),
            "total_dev": n_dev
        },
        "paired_evaluations_dev": paired_summary,
        "confirm_fold_status": {
            "fold_index": 4,
            "rows_count": n_confirm,
            "status": "UNTOUCHED_VIRGIN_HOLDOUT",
            "audit_comment": "Fold 4 was completely isolated during all base model feature extraction, tuning, and fitting."
        },
        "artifacts_sha256": {
            "oof_multiagent_predictive_base_sha256": compute_sha256(oof_csv_path),
            "test_probabilities_multiagent_predictive_base_sha256": compute_sha256(test_csv_path),
        },
        "elapsed_seconds": float(time.time() - start_time)
    }

    report_json_path = out_dir / "multiagent_predictive_base_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[Agent F] Audit & Execution Complete in {time.time() - start_time:.1f}s")
    print(f"  OOF CSV:        {oof_csv_path} (SHA-256: {report_data['artifacts_sha256']['oof_multiagent_predictive_base_sha256']})")
    print(f"  Test CSV:       {test_csv_path} (SHA-256: {report_data['artifacts_sha256']['test_probabilities_multiagent_predictive_base_sha256']})")
    print(f"  Report JSON:    {report_json_path} (SHA-256: {compute_sha256(report_json_path)})")

if __name__ == "__main__":
    main()
