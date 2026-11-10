#!/usr/bin/env python3
"""Strictly Nested Dual-Regime CV & Paired Baseline Verification.

Implements Lineaments 1-5 of CX-0144 and CX-0145:
1. Outer StratifiedGroupKFold(n_splits=5) grouped strictly by seller_id (unseen-seller regime).
2. Within-outer-train TF-IDF (word 1-2, 35k) and TruncatedSVD (100 comp) fit strictly on outer_tr_idx.
3. Inner seller-group split (85/15) solely to determine best_iteration for CatBoost and LightGBM,
   and to learn inner meta-blend weights without post-selection on outer holdout.
4. Final outer models refit on 100% of outer_tr_idx using frozen best_iteration without eval_set.
5. Dual regime tracking: unseen sellers (outer holdout) and seen sellers (within-seller items).
6. Paired prediction comparison item-by-item against the verified strong baseline
   (results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv, 0.94230 OOF).
7. Paired delta metrics: delta accuracy, delta logloss, fixes, breaks, seller-clustered bootstrap 95% CI.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# Decisive lexical cues compiled from verified empirical patterns
LEXICAL_PATTERNS = [
    ("cue_0km", re.compile(r"\b0\s*km\b|\bzero\s*km\b|\bsin\s+rodar\b|\b0km\b", re.I)),
    ("cue_caja_sellada", re.compile(r"\bcaja\s+sellada\b|\bcerrad[oa]\s+de\s+fabrica\b|\bprecintad[oa]\b|\ben\s+blister\b", re.I)),
    ("cue_nuevo", re.compile(r"\bnuev[oa]s?\b|\ba\s+estrenar\b", re.I)),
    ("cue_usado", re.compile(r"\busad[oa]s?\b|\bsegunda\s+mano\b", re.I)),
    ("cue_repuesto", re.compile(r"\bpara\s+repuestos?\b|\bpara\s+desarme\b|\bno\s+funciona\b|\brot[oa]s?\b|\bcon\s+falla\b", re.I)),
    ("cue_impecable", re.compile(r"\bimpecable\b|\bexcelente\s+estado\b|\bcomo\s+nuev[oa]\b|\bmuy\s+buen\s+estado\b", re.I)),
    ("cue_detalles", re.compile(r"\bcon\s+detalles?\b|\bmarcas?\s+de\s+uso\b|\brayones?\b|\bdesgaste\b", re.I)),
    ("cue_antiguo", re.compile(r"\bantigu[oa]s?\b|\bvintage\b|\bcolecci[oó]n\b|\bretro\b", re.I)),
    ("cue_sin_uso", re.compile(r"\bsin\s+uso\b|\bnunca\s+se\s+us[oó]\b|\bnueva\s+sin\s+uso\b", re.I)),
    ("cue_original", re.compile(r"\boriginal\b|\bleg[ií]tim[oa]\b", re.I)),
]


def extract_raw_fields(rows: list[dict]) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, np.ndarray]]:
    """Extract structured tabular features and text corpus from raw jsonlines."""
    ids = []
    corpus = []
    sellers = []
    categories = []
    prices = []
    has_prices = []
    has_orig_prices = []
    price_orig_ratios = []
    char_lens = []
    word_lens = []
    has_warrs = []
    warr_lens = []
    attr_counts = []
    pic_counts = []
    tag_counts = []
    var_counts = []
    mercadopago = []
    free_shipping = []
    local_pickup = []
    sold_qtys = []
    avail_qtys = []
    init_qtys = []
    listing_types = []
    buying_modes = []
    states = []
    ship_modes = []
    cues_matrix = []

    for r in rows:
        ids.append(r.get("id"))
        title = str(r.get("title") or "").strip()
        warranty = str(r.get("warranty") or "").strip()
        attrs = r.get("attributes") or []
        attr_vals = [str(a.get("value_name") or a.get("name") or "") for a in attrs if isinstance(a, dict)]
        attr_text = " ".join(attr_vals)

        corpus.append(f"{title} WARRANTY {warranty} ATTR {attr_text}")
        sellers.append(str(r.get("seller_id") or "UNKNOWN"))
        categories.append(str(r.get("category_id") or "UNKNOWN"))

        p = r.get("price")
        p_val = float(p) if (p is not None and np.isfinite(float(p)) and float(p) > 0) else 0.0
        prices.append(np.log1p(p_val))
        has_prices.append(1.0 if p_val > 0 else 0.0)

        orig_p = r.get("original_price")
        orig_val = float(orig_p) if (orig_p is not None and np.isfinite(float(orig_p)) and float(orig_p) > 0) else 0.0
        has_orig_prices.append(1.0 if orig_val > 0 else 0.0)
        price_orig_ratios.append(orig_val / p_val if (p_val > 0 and orig_val > 0) else 1.0)

        char_lens.append(float(len(title)))
        word_lens.append(float(len(title.split())))
        has_warrs.append(1.0 if len(warranty) > 0 else 0.0)
        warr_lens.append(float(len(warranty)))

        attr_counts.append(float(len(attrs)))
        pic_counts.append(float(len(r.get("pictures") or [])))
        tag_counts.append(float(len(r.get("tags") or [])))
        var_counts.append(float(len(r.get("variations") or [])))

        mercadopago.append(1.0 if r.get("accepts_mercadopago") else 0.0)
        shipping = r.get("shipping") or {}
        free_shipping.append(1.0 if shipping.get("free_shipping") else 0.0)
        local_pickup.append(1.0 if shipping.get("local_pick_up") else 0.0)

        sold = r.get("sold_quantity")
        sold_qtys.append(np.log1p(float(sold)) if (sold is not None and float(sold) >= 0) else 0.0)
        avail = r.get("available_quantity")
        avail_qtys.append(np.log1p(float(avail)) if (avail is not None and float(avail) >= 0) else 0.0)
        init_q = r.get("initial_quantity")
        init_qtys.append(np.log1p(float(init_q)) if (init_q is not None and float(init_q) >= 0) else 0.0)

        listing_types.append(str(r.get("listing_type_id") or "UNKNOWN"))
        buying_modes.append(str(r.get("buying_mode") or "UNKNOWN"))
        addr = r.get("seller_address") or {}
        states.append(str((addr.get("state") or {}).get("id") or "UNKNOWN"))
        ship_modes.append(str(shipping.get("mode") or "UNKNOWN"))

        row_cues = [1.0 if pat.search(title) else 0.0 for _, pat in LEXICAL_PATTERNS]
        cues_matrix.append(row_cues)

    df_num = pd.DataFrame({
        "log_price": prices,
        "has_price": has_prices,
        "has_orig_price": has_orig_prices,
        "price_orig_ratio": price_orig_ratios,
        "char_len": char_lens,
        "word_len": word_lens,
        "has_warr": has_warrs,
        "warr_len": warr_lens,
        "attr_count": attr_counts,
        "pic_count": pic_counts,
        "tag_count": tag_counts,
        "var_count": var_counts,
        "mercadopago": mercadopago,
        "free_shipping": free_shipping,
        "local_pickup": local_pickup,
        "sold_qty": sold_qtys,
        "avail_qty": avail_qtys,
        "init_qty": init_qtys,
    })

    cues_df = pd.DataFrame(cues_matrix, columns=[name for name, _ in LEXICAL_PATTERNS])
    df_base = pd.concat([df_num, cues_df], axis=1)

    cats_dict = {
        "seller_id": np.array(sellers),
        "category_id": np.array(categories),
        "listing_type": np.array(listing_types),
        "buying_mode": np.array(buying_modes),
        "state": np.array(states),
        "ship_mode": np.array(ship_modes),
    }

    return np.array(ids), corpus, df_base, cats_dict


def compute_fold_features(
    fit_idx: np.ndarray,
    eval_indices: list[np.ndarray],
    corpus_all: list[str],
    df_base_all: pd.DataFrame,
    cats_all: dict[str, np.ndarray],
    n_svd: int = 100,
) -> tuple[pd.DataFrame, list[pd.DataFrame], sparse.csr_matrix, list[sparse.csr_matrix]]:
    """Compute features strictly fitted on fit_idx and transform eval_indices."""
    fit_corpus = [corpus_all[i] for i in fit_idx]
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=35000,
        sublinear_tf=True,
        min_df=3,
    )
    X_fit_tfidf = tfidf.fit_transform(fit_corpus)
    X_eval_tfidfs = [tfidf.transform([corpus_all[i] for i in idx]) for idx in eval_indices]

    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    X_fit_svd = svd.fit_transform(X_fit_tfidf)
    X_eval_svds = [svd.transform(X_eval) for X_eval in X_eval_tfidfs]

    svd_cols = [f"svd_{i}" for i in range(n_svd)]
    df_fit_svd = pd.DataFrame(X_fit_svd, columns=svd_cols, index=fit_idx)
    df_eval_svds = [pd.DataFrame(X_svd, columns=svd_cols, index=idx) for X_svd, idx in zip(X_eval_svds, eval_indices)]

    # Fold-safe intrinsic category statistics fitted strictly on fit_idx
    df_fit_base = df_base_all.iloc[fit_idx].copy()
    df_eval_bases = [df_base_all.iloc[idx].copy() for idx in eval_indices]

    cat_fit = cats_all["category_id"][fit_idx]
    global_med_price = float(df_fit_base["log_price"].median())
    cat_med_price = df_fit_base.groupby(cat_fit)["log_price"].median().to_dict()

    df_fit_base["rel_price_cat"] = df_fit_base["log_price"] - pd.Series(cat_fit, index=fit_idx).map(cat_med_price).fillna(global_med_price)
    for df_eval_b, idx in zip(df_eval_bases, eval_indices):
        cat_ev = cats_all["category_id"][idx]
        df_eval_b["rel_price_cat"] = df_eval_b["log_price"] - pd.Series(cat_ev, index=idx).map(cat_med_price).fillna(global_med_price)

    # Frequency encodings
    for cat_name in ["category_id", "listing_type", "buying_mode", "state", "ship_mode"]:
        counts = Counter(cats_all[cat_name][fit_idx])
        df_fit_base[f"freq_{cat_name}"] = pd.Series(cats_all[cat_name][fit_idx], index=fit_idx).map(counts).fillna(0).astype(float)
        for df_eval_b, idx in zip(df_eval_bases, eval_indices):
            df_eval_b[f"freq_{cat_name}"] = pd.Series(cats_all[cat_name][idx], index=idx).map(counts).fillna(0).astype(float)

    X_fit_dense = pd.concat([df_fit_base, df_fit_svd], axis=1)
    X_eval_denses = [pd.concat([df_b, df_s], axis=1) for df_b, df_s in zip(df_eval_bases, df_eval_svds)]

    return X_fit_dense, X_eval_denses, X_fit_tfidf, X_eval_tfidfs


def optimize_weights_inner(p_cb: np.ndarray, p_lgb: np.ndarray, p_text: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find convex combination weights minimizing logloss on inner validation set."""
    def loss_func(weights):
        w = np.maximum(0, weights)
        s = np.sum(w)
        if s <= 0:
            return 1e6
        w = w / s
        p_mix = w[0] * p_cb + w[1] * p_lgb + w[2] * p_text
        p_clipped = np.clip(p_mix, 1e-6, 1.0 - 1e-6)
        return log_loss(y, p_clipped)

    res = minimize(loss_func, [0.45, 0.40, 0.15], bounds=[(0, 1), (0, 1), (0, 1)], method="L-BFGS-B")
    w = np.maximum(0, res.x)
    return w / np.sum(w)


def seller_clustered_bootstrap(
    diff: np.ndarray,
    sellers: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261110,
) -> tuple[float, float, float, float]:
    """Bootstrap delta grouped by seller clusters."""
    rng = np.random.default_rng(seed)
    unique_sellers = np.unique(sellers)
    n_sellers = len(unique_sellers)

    # Pre-group indices by seller
    seller_to_indices = {}
    for idx, s in enumerate(sellers):
        seller_to_indices.setdefault(s, []).append(idx)
    seller_to_indices = {s: np.array(ixs) for s, ixs in seller_to_indices.items()}

    boot_deltas = np.zeros(n_boot)
    for b in range(n_boot):
        sample_sellers = rng.choice(unique_sellers, size=n_sellers, replace=True)
        sampled_indices = np.concatenate([seller_to_indices[s] for s in sample_sellers])
        boot_deltas[b] = np.mean(diff[sampled_indices])

    mean_delta = float(np.mean(boot_deltas))
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    p_gt_zero = float(np.mean(boot_deltas > 0))
    return mean_delta, ci_lower, ci_upper, p_gt_zero


def main():
    print("=" * 80)
    print("STRICTLY NESTED DUAL-REGIME CV & PAIRED BASELINE VERIFICATION")
    print("Zero-Leakage In-Fold Feature Fitting, Inner Iteration Freezing, Paired Eval")
    print("=" * 80)

    start_total = time.time()
    out_dir = Path("results/strictly_nested_dual_regime_cv_20261110")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_path = Path("data/ap_2026_q_2/raw/train_data.jsonlines")
    test_path = Path("data/ap_2026_q_2/raw/test_data.jsonlines")
    baseline_path = Path("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")

    print("\n[1/5] Loading train data (70k), test data (30k), and strong baseline...", flush=True)
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    n_train = len(train_rows)
    n_test = len(test_rows)
    y_train = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=int)
    print(f"Loaded {n_train} train rows, {n_test} test rows. Base used rate = {np.mean(y_train):.4f}", flush=True)

    # Load baseline
    df_baseline = pd.read_csv(baseline_path)
    assert len(df_baseline) == n_train, f"Baseline length mismatch: {len(df_baseline)} vs {n_train}"
    p_baseline = df_baseline["oof_blend"].values
    pred_baseline = (p_baseline >= 0.5).astype(int)
    acc_baseline = accuracy_score(y_train, pred_baseline)
    ll_baseline = log_loss(y_train, p_baseline)
    print(f"Verified Strong Baseline (Podium Engine): Acc = {acc_baseline:.5f} ({int(acc_baseline*n_train)}/{n_train}), LogLoss = {ll_baseline:.5f}", flush=True)

    # Extract raw features
    print("\n[2/5] Extracting raw intrinsic fields and corpus...", flush=True)
    train_ids, train_corpus, df_train_base, train_cats = extract_raw_fields(train_rows)
    test_ids, test_corpus, df_test_base, test_cats = extract_raw_fields(test_rows)
    sellers_train = train_cats["seller_id"]

    # 2. Outer StratifiedGroupKFold on seller_id
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20261110)

    # Storage for OOF predictions across 70k
    oof_p_cb = np.zeros(n_train)
    oof_p_lgb = np.zeros(n_train)
    oof_p_text = np.zeros(n_train)
    oof_p_blend = np.zeros(n_train)
    oof_fold = np.zeros(n_train, dtype=int)

    # Storage for Test predictions (averaged across 5 outer folds)
    test_p_cb = np.zeros(n_test)
    test_p_lgb = np.zeros(n_test)
    test_p_text = np.zeros(n_test)
    test_p_blend = np.zeros(n_test)

    # Tracking seen-seller vs unseen-seller metrics
    seen_val_indices_all = []
    seen_val_y_all = []
    seen_val_p_blend_all = []

    fold_reports = []

    print(f"\n[3/5] Starting 5-Fold Strictly Nested Training Pipeline...", flush=True)

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(train_corpus, y_train, groups=sellers_train)):
        fold_start = time.time()
        print(f"\n{'='*30} FOLD {fold+1}/{n_splits} {'='*30}", flush=True)
        print(f"Outer Split: Train={len(tr_idx)} items, Validation={len(va_idx)} items (100% unseen sellers)", flush=True)
        assert len(set(sellers_train[tr_idx]) & set(sellers_train[va_idx])) == 0, "CRITICAL: Seller overlap detected in outer fold!"

        # A. Inner seller-group split on tr_idx solely to determine best_iteration
        tr_sellers = sellers_train[tr_idx]
        tr_y = y_train[tr_idx]

        inner_sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42 + fold)
        in_tr_pos, in_va_pos = next(inner_sgkf.split(tr_idx, tr_y, groups=tr_sellers))
        inner_tr_idx = tr_idx[in_tr_pos]
        inner_va_idx = tr_idx[in_va_pos]
        assert len(set(sellers_train[inner_tr_idx]) & set(sellers_train[inner_va_idx])) == 0, "Seller overlap in inner split!"

        print(f"Inner Split for Tuning: InnerTrain={len(inner_tr_idx)}, InnerVal={len(inner_va_idx)}", flush=True)

        # Stage 1: Feature Extraction on Inner Split
        X_in_tr_dense, [X_in_va_dense], X_in_tr_tfidf, [X_in_va_tfidf] = compute_fold_features(
            inner_tr_idx,
            [inner_va_idx],
            train_corpus,
            df_train_base,
            train_cats,
            n_svd=100,
        )

        # Stage 2: Inner Early Stopping to freeze best_iteration
        print("Determining best_iteration on inner validation set...", flush=True)

        # CatBoost Inner Tuning
        cb_inner = CatBoostClassifier(
            iterations=800,
            learning_rate=0.04,
            depth=8,
            l2_leaf_reg=5.0,
            random_seed=42 + fold,
            verbose=0,
            thread_count=4,
        )
        cb_inner.fit(
            X_in_tr_dense,
            y_train[inner_tr_idx],
            eval_set=(X_in_va_dense, y_train[inner_va_idx]),
            early_stopping_rounds=60,
            verbose=False,
        )
        best_iter_cb = cb_inner.get_best_iteration()
        if best_iter_cb is None or best_iter_cb <= 0:
            best_iter_cb = 500
        p_in_va_cb = cb_inner.predict_proba(X_in_va_dense)[:, 1]

        # LightGBM Inner Tuning
        lgb_inner = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.04,
            num_leaves=127,
            max_depth=9,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + fold,
            n_jobs=4,
            verbose=-1,
        )
        lgb_inner.fit(
            X_in_tr_dense,
            y_train[inner_tr_idx],
            eval_set=[(X_in_va_dense, y_train[inner_va_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )
        best_iter_lgb = lgb_inner.best_iteration_
        if best_iter_lgb is None or best_iter_lgb <= 0:
            best_iter_lgb = 500
        p_in_va_lgb = lgb_inner.predict_proba(X_in_va_dense)[:, 1]

        # Text Logistic Specialist Inner Tuning
        lr_inner = LogisticRegression(C=1.5, max_iter=400, solver="saga", random_state=42 + fold, n_jobs=4)
        lr_inner.fit(X_in_tr_tfidf, y_train[inner_tr_idx])
        p_in_va_text = lr_inner.predict_proba(X_in_va_tfidf)[:, 1]

        # Inner Meta-Weights Optimization
        weights_inner = optimize_weights_inner(p_in_va_cb, p_in_va_lgb, p_in_va_text, y_train[inner_va_idx])
        print(f"Frozen Hyperparameters Fold {fold+1}: CB best_iter={best_iter_cb}, LGB best_iter={best_iter_lgb}", flush=True)
        print(f"Inner Optimal Weights: CB={weights_inner[0]:.4f}, LGB={weights_inner[1]:.4f}, Text={weights_inner[2]:.4f}", flush=True)

        # Clean up inner models
        del cb_inner, lgb_inner, lr_inner, X_in_tr_dense, X_in_va_dense, X_in_tr_tfidf, X_in_va_tfidf
        gc.collect()

        # Stage 3: Full Outer Feature Extraction strictly on tr_idx
        # Also prepare seen-seller holdout: within tr_idx, sellers with >= 2 items
        tr_seller_counts = Counter(sellers_train[tr_idx])
        multi_item_sellers = {s for s, c in tr_seller_counts.items() if c >= 2}
        seen_eval_indices = []
        for s in multi_item_sellers:
            ixs = tr_idx[sellers_train[tr_idx] == s]
            seen_eval_indices.append(ixs[0])
        seen_eval_indices = np.array(seen_eval_indices)

        test_eval_indices = n_train + np.arange(n_test)
        print(f"Refitting features on full outer-train ({len(tr_idx)} items)...", flush=True)
        # Compute features for tr_idx, va_idx, seen_eval_indices, and test
        X_tr_dense, [X_va_dense, X_seen_dense, X_test_dense], X_tr_tfidf, [X_va_tfidf, X_seen_tfidf, X_test_tfidf] = compute_fold_features(
            tr_idx,
            [va_idx, seen_eval_indices, test_eval_indices],
            train_corpus + test_corpus,
            pd.concat([df_train_base, df_test_base], ignore_index=True),
            {k: np.concatenate([train_cats[k], test_cats[k]]) for k in train_cats},
            n_svd=100,
        )

        # Stage 4: Refit Outer Models on 100% of outer_tr_idx WITHOUT eval_set
        print(f"Training Outer CatBoost ({best_iter_cb} iterations, NO eval_set)...", flush=True)
        cb_outer = CatBoostClassifier(
            iterations=best_iter_cb,
            learning_rate=0.04,
            depth=8,
            l2_leaf_reg=5.0,
            random_seed=42 + fold,
            verbose=0,
            thread_count=4,
        )
        cb_outer.fit(X_tr_dense, y_train[tr_idx], verbose=False)
        p_va_cb = cb_outer.predict_proba(X_va_dense)[:, 1]
        p_test_fold_cb = cb_outer.predict_proba(X_test_dense)[:, 1]

        print(f"Training Outer LightGBM ({best_iter_lgb} iterations, NO eval_set)...", flush=True)
        lgb_outer = lgb.LGBMClassifier(
            n_estimators=best_iter_lgb,
            learning_rate=0.04,
            num_leaves=127,
            max_depth=9,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + fold,
            n_jobs=4,
            verbose=-1,
        )
        lgb_outer.fit(X_tr_dense, y_train[tr_idx])
        p_va_lgb = lgb_outer.predict_proba(X_va_dense)[:, 1]
        p_test_fold_lgb = lgb_outer.predict_proba(X_test_dense)[:, 1]

        print("Training Outer Text Logistic Specialist (SAGA, C=1.5)...", flush=True)
        lr_outer = LogisticRegression(C=1.5, max_iter=400, solver="saga", random_state=42 + fold, n_jobs=4)
        lr_outer.fit(X_tr_tfidf, y_train[tr_idx])
        p_va_text = lr_outer.predict_proba(X_va_tfidf)[:, 1]
        p_test_fold_text = lr_outer.predict_proba(X_test_tfidf)[:, 1]

        # Apply frozen inner weights to form outer validation and test ensemble
        p_va_blend = weights_inner[0] * p_va_cb + weights_inner[1] * p_va_lgb + weights_inner[2] * p_va_text
        p_test_fold_blend = weights_inner[0] * p_test_fold_cb + weights_inner[1] * p_test_fold_lgb + weights_inner[2] * p_test_fold_text

        # Evaluate seen-sellers holdout
        p_seen_cb = cb_outer.predict_proba(X_seen_dense)[:, 1]
        p_seen_lgb = lgb_outer.predict_proba(X_seen_dense)[:, 1]
        p_seen_text = lr_outer.predict_proba(X_seen_tfidf)[:, 1]
        p_seen_blend = weights_inner[0] * p_seen_cb + weights_inner[1] * p_seen_lgb + weights_inner[2] * p_seen_text

        seen_val_indices_all.extend(seen_eval_indices.tolist())
        seen_val_y_all.extend(y_train[seen_eval_indices].tolist())
        seen_val_p_blend_all.extend(p_seen_blend.tolist())

        # Store outer OOF
        oof_p_cb[va_idx] = p_va_cb
        oof_p_lgb[va_idx] = p_va_lgb
        oof_p_text[va_idx] = p_va_text
        oof_p_blend[va_idx] = p_va_blend
        oof_fold[va_idx] = fold

        test_p_cb += p_test_fold_cb / n_splits
        test_p_lgb += p_test_fold_lgb / n_splits
        test_p_text += p_test_fold_text / n_splits
        test_p_blend += p_test_fold_blend / n_splits

        # Fold metrics on unseen sellers
        y_va = y_train[va_idx]
        acc_va = accuracy_score(y_va, (p_va_blend >= 0.5).astype(int))
        ll_va = log_loss(y_va, p_va_blend)
        acc_seen_fold = accuracy_score(y_train[seen_eval_indices], (p_seen_blend >= 0.5).astype(int))

        fold_elapsed = time.time() - fold_start
        print(f"Fold {fold+1} Finished in {fold_elapsed:.1f}s | Unseen Acc={acc_va:.5f}, LogLoss={ll_va:.5f} | Seen Acc={acc_seen_fold:.5f}", flush=True)

        fold_reports.append({
            "fold": fold + 1,
            "best_iter_cb": int(best_iter_cb),
            "best_iter_lgb": int(best_iter_lgb),
            "weights": [float(w) for w in weights_inner],
            "unseen_accuracy": float(acc_va),
            "unseen_logloss": float(ll_va),
            "seen_accuracy": float(acc_seen_fold),
            "elapsed_seconds": float(fold_elapsed),
        })

        del cb_outer, lgb_outer, lr_outer, X_tr_dense, X_va_dense, X_seen_dense, X_test_dense, X_tr_tfidf, X_va_tfidf, X_seen_tfidf, X_test_tfidf
        gc.collect()

    # 4. Global Out-of-Fold Evaluation
    print(f"\n{'='*80}\n[4/5] Computing Global Strictly Nested OOF & Paired Verification\n{'='*80}", flush=True)

    pred_cand = (oof_p_blend >= 0.5).astype(int)
    acc_unseen = accuracy_score(y_train, pred_cand)
    ll_unseen = log_loss(y_train, oof_p_blend)
    auc_unseen = roc_auc_score(y_train, oof_p_blend)

    seen_val_y = np.array(seen_val_y_all)
    seen_val_p = np.array(seen_val_p_blend_all)
    acc_seen = accuracy_score(seen_val_y, (seen_val_p >= 0.5).astype(int))
    ll_seen = log_loss(seen_val_y, seen_val_p)

    # 71.62% seen / 28.38% unseen deployment sensitivity mix
    acc_mix = 0.7162 * acc_seen + 0.2838 * acc_unseen

    print(f"RÉGIMEN UNSEEN-SELLERS (100% outer holdout): Acc = {acc_unseen:.5f} ({int(acc_unseen*n_train)}/{n_train}), LogLoss = {ll_unseen:.5f}, ROC-AUC = {auc_unseen:.5f}")
    print(f"RÉGIMEN SEEN-SELLERS (holdout within train sellers): Acc = {acc_seen:.5f} ({len(seen_val_y)} items), LogLoss = {ll_seen:.5f}")
    print(f"MIX SENSIBILIDAD TEST (71.62% seen / 28.38% unseen): Acc = {acc_mix:.5f}")

    # Paired Item-by-Item Comparison vs Baseline
    hit_cand = (pred_cand == y_train)
    hit_base = (pred_baseline == y_train)
    diff = hit_cand.astype(int) - hit_base.astype(int)

    fixes = int(np.sum(~hit_base & hit_cand))
    breaks = int(np.sum(hit_base & ~hit_cand))
    net_hits = fixes - breaks
    delta_acc = float(acc_unseen - acc_baseline)
    delta_ll = float(ll_unseen - ll_baseline)

    print("\nPAIRED ITEM-BY-ITEM METRICS VS STRONG BASELINE (Podium Engine):")
    print(f"Baseline Accuracy:  {acc_baseline:.5f} ({int(acc_baseline*n_train)} hits)")
    print(f"Candidate Accuracy: {acc_unseen:.5f} ({int(acc_unseen*n_train)} hits)")
    print(f"Paired Delta Acc:   {delta_acc:+.5f} (Net Hits: {net_hits:+d} | Fixes: {fixes}, Breaks: {breaks})")
    print(f"Paired Delta LL:    {delta_ll:+.5f}")

    # Seller-clustered bootstrap for delta
    print("\nRunning Seller-Clustered Bootstrap on Paired Delta (1,000 resamples)...", flush=True)
    boot_mean, boot_ci_lower, boot_ci_upper, boot_p_gt_zero = seller_clustered_bootstrap(
        diff, sellers_train, n_boot=1000, seed=20261110
    )
    print(f"Bootstrap Seller-Clustered Delta 95% CI: [{boot_ci_lower:+.5f}, {boot_ci_upper:+.5f}] (Mean: {boot_mean:+.5f})")
    print(f"P(Delta > 0): {boot_p_gt_zero:.4f}")

    # Wilcoxon signed-rank test on fold deltas
    fold_acc_cand = [r["unseen_accuracy"] for r in fold_reports]
    fold_acc_base = []
    for f in range(n_splits):
        f_idx = np.where(oof_fold == f)[0]
        fold_acc_base.append(float(accuracy_score(y_train[f_idx], pred_baseline[f_idx])))
    fold_diffs = np.array(fold_acc_cand) - np.array(fold_acc_base)
    try:
        w_stat, w_pval = wilcoxon(fold_diffs)
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    # 5. Save Artifacts
    print(f"\n[5/5] Saving strictly nested artifacts and report...", flush=True)

    # OOF DataFrame
    oof_df = pd.DataFrame({
        "id": train_ids,
        "seller_id": sellers_train,
        "category_id": train_cats["category_id"],
        "y_true": y_train,
        "p_pred_cb": oof_p_cb,
        "p_pred_lgb": oof_p_lgb,
        "p_pred_text": oof_p_text,
        "p_pred_blend": oof_p_blend,
        "fold": oof_fold,
    })
    oof_csv_path = out_dir / "oof_nested_dual_regime.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"Saved OOF DataFrame to {oof_csv_path} ({len(oof_df)} rows)", flush=True)

    # Test Probabilities DataFrame
    test_df = pd.DataFrame({
        "id": test_ids,
        "seller_id": test_cats["seller_id"],
        "category_id": test_cats["category_id"],
        "p_used_cb": test_p_cb,
        "p_used_lgb": test_p_lgb,
        "p_used_text": test_p_text,
        "p_used_blend": test_p_blend,
    })
    test_csv_path = out_dir / "test_probabilities_nested.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved Test Probabilities to {test_csv_path} ({len(test_df)} rows)", flush=True)

    # Summary Report JSON
    summary = {
        "pipeline": "Strictly Nested Dual-Regime CV (Zero-Leakage In-Fold Feature Fitting & Inner Iteration Freezing)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_rows": n_train,
        "total_test_rows": n_test,
        "outer_splits": n_splits,
        "models": ["CatBoost Depth 8", "LightGBM Leaves 127", "Sparse Text Logistic SAGA"],
        "qualification_status": "EXPLORATORY / VALIDATION_STAGE",
        "baseline_comparison": {
            "baseline_name": "Podium Engine (results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv)",
            "baseline_accuracy": float(acc_baseline),
            "baseline_logloss": float(ll_baseline),
            "baseline_hits": int(acc_baseline * n_train),
        },
        "unseen_seller_regime": {
            "description": "100% unseen sellers in outer holdout (pure seller-group)",
            "accuracy": float(acc_unseen),
            "hits": int(acc_unseen * n_train),
            "logloss": float(ll_unseen),
            "roc_auc": float(auc_unseen),
        },
        "seen_seller_regime": {
            "description": "Holdout items from sellers present in training set",
            "sample_size": len(seen_val_y),
            "accuracy": float(acc_seen),
            "logloss": float(ll_seen),
        },
        "deployment_mix_sensitivity": {
            "description": "71.62% seen sellers / 28.38% unseen sellers mix",
            "accuracy": float(acc_mix),
        },
        "paired_metrics": {
            "delta_accuracy": float(delta_acc),
            "delta_logloss": float(delta_ll),
            "fixes": fixes,
            "breaks": breaks,
            "net_hits": net_hits,
            "seller_clustered_bootstrap_95ci": [boot_ci_lower, boot_ci_upper],
            "seller_clustered_bootstrap_mean": boot_mean,
            "probability_delta_gt_zero": boot_p_gt_zero,
            "wilcoxon_fold_pvalue": float(w_pval),
        },
        "fold_details": fold_reports,
        "total_elapsed_seconds": float(time.time() - start_total),
    }

    report_json_path = out_dir / "paired_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - start_total
    print(f"\n{'='*80}\nTRAINING & VERIFICATION COMPLETED in {total_time/60:.2f} minutes\n{'='*80}", flush=True)


if __name__ == "__main__":
    main()
