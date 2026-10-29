"""Leak-free external seller-group validation and complementary model testing.

Evaluates strong baselines (HistGB, LightGBM, CatBoost) and tests complementary
hypotheses (multi-model ensembling, calibrated thresholding, semantic condition cues,
and relative category price) on strict, external seller-group partitions.
Freeze all choices on tune before evaluating confirm.
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# Base directories
ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
RESULTS_DIR = ROOT / "results" / "revision_submissions"


def load_train_data():
    print(f"Loading train data from {TRAIN_JSON}...", flush=True)
    t0 = time.time()
    rows = []
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} train records in {time.time()-t0:.2f}s", flush=True)
    
    # Target: 1 if used, 0 if new (competition standard)
    y_used = np.array([1 if r["condition"] == "used" else 0 for r in rows], dtype=int)
    ids = np.array([str(r["id"]) for r in rows])
    sellers = np.array([str(r["seller_id"]) for r in rows])
    titles = [r.get("title") or "" for r in rows]
    warranties = [r.get("warranty") or "" for r in rows]
    
    return rows, ids, sellers, y_used, titles, warranties


def split_seller_groups(seller, seed):
    """External seller-group split with disjoint new sellers in tune and confirm."""
    rng = np.random.default_rng(seed)
    unique = np.unique(seller)
    rng.shuffle(unique)
    
    # 10% completely new sellers in tune, 10% in confirm
    n_new = len(unique) // 10
    new_tune = set(unique[:n_new])
    new_confirm = set(unique[n_new: 2 * n_new])
    
    # Remaining sellers are seen in fit, but retain 10% rows in tune and 10% in confirm
    seen_sellers = ~(np.isin(seller, list(new_tune | new_confirm)))
    seen_row_indices = np.where(seen_sellers)[0]
    rng.shuffle(seen_row_indices)
    
    n_seen_tune = len(seen_row_indices) // 10
    seen_tune = seen_row_indices[:n_seen_tune]
    seen_confirm = seen_row_indices[n_seen_tune: 2 * n_seen_tune]
    
    tune_rows = np.flatnonzero(np.isin(seller, list(new_tune)))
    confirm_rows = np.flatnonzero(np.isin(seller, list(new_confirm)))
    
    tune = np.r_[tune_rows, seen_tune]
    confirm = np.r_[confirm_rows, seen_confirm]
    fit = np.setdiff1d(np.arange(len(seller)), np.r_[tune, confirm])
    
    # Assert strict disjointness
    assert not set(seller[tune]).intersection(new_confirm)
    assert not set(seller[confirm]).intersection(new_tune)
    assert not set(seller[fit]).intersection(new_tune)
    assert not set(seller[fit]).intersection(new_confirm)
    
    return fit, tune, confirm


def extract_metadata_features(rows, indices, fit_cat_medians=None):
    """Extract tabular, numerical, and semantic cue features."""
    data = []
    
    # Cue regex patterns
    CUE_PATTERNS = {
        "cue_como_nuevo": r"(?i)\bcomo\s+nuev[oa]s?\b",
        "cue_impecable": r"(?i)\bimpecable\b|\bexcelente\s+estado\b|\bmuy\s+buen\s+estado\b",
        "cue_en_caja": r"(?i)\ben\s+caja\b|\bcaja\s+original\b",
        "cue_sin_uso": r"(?i)\bsin\s+uso\b|\bnunca\s+usad[oa]\b",
        "cue_repuesto": r"(?i)\brepuesto\b|\bpara\s+repuesto\b|\bdesarme\b|\brot[oa]\b",
        "cue_detalles": r"(?i)\bcon\s+detalles?\b|\bdetalles?\s+de\s+uso\b",
        "cue_usado": r"(?i)\busad[oa]s?\b",
        "cue_nuevo": r"(?i)\bnuev[oa]s?\b",
        "cue_vintage": r"(?i)\bvintage\b|\bantigu[oa]\b|\bde\s+colecci[oó]n\b",
        "cue_lote": r"(?i)\blote\b|\bcolecci[oó]n\b",
        "cue_original": r"(?i)\boriginal\b",
    }
    
    WARR_PATTERNS = {
        "warr_sin": r"(?i)\bsin\s+garant[ií]a\b|\bno\s+tiene\s+garant[ií]a\b",
        "warr_meses": r"(?i)\bmes(es)?\b",
        "warr_dias": r"(?i)\bd[ií]as?\b",
        "warr_fabrica": r"(?i)\bf[aá]brica\b|\boficial\b|\bde\s+por\s+vida\b",
        "warr_prueba": r"(?i)\bprueba\b|\bfunciona\b|\bprobar\b",
    }

    for idx in indices:
        r = rows[idx]
        title = r.get("title") or ""
        warranty = r.get("warranty") or ""
        sh = r.get("shipping") or {}
        adr = r.get("seller_address") or {}
        city = ((adr.get("city") or {}).get("name") or "").lower().strip()
        state = (adr.get("state") or {}).get("id") or ""
        pics = r.get("pictures") or []
        attrs = r.get("attributes") or []
        variations = r.get("variations") or []
        tags = r.get("tags") or []
        
        price = float(r.get("price") or 0)
        log_price = np.log1p(price)
        cat_id = str(r.get("category_id") or "")
        
        # Relative price to category median
        cat_med = fit_cat_medians.get(cat_id, fit_cat_medians.get("__overall__", log_price)) if fit_cat_medians else log_price
        rel_price_cat = log_price - cat_med
        
        init_q = float(r.get("initial_quantity") or 0)
        avail_q = float(r.get("available_quantity") or 0)
        sold_q = float(r.get("sold_quantity") or 0)
        
        feat = {
            "log_price": log_price,
            "rel_price_cat": rel_price_cat,
            "log_init_q": np.log1p(init_q),
            "log_avail_q": np.log1p(avail_q),
            "log_sold_q": np.log1p(sold_q),
            "sold_ratio": sold_q / (init_q + 1.0),
            "init_is_one": int(init_q == 1),
            "sold_is_zero": int(sold_q == 0),
            "title_len": len(title),
            "title_words": len(title.split()),
            "title_upper_frac": sum(ch.isupper() for ch in title) / max(len(title), 1),
            "title_digits": sum(ch.isdigit() for ch in title),
            "n_pics": len(pics),
            "n_attrs": len(attrs),
            "n_variations": len(variations),
            "has_warranty": int(bool(warranty)),
            "warranty_len": len(warranty),
            "free_shipping": int(bool(sh.get("free_shipping"))),
            "local_pickup": int(bool(sh.get("local_pick_up"))),
            "accepts_mp": int(bool(r.get("accepts_mercadopago"))),
            "has_video": int(bool(r.get("video_id"))),
            "auto_relist": int(bool(r.get("automatic_relist"))),
            "tag_dragged_bids": int("dragged_bids_and_visits" in tags),
            "category_id": cat_id,
            "listing_type_id": str(r.get("listing_type_id") or ""),
            "buying_mode": str(r.get("buying_mode") or ""),
            "shipping_mode": str(sh.get("mode") or ""),
            "state": state,
        }
        
        # Cues
        for name, pat in CUE_PATTERNS.items():
            feat[name] = int(bool(re.search(pat, title)))
        for name, pat in WARR_PATTERNS.items():
            feat[name] = int(bool(re.search(pat, warranty)))
            
        data.append(feat)
        
    return data


def compute_category_medians(rows, fit_indices):
    cat_prices = defaultdict(list)
    all_prices = []
    for idx in fit_indices:
        r = rows[idx]
        price = float(r.get("price") or 0)
        lp = np.log1p(price)
        cat_id = str(r.get("category_id") or "")
        cat_prices[cat_id].append(lp)
        all_prices.append(lp)
        
    overall_med = float(np.median(all_prices)) if all_prices else 0.0
    medians = {"__overall__": overall_med}
    for cat_id, lps in cat_prices.items():
        medians[cat_id] = float(np.median(lps))
    return medians


def cluster_bootstrap_ci(y_true, base_pred, test_pred, clusters, n_boot=2000, seed=42):
    """Cluster bootstrap confidence interval for net correct answers (test - base) by seller cluster."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    n_clust = len(unique_clusters)
    
    # Pre-index items per cluster
    clust_to_idx = defaultdict(list)
    for i, c in enumerate(clusters):
        clust_to_idx[c].append(i)
        
    # Precompute per-sample fix/break difference: +1 if test right and base wrong, -1 if test wrong and base right, 0 otherwise
    base_correct = (base_pred == y_true)
    test_correct = (test_pred == y_true)
    diff = test_correct.astype(int) - base_correct.astype(int)
    
    # Precompute per-cluster sum of differences
    clust_sums = np.array([np.sum(diff[clust_to_idx[c]]) for c in unique_clusters])
    
    boot_nets = np.zeros(n_boot)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_clust, size=n_clust)
        boot_nets[b] = np.sum(clust_sums[sample_idx])
        
    ci_low = float(np.percentile(boot_nets, 2.5))
    ci_high = float(np.percentile(boot_nets, 97.5))
    return ci_low, ci_high, float(np.mean(boot_nets))


def main():
    rows, ids, sellers, y_used, titles, warranties = load_train_data()
    
    seeds = [2026, 314159]
    all_results = []
    
    for seed in seeds:
        print(f"\n=======================================================", flush=True)
        print(f"STARTING EXPERIMENT ON SEED {seed}", flush=True)
        print(f"=======================================================", flush=True)
        
        fit, tune, confirm = split_seller_groups(sellers, seed)
        print(f"Split sizes: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}", flush=True)
        
        # Measure overlap between seed confirm sets
        # (will be checked at end across seeds)
        
        # 1. Compute Category medians strictly on fit
        cat_medians = compute_category_medians(rows, fit)
        
        # 2. Extract tabular metadata for fit, tune, confirm
        print("Extracting metadata features...", flush=True)
        meta_fit = extract_metadata_features(rows, fit, cat_medians)
        meta_tune = extract_metadata_features(rows, tune, cat_medians)
        meta_confirm = extract_metadata_features(rows, confirm, cat_medians)
        
        # 3. Build text vectorizers strictly on fit
        print("Fitting TF-IDF text vectorizers on fit...", flush=True)
        vec_title_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=75000,
                                         sublinear_tf=True, strip_accents="unicode")
        vec_title_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                         max_features=120000, sublinear_tf=True)
        vec_warr = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000,
                                   sublinear_tf=True, strip_accents="unicode")
        
        t0 = time.time()
        w_fit = vec_title_word.fit_transform([titles[i] for i in fit])
        c_fit = vec_title_char.fit_transform([titles[i] for i in fit])
        wr_fit = vec_warr.fit_transform([warranties[i] for i in fit])
        
        dict_vec = DictVectorizer(sparse=True)
        m_fit = dict_vec.fit_transform(meta_fit)
        
        X_sparse_fit = hstack([w_fit, c_fit, wr_fit, m_fit], format="csr")
        print(f"X_sparse_fit shape: {X_sparse_fit.shape} in {time.time()-t0:.2f}s", flush=True)
        
        # Transform tune and confirm
        w_tune = vec_title_word.transform([titles[i] for i in tune])
        c_tune = vec_title_char.transform([titles[i] for i in tune])
        wr_tune = vec_warr.transform([warranties[i] for i in tune])
        m_tune = dict_vec.transform(meta_tune)
        X_sparse_tune = hstack([w_tune, c_tune, wr_tune, m_tune], format="csr")
        
        w_confirm = vec_title_word.transform([titles[i] for i in confirm])
        c_confirm = vec_title_char.transform([titles[i] for i in confirm])
        wr_confirm = vec_warr.transform([warranties[i] for i in confirm])
        m_confirm = dict_vec.transform(meta_confirm)
        X_sparse_confirm = hstack([w_confirm, c_confirm, wr_confirm, m_confirm], format="csr")
        
        # 4. Train Sparse Linear Model and evaluate 4-fold GroupKFold OOF inside fit
        print("Training Sparse Linear Model with 4-fold GroupKFold by seller on fit...", flush=True)
        oof_p_sparse = np.zeros(len(fit), dtype=np.float32)
        gkf = GroupKFold(n_splits=4)
        for fold, (tr, va) in enumerate(gkf.split(X_sparse_fit, y_used[fit], sellers[fit])):
            lr_fold = LogisticRegression(C=2.5, max_iter=200, solver="liblinear", random_state=seed + fold)
            lr_fold.fit(X_sparse_fit[tr], y_used[fit][tr])
            oof_p_sparse[va] = lr_fold.predict_proba(X_sparse_fit[va])[:, 1]
            acc_fold = accuracy_score(y_used[fit][va], oof_p_sparse[va] > 0.5)
            print(f"  Sparse OOF fold {fold}: acc={acc_fold:.4f}", flush=True)
            
        print(f"Total Sparse OOF on fit: acc={accuracy_score(y_used[fit], oof_p_sparse > 0.5):.4f}", flush=True)
        
        # Fit full sparse model on fit
        lr_full = LogisticRegression(C=2.5, max_iter=250, solver="liblinear", random_state=seed)
        lr_full.fit(X_sparse_fit, y_used[fit])
        p_sparse_tune = lr_full.predict_proba(X_sparse_tune)[:, 1]
        p_sparse_confirm = lr_full.predict_proba(X_sparse_confirm)[:, 1]
        
        # 5. Build tabular matrix for Tree Models
        NUM_COLS = [
            "log_price", "rel_price_cat", "log_init_q", "log_avail_q", "log_sold_q",
            "sold_ratio", "init_is_one", "sold_is_zero", "title_len", "title_words",
            "title_upper_frac", "title_digits", "n_pics", "n_attrs", "n_variations",
            "has_warranty", "warranty_len", "free_shipping", "local_pickup",
            "accepts_mp", "has_video", "auto_relist", "tag_dragged_bids",
            "cue_como_nuevo", "cue_impecable", "cue_en_caja", "cue_sin_uso",
            "cue_repuesto", "cue_detalles", "cue_usado", "cue_nuevo", "cue_vintage",
            "cue_lote", "cue_original", "warr_sin", "warr_meses", "warr_dias",
            "warr_fabrica", "warr_prueba"
        ]
        CAT_COLS = ["category_id", "listing_type_id", "buying_mode", "shipping_mode", "state"]
        
        # Prepare DataFrames for tabular models
        df_tab_fit = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_fit])
        df_tab_tune = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_tune])
        df_tab_confirm = pd.DataFrame([{col: m[col] for col in NUM_COLS + CAT_COLS} for m in meta_confirm])
        
        # Target encoding for categorical features inside fit (clean GroupKFold TE)
        # For LightGBM and HistGB, convert categoricals to integer codes
        cat_maps = {}
        for c in CAT_COLS:
            unique_vals = {v: i + 1 for i, v in enumerate(df_tab_fit[c].unique())}
            cat_maps[c] = unique_vals
            df_tab_fit[c + "_code"] = df_tab_fit[c].map(lambda x: unique_vals.get(x, 0))
            df_tab_tune[c + "_code"] = df_tab_tune[c].map(lambda x: unique_vals.get(x, 0))
            df_tab_confirm[c + "_code"] = df_tab_confirm[c].map(lambda x: unique_vals.get(x, 0))
            
        tree_features = NUM_COLS + [c + "_code" for c in CAT_COLS]
        
        X_tab_fit = df_tab_fit[tree_features].values.astype(np.float32)
        X_tab_tune = df_tab_tune[tree_features].values.astype(np.float32)
        X_tab_confirm = df_tab_confirm[tree_features].values.astype(np.float32)
        
        # Append clean sparse text OOF as a feature for tree models
        X_tree_fit = np.c_[X_tab_fit, oof_p_sparse]
        X_tree_tune = np.c_[X_tab_tune, p_sparse_tune]
        X_tree_confirm = np.c_[X_tab_confirm, p_sparse_confirm]
        
        LOW_CAT_COLS = ["listing_type_id", "buying_mode", "shipping_mode", "state"]
        low_cat_indices = [tree_features.index(c + "_code") for c in LOW_CAT_COLS]
        
        # 6. Train HistGradientBoosting
        print("Training HistGradientBoosting on fit...", flush=True)
        hgb = HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=31,
            learning_rate=0.07,
            l2_regularization=5.0,
            random_state=seed,
            categorical_features=low_cat_indices
        )
        t0 = time.time()
        hgb.fit(X_tree_fit, y_used[fit])
        print(f"HistGB fitted in {time.time()-t0:.2f}s", flush=True)
        p_hist_tune = hgb.predict_proba(X_tree_tune)[:, 1]
        p_hist_confirm = hgb.predict_proba(X_tree_confirm)[:, 1]
        
        # 7. Train LightGBM
        print("Training LightGBM on fit...", flush=True)
        lgb_model = lgb.LGBMClassifier(
            num_leaves=35,
            learning_rate=0.05,
            n_estimators=350,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=12,
            verbose=-1
        )
        t0 = time.time()
        lgb_model.fit(X_tree_fit, y_used[fit])
        print(f"LightGBM fitted in {time.time()-t0:.2f}s", flush=True)
        p_lgb_tune = lgb_model.predict_proba(X_tree_tune)[:, 1]
        p_lgb_confirm = lgb_model.predict_proba(X_tree_confirm)[:, 1]
        
        # 8. Train CatBoost
        print("Training CatBoost on fit...", flush=True)
        df_cat_fit = df_tab_fit[NUM_COLS + CAT_COLS].copy()
        df_cat_fit["sparse_p"] = oof_p_sparse
        for c in CAT_COLS:
            df_cat_fit[c] = df_cat_fit[c].astype(str)
            
        df_cat_tune = df_tab_tune[NUM_COLS + CAT_COLS].copy()
        df_cat_tune["sparse_p"] = p_sparse_tune
        for c in CAT_COLS:
            df_cat_tune[c] = df_cat_tune[c].astype(str)
            
        df_cat_confirm = df_tab_confirm[NUM_COLS + CAT_COLS].copy()
        df_cat_confirm["sparse_p"] = p_sparse_confirm
        for c in CAT_COLS:
            df_cat_confirm[c] = df_cat_confirm[c].astype(str)
            
        cat_pool_fit = Pool(df_cat_fit, y_used[fit], cat_features=CAT_COLS)
        cat_pool_tune = Pool(df_cat_tune, cat_features=CAT_COLS)
        cat_pool_confirm = Pool(df_cat_confirm, cat_features=CAT_COLS)
        
        cb_model = CatBoostClassifier(
            iterations=800,
            depth=5,
            learning_rate=0.08,
            l2_leaf_reg=5.0,
            random_seed=seed,
            thread_count=12,
            verbose=0
        )
        t0 = time.time()
        cb_model.fit(cat_pool_fit)
        print(f"CatBoost fitted in {time.time()-t0:.2f}s", flush=True)
        p_cat_tune = cb_model.predict_proba(cat_pool_tune)[:, 1]
        p_cat_confirm = cb_model.predict_proba(cat_pool_confirm)[:, 1]
        
        # 9. Evaluate seller-history gate against strong baselines (Experiment 1 check)
        print("\n--- Evaluating Seller-History Gate against Strong Baselines ---", flush=True)
        counts = defaultdict(lambda: [0, 0])
        for s, y_val in zip(sellers[fit], y_used[fit]):
            counts[s][0] += int(y_val)
            counts[s][1] += 1
        p0 = float(np.mean(y_used[fit]))
        
        def get_prior(indices):
            pos = np.array([counts[s][0] for s in sellers[indices]])
            n = np.array([counts[s][1] for s in sellers[indices]])
            return (pos + 3.0 * p0) / (n + 3.0), n
            
        pp_tune, n_tune = get_prior(tune)
        pp_confirm, n_confirm = get_prior(confirm)
        
        # Test gate against HistGB, LightGBM, and CatBoost
        models_dict = {
            "HistGB": (p_hist_tune, p_hist_confirm),
            "LightGBM": (p_lgb_tune, p_lgb_confirm),
            "CatBoost": (p_cat_tune, p_cat_confirm),
            "SparseLinear": (p_sparse_tune, p_sparse_confirm)
        }
        
        gate_results = {}
        for m_name, (pt, pc) in models_dict.items():
            # Tune gate
            gate_t = (n_tune >= 3) & (np.abs(pp_tune - 0.5) >= 0.30) & ((pp_tune >= 0.5) != (pt >= 0.5))
            pmix_t = pt.copy()
            pmix_t[gate_t] = 0.6 * pp_tune[gate_t] + 0.4 * pt[gate_t]
            
            # Confirm gate (frozen rule)
            gate_c = (n_confirm >= 3) & (np.abs(pp_confirm - 0.5) >= 0.30) & ((pp_confirm >= 0.5) != (pc >= 0.5))
            pmix_c = pc.copy()
            pmix_c[gate_c] = 0.6 * pp_confirm[gate_c] + 0.4 * pc[gate_c]
            
            # Metrics on confirm
            y_c = y_used[confirm]
            b_pred = (pc >= 0.5)
            g_pred = (pmix_c >= 0.5)
            
            fixes = int(np.sum((b_pred != y_c) & (g_pred == y_c)))
            breaks = int(np.sum((b_pred == y_c) & (g_pred != y_c)))
            net = fixes - breaks
            acc_base = accuracy_score(y_c, b_pred)
            acc_gate = accuracy_score(y_c, g_pred)
            ll_base = log_loss(y_c, pc)
            ll_gate = log_loss(y_c, pmix_c)
            
            # Bootstrap CI on net
            ci_low, ci_high, _ = cluster_bootstrap_ci(y_c, b_pred, g_pred, sellers[confirm])
            
            gate_results[m_name] = {
                "n_confirm": len(confirm),
                "n_gated": int(gate_c.sum()),
                "fixes": fixes,
                "breaks": breaks,
                "net": net,
                "acc_base": float(acc_base),
                "acc_gate": float(acc_gate),
                "ll_base": float(ll_base),
                "ll_gate": float(ll_gate),
                "ci_95": [ci_low, ci_high]
            }
            print(f"Gate on {m_name} (Confirm): gated={gate_c.sum()}, fix={fixes}, brk={breaks}, NET={net:+d}, Acc: {acc_base:.5f}->{acc_gate:.5f}, CI95: [{ci_low:+.1f}, {ci_high:+.1f}]", flush=True)

        # 10. Test Complementary Hypothesis 1: Multi-Model Ensembling and Threshold Calibration
        print("\n--- Testing Multi-Model Ensemble and Threshold Calibration ---", flush=True)
        # Search weights and threshold on TUNE
        best_tune_acc = 0.0
        best_weights = None
        best_thr = 0.50
        
        # Grid search over convex blends on tune
        for w_cat in [0.3, 0.4, 0.5, 0.6]:
            for w_lgb in [0.2, 0.3, 0.4]:
                w_sp = round(1.0 - w_cat - w_lgb, 2)
                if w_sp < 0:
                    continue
                p_blend_tune = w_cat * p_cat_tune + w_lgb * p_lgb_tune + w_sp * p_sparse_tune
                for thr in np.arange(0.47, 0.53, 0.005):
                    pred_tune = (p_blend_tune >= thr).astype(int)
                    acc = accuracy_score(y_used[tune], pred_tune)
                    if acc > best_tune_acc:
                        best_tune_acc = acc
                        best_weights = (w_cat, w_lgb, w_sp)
                        best_thr = float(thr)
                        
        print(f"Tune Best Ensemble: weights={best_weights}, thr={best_thr:.3f}, Tune Acc={best_tune_acc:.5f}", flush=True)
        
        # Freeze weights and threshold, evaluate once on CONFIRM
        w_cat, w_lgb, w_sp = best_weights
        p_ens_confirm = w_cat * p_cat_confirm + w_lgb * p_lgb_confirm + w_sp * p_sparse_confirm
        pred_ens_confirm = (p_ens_confirm >= best_thr).astype(int)
        
        # Comparison vs Single Strong Baselines on CONFIRM
        y_c = y_used[confirm]
        cb_pred_c = (p_cat_confirm >= 0.5).astype(int)
        hgb_pred_c = (p_hist_confirm >= 0.5).astype(int)
        lgb_pred_c = (p_lgb_confirm >= 0.5).astype(int)
        
        # Net vs CatBoost
        fixes_vs_cb = int(np.sum((cb_pred_c != y_c) & (pred_ens_confirm == y_c)))
        breaks_vs_cb = int(np.sum((cb_pred_c == y_c) & (pred_ens_confirm != y_c)))
        net_vs_cb = fixes_vs_cb - breaks_vs_cb
        ci_cb_low, ci_cb_high, _ = cluster_bootstrap_ci(y_c, cb_pred_c, pred_ens_confirm, sellers[confirm])
        
        # Seen vs New sellers breakdown
        seen_mask = (n_confirm > 0)
        new_mask = (n_confirm == 0)
        
        acc_ens_total = accuracy_score(y_c, pred_ens_confirm)
        acc_cb_total = accuracy_score(y_c, cb_pred_c)
        ll_ens_total = log_loss(y_c, p_ens_confirm)
        ll_cb_total = log_loss(y_c, p_cat_confirm)
        
        acc_ens_seen = accuracy_score(y_c[seen_mask], pred_ens_confirm[seen_mask])
        acc_cb_seen = accuracy_score(y_c[seen_mask], cb_pred_c[seen_mask])
        acc_ens_new = accuracy_score(y_c[new_mask], pred_ens_confirm[new_mask])
        acc_cb_new = accuracy_score(y_c[new_mask], cb_pred_c[new_mask])
        
        # Flip directions vs CatBoost
        flips_u2n = int(np.sum((cb_pred_c == 1) & (pred_ens_confirm == 0)))
        flips_n2u = int(np.sum((cb_pred_c == 0) & (pred_ens_confirm == 1)))
        
        print(f"CONFIRM RESULTS (Seed {seed}):", flush=True)
        print(f"  CatBoost Base Acc: {acc_cb_total:.5f}, LL: {ll_cb_total:.5f}", flush=True)
        print(f"  Ensemble Acc:     {acc_ens_total:.5f}, LL: {ll_ens_total:.5f}", flush=True)
        print(f"  Fixes/Breaks vs CB: {fixes_vs_cb}/{breaks_vs_cb} -> NET: {net_vs_cb:+d}", flush=True)
        print(f"  Bootstrap 95% CI vs CB: [{ci_cb_low:+.1f}, {ci_cb_high:+.1f}]", flush=True)
        print(f"  Seen sellers Acc: {acc_cb_seen:.5f} -> {acc_ens_seen:.5f} (n={seen_mask.sum()})", flush=True)
        print(f"  New sellers Acc:  {acc_cb_new:.5f} -> {acc_ens_new:.5f} (n={new_mask.sum()})", flush=True)
        print(f"  Flips: used->new={flips_u2n}, new->used={flips_n2u}", flush=True)
        
        # Save paired predictions on confirm
        pairs_file = RESULTS_DIR / f"paired_predictions_seed_{seed}.csv"
        with open(pairs_file, "w", newline="", encoding="utf-8") as pf:
            w = csv.writer(pf)
            w.writerow([
                "id", "seller_id", "y_true_used", "is_seen_seller",
                "p_sparse_used", "p_hist_used", "p_lgb_used", "p_cat_used",
                "p_ensemble_used", "pred_cat_used", "pred_ensemble_used"
            ])
            for i, idx in enumerate(confirm):
                w.writerow([
                    ids[idx], sellers[idx], y_used[idx], int(seen_mask[i]),
                    round(float(p_sparse_confirm[i]), 5),
                    round(float(p_hist_confirm[i]), 5),
                    round(float(p_lgb_confirm[i]), 5),
                    round(float(p_cat_confirm[i]), 5),
                    round(float(p_ens_confirm[i]), 5),
                    int(cb_pred_c[i]),
                    int(pred_ens_confirm[i])
                ])
        print(f"Saved paired predictions to {pairs_file}", flush=True)
        
        seed_record = {
            "seed": seed,
            "fit_size": len(fit),
            "tune_size": len(tune),
            "confirm_size": len(confirm),
            "confirm_seen_size": int(seen_mask.sum()),
            "confirm_new_size": int(new_mask.sum()),
            "gate_against_baselines": gate_results,
            "best_ensemble_weights": list(best_weights),
            "best_ensemble_threshold": best_thr,
            "tune_ensemble_acc": float(best_tune_acc),
            "confirm_metrics": {
                "catboost_acc": float(acc_cb_total),
                "catboost_ll": float(ll_cb_total),
                "histgb_acc": float(accuracy_score(y_c, hgb_pred_c)),
                "lightgbm_acc": float(accuracy_score(y_c, lgb_pred_c)),
                "sparse_acc": float(accuracy_score(y_c, (p_sparse_confirm >= 0.5).astype(int))),
                "ensemble_acc": float(acc_ens_total),
                "ensemble_ll": float(ll_ens_total),
                "fixes_vs_catboost": fixes_vs_cb,
                "breaks_vs_catboost": breaks_vs_cb,
                "net_vs_catboost": net_vs_cb,
                "ci_95_vs_catboost": [ci_cb_low, ci_cb_high],
                "seen_catboost_acc": float(acc_cb_seen),
                "seen_ensemble_acc": float(acc_ens_seen),
                "new_catboost_acc": float(acc_cb_new),
                "new_ensemble_acc": float(acc_ens_new),
                "flips_used_to_new": flips_u2n,
                "flips_new_to_used": flips_n2u
            }
        }
        all_results.append(seed_record)
        
    # Check overlap between confirms of the two seeds
    fit1, tune1, confirm1 = split_seller_groups(sellers, 2026)
    fit2, tune2, confirm2 = split_seller_groups(sellers, 314159)
    shared_confirm_ids = set(ids[confirm1]).intersection(set(ids[confirm2]))
    shared_confirm_sellers = set(sellers[confirm1]).intersection(set(sellers[confirm2]))
    print(f"\nConfirm Split Overlap: {len(shared_confirm_ids)} shared IDs, {len(shared_confirm_sellers)} shared sellers.", flush=True)
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shared_confirm_ids": len(shared_confirm_ids),
        "shared_confirm_sellers": len(shared_confirm_sellers),
        "results": all_results
    }
    
    results_json = RESULTS_DIR / "round142_honest_validation_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_json}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
