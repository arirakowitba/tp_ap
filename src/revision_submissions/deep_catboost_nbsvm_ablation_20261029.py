"""Ablation of NB-SVM OOF inside Strong Deep CatBoost Architecture.

Implements the full deep tabular feature set (from Round 126 / feats.py: pic stats,
dates/durations, regex condition cues, payment methods, seller unsupervised aggregations)
combined with leak-free internal GroupKFold target encodings and text OOF signals.

Evaluates on honest external seller-group partitions (fit ~45k, tune ~12k, confirm ~13k)
across two independent seeds (2026 and 314159).
Tests whether adding NB-SVM OOF probability directly as an internal feature inside
the deep CatBoost model improves over the identical deep CatBoost baseline.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
RESULTS_DIR = ROOT / "results" / "revision_submissions"


def lst(x):
    return x if isinstance(x, list) else []


def dims(p):
    try:
        w, h = p.get("max_size", p.get("size", "0x0")).split("x")
        return int(w), int(h)
    except Exception:
        return 0, 0


def picnum(p):
    m = re.search(r"/(\d+)-MLA\d+_(\d+)-", p.get("url", "") or "")
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def load_dataset():
    t0 = time.time()
    rows = []
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    ids = np.array([str(r["id"]) for r in rows])
    sellers = np.array([str(r["seller_id"]) for r in rows])
    y_used = np.array([1 if r["condition"] == "used" else 0 for r in rows], dtype=int)
    print(f"Loaded {len(rows)} train records in {time.time()-t0:.2f}s", flush=True)
    return rows, ids, sellers, y_used


def extract_deep_features(rows):
    t0 = time.time()
    N = len(rows)
    
    # Text collections
    titles = [r.get("title") or "" for r in rows]
    warranties = [r.get("warranty") or "" for r in rows]
    combos = []
    
    # Numerical and binary containers
    feats = {}
    
    # 1. Price and quantities
    price = np.array([float(r.get("price") or 0) for r in rows], dtype=np.float32)
    base_price = np.array([float(r.get("base_price") or 0) for r in rows], dtype=np.float32)
    init_q = np.array([float(r.get("initial_quantity") or 0) for r in rows], dtype=np.float32)
    avail_q = np.array([float(r.get("available_quantity") or 0) for r in rows], dtype=np.float32)
    sold_q = np.array([float(r.get("sold_quantity") or 0) for r in rows], dtype=np.float32)
    
    feats["log_price"] = np.log1p(price)
    feats["price_eq_base"] = (price == base_price).astype(np.float32)
    feats["sold_ratio"] = sold_q / (init_q + 1.0)
    feats["avail_ratio"] = avail_q / (init_q + 1.0)
    feats["init_minus_avail"] = init_q - avail_q
    feats["price_cents"] = ((price * 100) % 100).round()
    feats["price_round100"] = ((price % 100) == 0).astype(np.float32)
    feats["price_round10"] = ((price % 10) == 0).astype(np.float32)
    feats["price_end9"] = ((np.round(price) % 10) == 9).astype(np.float32)
    
    feats["has_original_price"] = np.array([1.0 if r.get("original_price") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["official_store"] = np.array([1.0 if r.get("official_store_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_video"] = np.array([1.0 if r.get("video_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_catalog"] = np.array([1.0 if r.get("catalog_product_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_parent"] = np.array([1.0 if r.get("parent_item_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["automatic_relist"] = np.array([1.0 if r.get("automatic_relist") else 0.0 for r in rows], dtype=np.float32)
    feats["accepts_mp"] = np.array([1.0 if r.get("accepts_mercadopago") else 0.0 for r in rows], dtype=np.float32)
    feats["n_deals"] = np.array([float(len(lst(r.get("deal_ids")))) for r in rows], dtype=np.float32)
    feats["currency_usd"] = np.array([1.0 if r.get("currency_id") == "USD" else 0.0 for r in rows], dtype=np.float32)
    
    # 2. Tags
    for t_tag in ["dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits", "free_relist", "poor_quality_thumbnail"]:
        feats[f"tag_{t_tag}"] = np.array([1.0 if t_tag in lst(r.get("tags")) else 0.0 for r in rows], dtype=np.float32)
    feats["n_tags"] = np.array([float(len(lst(r.get("tags")))) for r in rows], dtype=np.float32)
    
    # 3. Payment methods
    for code in ["MLATB", "MLAWC", "MLAMO", "MLAOT", "MLACD", "MLABC", "MLAVE", "MLAMC", "MLAAM", "MLADC", "MLAVS", "MLAMS", "MLAWT", "MLAMP", "MLAAB"]:
        feats[f"pm_{code}"] = np.array([1.0 if code in [m.get("id") for m in lst(r.get("non_mercado_pago_payment_methods"))] else 0.0 for r in rows], dtype=np.float32)
    feats["n_pm"] = np.array([float(len(lst(r.get("non_mercado_pago_payment_methods")))) for r in rows], dtype=np.float32)
    
    # 4. Shipping
    feats["ship_local_pick_up"] = np.array([1.0 if (r.get("shipping") or {}).get("local_pick_up") else 0.0 for r in rows], dtype=np.float32)
    feats["ship_free"] = np.array([1.0 if (r.get("shipping") or {}).get("free_shipping") else 0.0 for r in rows], dtype=np.float32)
    feats["ship_dims"] = np.array([1.0 if (r.get("shipping") or {}).get("dimensions") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["ship_n_methods"] = np.array([float(len(lst((r.get("shipping") or {}).get("methods")))) for r in rows], dtype=np.float32)
    feats["ship_n_tags"] = np.array([float(len(lst((r.get("shipping") or {}).get("tags")))) for r in rows], dtype=np.float32)
    
    # 5. Pictures
    n_pics = []
    pic_max_area = []
    pic_mean_area = []
    pic_prefix = []
    pic_code = []
    thumb_code = []
    for r in rows:
        pics = lst(r.get("pictures"))
        n_pics.append(float(len(pics)))
        if pics:
            areas = [dims(p)[0] * dims(p)[1] for p in pics]
            pic_max_area.append(float(max(areas)))
            pic_mean_area.append(float(np.mean(areas)))
            pnum = picnum(pics[0])
            pic_prefix.append(float(pnum[0]))
            pic_code.append(float(pnum[1]))
        else:
            pic_max_area.append(-1.0)
            pic_mean_area.append(-1.0)
            pic_prefix.append(-1.0)
            pic_code.append(-1.0)
        u = r.get("thumbnail") or ""
        m = re.search(r"_(\d+)-", u)
        thumb_code.append(float(m.group(1)) if m else -1.0)
        
    feats["n_pics"] = np.array(n_pics, dtype=np.float32)
    feats["pic_max_area"] = np.array(pic_max_area, dtype=np.float32)
    feats["pic_mean_area"] = np.array(pic_mean_area, dtype=np.float32)
    feats["pic_prefix"] = np.array(pic_prefix, dtype=np.float32)
    feats["pic_code"] = np.array(pic_code, dtype=np.float32)
    feats["thumb_code"] = np.array(thumb_code, dtype=np.float32)
    
    # 6. Variations / Attributes
    feats["n_variations"] = np.array([float(len(lst(r.get("variations")))) for r in rows], dtype=np.float32)
    feats["n_attributes"] = np.array([float(len(lst(r.get("attributes")))) for r in rows], dtype=np.float32)
    feats["var_sold"] = np.array([float(sum(v.get("sold_quantity", 0) or 0 for v in lst(r.get("variations")))) for r in rows], dtype=np.float32)
    
    # Build text combo
    for r in rows:
        t = r.get("title") or ""
        w = r.get("warranty") or ""
        attrs = " ".join(str(a.get("id", "")) for a in lst(r.get("attributes")))
        combos.append(f"{t} || {w} || {attrs}")
        
    # 7. Dates and Durations
    ref = pd.Timestamp("2013-01-01", tz="UTC")
    dates_created = pd.to_datetime([r.get("date_created") for r in rows], utc=True, errors="coerce")
    dates_updated = pd.to_datetime([r.get("last_updated") for r in rows], utc=True, errors="coerce")
    st = pd.to_datetime([r.get("start_time") for r in rows], unit="ms", utc=True, errors="coerce")
    sp = pd.to_datetime([r.get("stop_time") for r in rows], unit="ms", utc=True, errors="coerce")
    
    feats["created_days"] = np.nan_to_num((dates_created - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["dur_days"] = np.nan_to_num((sp - st).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["upd_minus_created"] = np.nan_to_num((dates_updated - dates_created).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["created_hour"] = np.nan_to_num(dates_created.hour, nan=0.0).astype(np.float32)
    feats["created_dow"] = np.nan_to_num(dates_created.dayofweek, nan=0.0).astype(np.float32)
    
    # 8. Text Regex and Stats
    feats["warranty_null"] = np.array([1.0 if not r.get("warranty") else 0.0 for r in rows], dtype=np.float32)
    feats["warranty_len"] = np.array([float(len(w)) for w in warranties], dtype=np.float32)
    for kw, pat in {
        "w_sin": r"\bsin\b", "w_no": r"\bno\b", "w_meses": r"mes", "w_anio": r"a[ñn]o",
        "w_fabrica": r"f[aá]brica", "w_oficial": r"oficial", "w_usado": r"usad", "w_nuevo": r"nuev",
        "w_dias": r"d[ií]as", "w_garantia": r"garant", "w_prueba": r"prueba|probad|funciona"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, w.lower())) else 0.0 for w in warranties], dtype=np.float32)
        
    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_words"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["title_upper_frac"] = np.array([sum(ch.isupper() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_digits"] = np.array([float(sum(ch.isdigit() for ch in t)) for t in titles], dtype=np.float32)
    for kw, pat in {
        "t_usado": r"usad", "t_nuevo": r"nuev", "t_original": r"original",
        "t_antiguo": r"antig|vintage|retro|colecci", "t_libro": r"libro", "t_lote": r"lote",
        "t_impecable": r"impecable|excelente estado|muy buen estado"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, t.lower())) else 0.0 for t in titles], dtype=np.float32)
        
    # Categoricals
    cat_df = pd.DataFrame({
        "category_id": [str(r.get("category_id") or "") for r in rows],
        "listing_type_id": [str(r.get("listing_type_id") or "") for r in rows],
        "buying_mode": [str(r.get("buying_mode") or "") for r in rows],
        "state": [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "") for r in rows],
        "city": [str(((r.get("seller_address") or {}).get("city") or {}).get("name") or "").lower().strip() for r in rows],
        "seller_id": [str(r.get("seller_id") or "") for r in rows],
        "ship_mode": [str((r.get("shipping") or {}).get("mode") or "") for r in rows],
    })
    
    # 9. Unsupervised aggregations (frequencies and relative price)
    for col in ["seller_id", "category_id", "city"]:
        counts = cat_df[col].value_counts()
        feats[f"freq_{col}"] = cat_df[col].map(counts).fillna(1.0).values.astype(np.float32)
        
    df_num = pd.DataFrame(feats)
    df_num["seller_id"] = cat_df["seller_id"]
    df_num["category_id"] = cat_df["category_id"]
    
    # Seller unsupervised stats
    g_seller = df_num.groupby("seller_id", observed=True)
    feats["seller_n_cats"] = g_seller["category_id"].transform("nunique").values.astype(np.float32)
    feats["seller_mean_logprice"] = g_seller["log_price"].transform("mean").values.astype(np.float32)
    feats["seller_mean_init_q"] = g_seller["sold_ratio"].transform("mean").values.astype(np.float32)
    feats["sell_mean_w_null"] = g_seller["warranty_null"].transform("mean").values.astype(np.float32)
    feats["sell_mean_t_usado"] = g_seller["t_usado"].transform("mean").values.astype(np.float32)
    feats["sell_mean_t_nuevo"] = g_seller["t_nuevo"].transform("mean").values.astype(np.float32)
    feats["sell_mean_ship_free"] = g_seller["ship_free"].transform("mean").values.astype(np.float32)
    feats["sell_mean_auto_relist"] = g_seller["automatic_relist"].transform("mean").values.astype(np.float32)
    feats["sell_mean_n_pics"] = g_seller["n_pics"].transform("mean").values.astype(np.float32)
    
    # Relative price to category median
    cat_med = df_num.groupby("category_id", observed=True)["log_price"].transform("median")
    feats["rel_price_cat"] = (df_num["log_price"] - cat_med).values.astype(np.float32)
    
    F_df = pd.DataFrame(feats)
    for col in ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]:
        F_df[col] = cat_df[col].astype(str)
        
    print(f"Extracted deep feature matrix {F_df.shape} in {time.time()-t0:.2f}s", flush=True)
    return F_df, cat_df, titles, warranties, combos


def split_seller_groups(seller, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(seller)
    rng.shuffle(unique)
    n_new = len(unique) // 10
    new_tune = set(unique[:n_new])
    new_confirm = set(unique[n_new: 2 * n_new])
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
    return fit, tune, confirm


def cluster_bootstrap_ci(y_true, base_pred, test_pred, clusters, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    n_clust = len(unique_clusters)
    clust_to_idx = defaultdict(list)
    for i, c in enumerate(clusters):
        clust_to_idx[c].append(i)
    base_correct = (base_pred == y_true)
    test_correct = (test_pred == y_true)
    diff = test_correct.astype(int) - base_correct.astype(int)
    clust_sums = np.array([np.sum(diff[clust_to_idx[c]]) for c in unique_clusters])
    boot_nets = np.zeros(n_boot)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_clust, size=n_clust)
        boot_nets[b] = np.sum(clust_sums[sample_idx])
    return float(np.percentile(boot_nets, 2.5)), float(np.percentile(boot_nets, 97.5)), float(np.mean(boot_nets))


def run_ablation_on_seed(seed, F_df, cat_df, titles, warranties, combos, y_used, sellers, ids):
    print(f"\n==================================================================", flush=True)
    print(f"RUNNING DEEP CATBOOST + NB-SVM ABLATION ON SEED {seed}", flush=True)
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
    
    # Out-of-fold inside fit
    for tr, va in gkf.split(fit, y_fit, sellers[fit]):
        for c in te_cols:
            k_tr = cat_df[c].iloc[fit[tr]]
            k_va = cat_df[c].iloc[fit[va]]
            stats = pd.DataFrame({"k": k_tr, "y": y_fit[tr]}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + 5.0 * prior) / (stats["count"] + 5.0)
            fit_te[c][va] = k_va.map(smoothed).fillna(prior).values.astype(np.float32)
            
    # Full fit TE for tune and confirm
    for c in te_cols:
        k_fit = cat_df[c].iloc[fit]
        stats = pd.DataFrame({"k": k_fit, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
        smoothed = (stats["sum"] + 5.0 * prior) / (stats["count"] + 5.0)
        tune_te[c] = cat_df[c].iloc[tune].map(smoothed).fillna(prior).values.astype(np.float32)
        confirm_te[c] = cat_df[c].iloc[confirm].map(smoothed).fillna(prior).values.astype(np.float32)
        
    # 2. Text Representations
    print("Fitting Sparse TF-IDF and NB-SVM text models on fit...", flush=True)
    # Model A: Sparse TF-IDF on title + warranty + attributes combo
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
    
    # Model B: NB-SVM CountVectorizer on title + warranty
    txt_nb_fit = [f"{titles[i]} __w__ {warranties[i]}" for i in fit]
    txt_nb_tune = [f"{titles[i]} __w__ {warranties[i]}" for i in tune]
    txt_nb_conf = [f"{titles[i]} __w__ {warranties[i]}" for i in confirm]
    
    cv_w = CountVectorizer(ngram_range=(1, 3), min_df=3, binary=True, strip_accents="unicode", max_features=100000)
    cv_c = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, binary=True, strip_accents="unicode", max_features=120000)
    
    X_nb_w_fit = cv_w.fit_transform(txt_nb_fit)
    X_nb_c_fit = cv_c.fit_transform([titles[i] for i in fit])
    X_nb_fit = sp.hstack([X_nb_w_fit, X_nb_c_fit]).tocsr().astype(np.float32)
    
    X_nb_w_tune = cv_w.transform(txt_nb_tune)
    X_nb_c_tune = cv_c.transform([titles[i] for i in tune])
    X_nb_tune = sp.hstack([X_nb_w_tune, X_nb_c_tune]).tocsr().astype(np.float32)
    
    X_nb_w_conf = cv_w.transform(txt_nb_conf)
    X_nb_c_conf = cv_c.transform([titles[i] for i in confirm])
    X_nb_conf = sp.hstack([X_nb_w_conf, X_nb_c_conf]).tocsr().astype(np.float32)
    
    # Clean GroupKFold OOF for both text models inside fit
    oof_sp = np.zeros(len(fit), dtype=np.float32)
    oof_nb = np.zeros(len(fit), dtype=np.float32)
    
    for tr, va in gkf.split(fit, y_fit, sellers[fit]):
        # Sparse Logistic Regression
        lr_sp = LogisticRegression(C=3.0, max_iter=250, solver="liblinear", random_state=seed)
        lr_sp.fit(X_sp_fit[tr], y_fit[tr])
        oof_sp[va] = lr_sp.predict_proba(X_sp_fit[va])[:, 1]
        
        # NB-SVM
        p_cnt = 1.0 + X_nb_fit[tr][y_fit[tr] == 1].sum(0)
        q_cnt = 1.0 + X_nb_fit[tr][y_fit[tr] == 0].sum(0)
        r_fold = np.asarray(np.log((p_cnt / p_cnt.sum()) / (q_cnt / q_cnt.sum()))).ravel()
        R_fold = sp.diags(r_fold)
        
        lr_nb = LogisticRegression(C=0.5, max_iter=250, solver="liblinear", random_state=seed)
        lr_nb.fit(X_nb_fit[tr] @ R_fold, y_fit[tr])
        oof_nb[va] = lr_nb.predict_proba(X_nb_fit[va] @ R_fold)[:, 1]
        
    print(f"Fit OOF Sparse Acc: {accuracy_score(y_fit, oof_sp > 0.5):.4f}, NBSVM Acc: {accuracy_score(y_fit, oof_nb > 0.5):.4f}", flush=True)
    
    # Full fit text models to predict tune & confirm
    lr_sp_full = LogisticRegression(C=3.0, max_iter=300, solver="liblinear", random_state=seed).fit(X_sp_fit, y_fit)
    p_sp_tune = lr_sp_full.predict_proba(X_sp_tune)[:, 1]
    p_sp_conf = lr_sp_full.predict_proba(X_sp_conf)[:, 1]
    
    p_cnt_f = 1.0 + X_nb_fit[y_fit == 1].sum(0)
    q_cnt_f = 1.0 + X_nb_fit[y_fit == 0].sum(0)
    r_full = np.asarray(np.log((p_cnt_f / p_cnt_f.sum()) / (q_cnt_f / q_cnt_f.sum()))).ravel()
    R_full = sp.diags(r_full)
    
    lr_nb_full = LogisticRegression(C=0.5, max_iter=300, solver="liblinear", random_state=seed).fit(X_nb_fit @ R_full, y_fit)
    p_nb_tune = lr_nb_full.predict_proba(X_nb_tune @ R_full)[:, 1]
    p_nb_conf = lr_nb_full.predict_proba(X_nb_conf @ R_full)[:, 1]
    
    # 3. Assemble Feature DataFrames for CatBoost
    CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]
    
    # Base DataFrame with deep features + TE + sparse text
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
        
    # Dual DataFrame: Base + txt_nbsvm_p
    df_dual_fit = df_base_fit.copy()
    df_dual_tune = df_base_tune.copy()
    df_dual_conf = df_base_conf.copy()
    
    df_dual_fit["txt_nbsvm_p"] = oof_nb
    df_dual_tune["txt_nbsvm_p"] = p_nb_tune
    df_dual_conf["txt_nbsvm_p"] = p_nb_conf
    
    # 4. Train Strong Deep CatBoost Baseline
    print("Training Strong Deep CatBoost Baseline (iterations=900, depth=6)...", flush=True)
    t0 = time.time()
    cb_base = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                 random_seed=seed, thread_count=12, verbose=0)
    cb_base.fit(Pool(df_base_fit, y_fit, cat_features=CAT_FEATURES))
    print(f"Deep CatBoost Baseline fitted in {time.time()-t0:.2f}s", flush=True)
    
    p_base_tune = cb_base.predict_proba(Pool(df_base_tune, cat_features=CAT_FEATURES))[:, 1]
    p_base_conf = cb_base.predict_proba(Pool(df_base_conf, cat_features=CAT_FEATURES))[:, 1]
    
    # 5. Train Deep CatBoost + NB-SVM Feature (Exact Ablation)
    print("Training Deep CatBoost + NB-SVM Feature (Exact Ablation)...", flush=True)
    t0 = time.time()
    cb_dual = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                 random_seed=seed, thread_count=12, verbose=0)
    cb_dual.fit(Pool(df_dual_fit, y_fit, cat_features=CAT_FEATURES))
    print(f"Deep CatBoost + NB-SVM fitted in {time.time()-t0:.2f}s", flush=True)
    
    p_dual_tune = cb_dual.predict_proba(Pool(df_dual_tune, cat_features=CAT_FEATURES))[:, 1]
    p_dual_conf = cb_dual.predict_proba(Pool(df_dual_conf, cat_features=CAT_FEATURES))[:, 1]
    
    # 6. Freeze Decision Threshold on TUNE before touching CONFIRM
    y_tu = y_used[tune]
    best_th_base, best_acc_base_tu = 0.50, 0.0
    best_th_dual, best_acc_dual_tu = 0.50, 0.0
    
    for th in np.linspace(0.40, 0.60, 41):
        acc_b = accuracy_score(y_tu, p_base_tune >= th)
        if acc_b > best_acc_base_tu:
            best_acc_base_tu = acc_b
            best_th_base = th
            
        acc_d = accuracy_score(y_tu, p_dual_tune >= th)
        if acc_d > best_acc_dual_tu:
            best_acc_dual_tu = acc_d
            best_th_dual = th
            
    print(f"Frozen Decision Thresholds on TUNE:")
    print(f"  Base Deep CatBoost: th={best_th_base:.3f} (Tune Acc: {best_acc_base_tu:.5f})")
    print(f"  Dual Deep CatBoost: th={best_th_dual:.3f} (Tune Acc: {best_acc_dual_tu:.5f})")
    
    # 7. Evaluate on Untouched CONFIRM Holdout
    y_co = y_used[confirm]
    pred_base_co = (p_base_conf >= best_th_base).astype(int)
    pred_dual_co = (p_dual_conf >= best_th_dual).astype(int)
    
    acc_base = accuracy_score(y_co, pred_base_co)
    acc_dual = accuracy_score(y_co, pred_dual_co)
    ll_base = log_loss(y_co, p_base_conf)
    ll_dual = log_loss(y_co, p_dual_conf)
    
    fixes = int(np.sum((pred_base_co != y_co) & (pred_dual_co == y_co)))
    breaks = int(np.sum((pred_base_co == y_co) & (pred_dual_co != y_co)))
    net = fixes - breaks
    
    ci_low, ci_high, _ = cluster_bootstrap_ci(y_co, pred_base_co, pred_dual_co, sellers[confirm])
    
    # Stratified: Seen vs New sellers in fit
    fit_sellers = set(sellers[fit])
    is_seen_co = np.array([s in fit_sellers for s in sellers[confirm]], dtype=bool)
    is_new_co = ~is_seen_co
    
    fixes_seen = int(np.sum((pred_base_co[is_seen_co] != y_co[is_seen_co]) & (pred_dual_co[is_seen_co] == y_co[is_seen_co])))
    breaks_seen = int(np.sum((pred_base_co[is_seen_co] == y_co[is_seen_co]) & (pred_dual_co[is_seen_co] != y_co[is_seen_co])))
    net_seen = fixes_seen - breaks_seen
    
    fixes_new = int(np.sum((pred_base_co[is_new_co] != y_co[is_new_co]) & (pred_dual_co[is_new_co] == y_co[is_new_co])))
    breaks_new = int(np.sum((pred_base_co[is_new_co] == y_co[is_new_co]) & (pred_dual_co[is_new_co] != y_co[is_new_co])))
    net_new = fixes_new - breaks_new
    
    print(f"\nCONFIRM RESULTS (Seed {seed}):")
    print(f"  Base Deep CatBoost Acc: {acc_base:.5f}, LL: {ll_base:.5f}")
    print(f"  Dual Deep CatBoost Acc: {acc_dual:.5f}, LL: {ll_dual:.5f}")
    print(f"  Fixes / Breaks:         {fixes} / {breaks} -> NET: {net:+d}")
    print(f"  Bootstrap 95% CI:       [{ci_low:+.1f}, {ci_high:+.1f}]")
    print(f"  Seen sellers net:       {net_seen:+d} (Fixes: {fixes_seen}, Breaks: {breaks_seen}, n={is_seen_co.sum()})")
    print(f"  New sellers net:        {net_new:+d} (Fixes: {fixes_new}, Breaks: {breaks_new}, n={is_new_co.sum()})")
    
    # Save paired predictions
    pairs_file = RESULTS_DIR / f"paired_deep_ablation_seed_{seed}.csv"
    with open(pairs_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "seller_id", "y_true", "is_seen", "p_base", "p_dual", "pred_base", "pred_dual"])
        for i in range(len(confirm)):
            w.writerow([
                ids[confirm[i]], sellers[confirm[i]], y_co[i], int(is_seen_co[i]),
                round(float(p_base_conf[i]), 5), round(float(p_dual_conf[i]), 5),
                pred_base_co[i], pred_dual_co[i]
            ])
            
    return {
        "seed": seed,
        "n_confirm": len(confirm),
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
    
    seeds = [int(s) for s in os.environ.get("AP_SEEDS", "2026,314159").split(",")]
    ablation_results = []
    
    for s in seeds:
        res = run_ablation_on_seed(s, F_df, cat_df, titles, warranties, combos, y_used, sellers, ids)
        ablation_results.append(res)
        
    out_json = RESULTS_DIR / "round151_deep_catboost_nbsvm_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nAll ablation results saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
