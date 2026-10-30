"""Feature Extraction for R151 (Pipeline A) and R157 (Pipeline B).
"""
import re
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import scipy.sparse as sp

def lst(x):
    return x if isinstance(x, list) else []

def dims(p):
    if not isinstance(p, dict):
        return 0, 0
    w, h = p.get("max_size", p.get("size", "0x0")).split("x")
    try:
        return int(w), int(h)
    except:
        return 0, 0

def picnum(p):
    if not isinstance(p, dict):
        return -1, -1
    url = p.get("url", "")
    m = re.findall(r"-(\d+)-", url)
    if not m:
        return -1, -1
    nums = [int(x) for x in m]
    return nums[0], (nums[1] if len(nums) > 1 else -1)

def extract_r151_features(rows):
    """Extracts exact deep feature set from R151 (deep_catboost_nbsvm_ablation_20261029.py)."""
    t0 = time.time()
    feats = {}
    
    prices = np.array([float(r.get("price") or 0.0) for r in rows], dtype=np.float32)
    feats["log_price"] = np.log1p(np.maximum(prices, 0.0))
    feats["price_is_zero"] = (prices == 0.0).astype(np.float32)
    
    sold = np.array([float(r.get("sold_quantity") or 0.0) for r in rows], dtype=np.float32)
    init_q = np.array([float(r.get("initial_quantity") or 0.0) for r in rows], dtype=np.float32)
    feats["sold_quantity"] = sold
    feats["initial_quantity"] = init_q
    feats["sold_ratio"] = sold / np.maximum(init_q, 1.0)
    feats["has_sold"] = (sold > 0.0).astype(np.float32)
    
    created = [str(r.get("date_created") or "") for r in rows]
    upd = [str(r.get("last_updated") or "") for r in rows]
    stop = [str(r.get("stop_time") or "") for r in rows]
    
    def parse_dt(s):
        try:
            return pd.to_datetime(s)
        except:
            return pd.NaT

    dt_cr = pd.to_datetime(created, errors="coerce")
    dt_up = pd.to_datetime(upd, errors="coerce")
    dt_st = pd.to_datetime(stop, errors="coerce")
    
    feats["dur_days"] = np.nan_to_num((dt_st - dt_cr).dt.total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["upd_minus_created"] = np.nan_to_num((dt_up - dt_cr).dt.total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["created_dow"] = dt_cr.dt.dayofweek.fillna(-1.0).values.astype(np.float32)
    
    pics = [lst(r.get("pictures")) for r in rows]
    feats["n_pics"] = np.array([float(len(p)) for p in pics], dtype=np.float32)
    
    areas, max_w, max_h = [], [], []
    prefixes, pic_codes = [], []
    for p in pics:
        if len(p) == 0:
            areas.append(0.0); max_w.append(0.0); max_h.append(0.0)
            prefixes.append("none"); pic_codes.append("none")
        else:
            whs = [dims(x) for x in p]
            a = [w * h for w, h in whs]
            areas.append(float(np.mean(a)))
            max_w.append(float(max(w for w, h in whs)))
            max_h.append(float(max(h for w, h in whs)))
            u = p[0].get("url", "")
            m_pre = re.search(r"http[s]?://([a-zA-Z0-9_\-]+)\.", u)
            prefixes.append(m_pre.group(1) if m_pre else "unk")
            p0 = p[0].get("id", "")
            pic_codes.append(str(p0)[:6] if p0 else "unk")
            
    feats["pic_mean_area"] = np.array(areas, dtype=np.float32)
    feats["pic_max_w"] = np.array(max_w, dtype=np.float32)
    feats["pic_max_h"] = np.array(max_h, dtype=np.float32)
    
    warranties = [r.get("warranty") or "" for r in rows]
    feats["warranty_null"] = np.array([1.0 if not (w and str(w).strip()) else 0.0 for w in warranties], dtype=np.float32)
    feats["warranty_len"] = np.array([float(len(str(w or ""))) for w in warranties], dtype=np.float32)
    for kw, pat in {
        "w_meses": r"mes|meses|m\b", "w_anos": r"ao|aos|ano|anos",
        "w_dias": r"da|das|dia|dias", "w_fabrica": r"fbrica|fabrica",
        "w_oficial": r"oficial", "w_sin": r"sin garant|no tiene"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, str(w or "").lower())) else 0.0 for w in warranties], dtype=np.float32)
        
    titles = [r.get("title") or "" for r in rows]
    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_words"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["title_upper_frac"] = np.array([sum(ch.isupper() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_digits"] = np.array([float(sum(ch.isdigit() for ch in t)) for t in titles], dtype=np.float32)
    
    for kw, pat in {
        "t_usado": r"usad", "t_nuevo": r"nuev", "t_original": r"original",
        "t_antiguo": r"antig|vintage|retro|colecci", "t_libro": r"libro", "t_lote": r"lote",
        "t_impecable": r"impecable|excelente estado", "t_caja": r"caja|blister|sellado",
        "t_reparar": r"reparar|detalle|repuesto|funciona"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, t.lower())) else 0.0 for t in titles], dtype=np.float32)
        
    ships = [r.get("shipping") or {} for r in rows]
    feats["ship_free"] = np.array([1.0 if s.get("free_shipping") else 0.0 for s in ships], dtype=np.float32)
    feats["ship_local_pick"] = np.array([1.0 if s.get("local_pick_up") else 0.0 for s in ships], dtype=np.float32)
    feats["automatic_relist"] = np.array([1.0 if r.get("automatic_relist") else 0.0 for r in rows], dtype=np.float32)
    feats["accepts_mercadopago"] = np.array([1.0 if r.get("accepts_mercadopago") else 0.0 for r in rows], dtype=np.float32)
    
    tags = [lst(r.get("tags")) for r in rows]
    feats["tag_dragged_bids_and_visits"] = np.array([1.0 if "dragged_bids_and_visits" in t else 0.0 for t in tags], dtype=np.float32)
    feats["tag_poor_quality_picture"] = np.array([1.0 if "poor_quality_picture" in t else 0.0 for t in tags], dtype=np.float32)
    feats["var_sold"] = np.array([float(sum((v.get("sold_quantity") or 0) for v in lst(r.get("variations")))) for r in rows], dtype=np.float32)

    cat_df = pd.DataFrame({
        "category_id": [str(r.get("category_id") or "") for r in rows],
        "listing_type_id": [str(r.get("listing_type_id") or "") for r in rows],
        "buying_mode": [str(r.get("buying_mode") or "") for r in rows],
        "state": [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "") for r in rows],
        "city": [str(((r.get("seller_address") or {}).get("city") or {}).get("name") or "").lower().strip() for r in rows],
        "seller_id": [str(r.get("seller_id") or "") for r in rows],
        "ship_mode": [str((r.get("shipping") or {}).get("mode") or "") for r in rows],
    })
    
    for col in ["seller_id", "category_id", "city"]:
        counts = cat_df[col].value_counts()
        feats[f"freq_{col}"] = cat_df[col].map(counts).fillna(1.0).values.astype(np.float32)
        
    df_num = pd.DataFrame(feats)
    df_num["seller_id"] = cat_df["seller_id"]
    df_num["category_id"] = cat_df["category_id"]
    
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
    
    cat_med = df_num.groupby("category_id", observed=True)["log_price"].transform("median")
    feats["rel_price_cat"] = (df_num["log_price"] - cat_med).values.astype(np.float32)
    
    F_df = pd.DataFrame(feats)
    for col in ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]:
        F_df[col] = cat_df[col].astype(str)
        
    print(f"Extracted R151 features {F_df.shape} in {time.time()-t0:.2f}s", flush=True)
    return F_df, cat_df

def compute_leak_free_te_r151(cat_df, fit, eval_indices_list, y_fit, sellers_fit, alpha=5.0):
    """Computes exact R151 target encoding with alpha=5."""
    prior = float(y_fit.mean())
    te_cols = ["seller_id", "category_id", "city"]
    gkf = GroupKFold(n_splits=4)
    
    fit_te = {c: np.full(len(fit), prior, dtype=np.float32) for c in te_cols}
    eval_tes = [{c: None for c in te_cols} for _ in eval_indices_list]
    
    for tr, va in gkf.split(fit, y_fit, sellers_fit):
        for c in te_cols:
            k_tr = cat_df[c].iloc[fit[tr]]
            k_va = cat_df[c].iloc[fit[va]]
            stats = pd.DataFrame({"k": k_tr, "y": y_fit[tr]}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
            fit_te[c][va] = k_va.map(smoothed).fillna(prior).values.astype(np.float32)
            
    for c in te_cols:
        k_fit = cat_df[c].iloc[fit]
        stats = pd.DataFrame({"k": k_fit, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
        smoothed = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
        for idx, eval_indices in enumerate(eval_indices_list):
            eval_tes[idx][c] = cat_df[c].iloc[eval_indices].map(smoothed).fillna(prior).values.astype(np.float32)
            
    return fit_te, eval_tes

def compute_txt_sparse_p(combos, titles, fit, eval_indices_list, y_fit, sellers_fit, seed=2026):
    """Computes leak-free txt_sparse_p via 4-fold GroupKFold on fit."""
    vec_combo = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000,
                                sublinear_tf=True, strip_accents="unicode", lowercase=True)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                               max_features=150000, sublinear_tf=True)
    
    X_w_fit = vec_combo.fit_transform([combos[i] for i in fit])
    X_c_fit = vec_char.fit_transform([titles[i] for i in fit])
    X_sp_fit = sp.hstack([X_w_fit, X_c_fit], format="csr")
    
    oof_sp = np.zeros(len(fit), dtype=np.float32)
    gkf = GroupKFold(n_splits=4)
    for tr, va in gkf.split(fit, y_fit, sellers_fit):
        lr_sp = LogisticRegression(C=3.0, max_iter=250, solver="liblinear", random_state=seed)
        lr_sp.fit(X_sp_fit[tr], y_fit[tr])
        oof_sp[va] = lr_sp.predict_proba(X_sp_fit[va])[:, 1]
        
    lr_full = LogisticRegression(C=3.0, max_iter=300, solver="liblinear", random_state=seed).fit(X_sp_fit, y_fit)
    eval_probs = []
    for eval_indices in eval_indices_list:
        X_w_ev = vec_combo.transform([combos[i] for i in eval_indices])
        X_c_ev = vec_char.transform([titles[i] for i in eval_indices])
        X_sp_ev = sp.hstack([X_w_ev, X_c_ev], format="csr")
        eval_probs.append(lr_full.predict_proba(X_sp_ev)[:, 1].astype(np.float32))
        
    return oof_sp, eval_probs
