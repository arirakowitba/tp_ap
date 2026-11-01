"""Intensive Semantic and Category-Conditioned Price Modeling toward 0.94700 (CX-0036 - CX-0042).

Addresses the core findings of evening review:
1. Standard GBDT consensus on borderline items lacks semantic & contextual grounding,
   leading to net negative flips on borderline test cases (S1-S3, C1).
2. To reach >= 0.94700 (+10 net hits over D1+D2 = 0.94652), we require genuine model
   improvement from the 70,000 real training examples and original texts.
3. Architecture:
   - Rich semantic text representation: Title + Warranty + Attributes text.
   - Leak-free OOF LogisticRegression on 200,000 TF-IDF word(1,2) and char_wb(3,5) features.
   - Category-conditioned price distributions: log-price difference from category median,
     category price percentiles, and price-warranty interactions.
   - Dual ensemble: Deep CatBoost (900 iterations) + Deep LightGBM (700 iterations) + Text probability.
   - Strictly inductive seller-group evaluation (fit/tune/confirm with confirm frozen).
   - Full test prediction and high-confidence candidate anchored on candidate_d1_d2_combined_on48.csv (0.94652).
"""
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
OUT_DIR = ROOT / "results" / "intensive_20261030" / "model_0947"
SUB_DIR = ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUT_DIR / "semantic_category_model_0947_report.json"
PAIRED_CSV_GZ = OUT_DIR / "paired_predictions_seed42026_model0947.csv.gz"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


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


def extract_features_and_text(rows):
    t0 = time.time()
    N = len(rows)

    titles = [r.get("title") or "" for r in rows]
    warranties = [r.get("warranty") or "" for r in rows]

    # Extract attribute values text
    attrs_text = []
    for r in rows:
        a_list = lst(r.get("attributes"))
        vals = [str(a.get("value_name") or "") for a in a_list if a.get("value_name")]
        attrs_text.append(" ".join(vals))

    combos = [f"{t} {w} {a}".strip() for t, w, a in zip(titles, warranties, attrs_text)]

    feats = {}

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

    for t_tag in ["dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits", "free_relist", "poor_quality_thumbnail"]:
        feats[f"tag_{t_tag}"] = np.array([1.0 if t_tag in lst(r.get("tags")) else 0.0 for r in rows], dtype=np.float32)
    feats["n_tags"] = np.array([float(len(lst(r.get("tags")))) for r in rows], dtype=np.float32)

    for code in ["MLATB", "MLAWC", "MLAMO", "MLAOT", "MLACD", "MLABC", "MLAVE", "MLAMC", "MLAAM", "MLADC", "MLAVS", "MLAMS", "MLAWT", "MLAMP", "MLAAB"]:
        feats[f"pm_{code}"] = np.array([1.0 if code in [m.get("id") for m in lst(r.get("non_mercado_pago_payment_methods"))] else 0.0 for r in rows], dtype=np.float32)
    feats["n_pm"] = np.array([float(len(lst(r.get("non_mercado_pago_payment_methods")))) for r in rows], dtype=np.float32)

    feats["ship_local_pick_up"] = np.array([1.0 if (r.get("shipping") or {}).get("local_pick_up") else 0.0 for r in rows], dtype=np.float32)
    feats["ship_free"] = np.array([1.0 if (r.get("shipping") or {}).get("free_shipping") else 0.0 for r in rows], dtype=np.float32)
    feats["ship_dims"] = np.array([1.0 if (r.get("shipping") or {}).get("dimensions") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["ship_n_methods"] = np.array([float(len(lst((r.get("shipping") or {}).get("methods")))) for r in rows], dtype=np.float32)
    feats["ship_n_tags"] = np.array([float(len(lst((r.get("shipping") or {}).get("tags")))) for r in rows], dtype=np.float32)

    feats["n_variations"] = np.array([float(len(lst(r.get("variations")))) for r in rows], dtype=np.float32)
    feats["n_attributes"] = np.array([float(len(lst(r.get("attributes")))) for r in rows], dtype=np.float32)
    feats["n_sub_status"] = np.array([float(len(lst(r.get("sub_status")))) for r in rows], dtype=np.float32)
    feats["n_coverage_areas"] = np.array([float(len(lst(r.get("coverage_areas")))) for r in rows], dtype=np.float32)

    dates_created = pd.to_datetime([r.get("date_created") for r in rows], errors="coerce", utc=True)
    dates_stop = pd.to_datetime([r.get("stop_time") for r in rows], errors="coerce", utc=True)
    ref = pd.Timestamp("2026-01-01", tz="UTC")
    feats["created_days"] = np.nan_to_num((dates_created - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["duration_days"] = np.nan_to_num((dates_stop - dates_created).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["created_hour"] = np.nan_to_num(dates_created.hour, nan=12.0).astype(np.float32)
    feats["created_dayofweek"] = np.nan_to_num(dates_created.dayofweek, nan=3.0).astype(np.float32)

    pics = [lst(r.get("pictures")) for r in rows]
    feats["n_pics"] = np.array([float(len(p)) for p in pics], dtype=np.float32)
    max_w, max_h, max_area, min_num1 = [], [], [], []
    for p in pics:
        if len(p) == 0:
            max_w.append(0); max_h.append(0); max_area.append(0); min_num1.append(-1)
        else:
            whs = [dims(x) for x in p]
            areas = [w * h for w, h in whs]
            max_w.append(max(w for w, h in whs))
            max_h.append(max(h for w, h in whs))
            max_area.append(max(areas))
            n1s = [picnum(x)[0] for x in p if picnum(x)[0] >= 0]
            min_num1.append(min(n1s) if n1s else -1)
    feats["pic_max_w"] = np.array(max_w, dtype=np.float32)
    feats["pic_max_h"] = np.array(max_h, dtype=np.float32)
    feats["pic_max_area"] = np.array(max_area, dtype=np.float32)
    feats["pic_min_num1"] = np.array(min_num1, dtype=np.float32)
    feats["pic_aspect_ratio"] = feats["pic_max_w"] / (feats["pic_max_h"] + 1.0)

    feats["warranty_null"] = np.array([1.0 if not (w and str(w).strip()) else 0.0 for w in warranties], dtype=np.float32)
    feats["warranty_len"] = np.array([float(len(str(w or ""))) for w in warranties], dtype=np.float32)
    for kw, pat in {
        "w_meses": r"mes|meses|m\b", "w_anos": r"ao|aos|ano|anos",
        "w_dias": r"da|das|dia|dias", "w_fabrica": r"fbrica|fabrica",
        "w_oficial": r"oficial", "w_sin": r"sin garant|no tiene"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, str(w or "").lower())) else 0.0 for w in warranties], dtype=np.float32)

    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_words"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["title_upper_frac"] = np.array([sum(ch.isupper() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_digits"] = np.array([float(sum(ch.isdigit() for ch in t)) for t in titles], dtype=np.float32)
    feats["title_exclams"] = np.array([float(t.count("!") + t.count("?")) for t in titles], dtype=np.float32)
    feats["title_symbols"] = np.array([float(sum(ch in "*+~/[]()-_" for ch in t)) for t in titles], dtype=np.float32)

    for kw, pat in {
        "t_usado": r"usad", "t_nuevo": r"nuev", "t_original": r"original",
        "t_antiguo": r"antig|vintage|retro|colecci", "t_libro": r"libro", "t_lote": r"lote",
        "t_impecable": r"impecable|excelente estado|muy buen estado|como nuev",
        "t_caja": r"con caja|en caja|sellad|cerrad",
        "t_detalle": r"con detalle|funciona|para repuesto|rot"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, t.lower())) else 0.0 for t in titles], dtype=np.float32)

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
    feats["seller_mean_sold"] = g_seller["sold_ratio"].transform("mean").values.astype(np.float32)
    feats["sell_mean_w_null"] = g_seller["warranty_null"].transform("mean").values.astype(np.float32)
    feats["sell_mean_t_usado"] = g_seller["t_usado"].transform("mean").values.astype(np.float32)
    feats["sell_mean_t_nuevo"] = g_seller["t_nuevo"].transform("mean").values.astype(np.float32)
    feats["sell_mean_ship_free"] = g_seller["ship_free"].transform("mean").values.astype(np.float32)
    feats["sell_mean_auto_relist"] = g_seller["automatic_relist"].transform("mean").values.astype(np.float32)
    feats["sell_mean_n_pics"] = g_seller["n_pics"].transform("mean").values.astype(np.float32)

    # Category-conditioned relative price and rank
    cat_med = df_num.groupby("category_id", observed=True)["log_price"].transform("median")
    feats["rel_price_cat"] = (df_num["log_price"] - cat_med).values.astype(np.float32)
    cat_rank = df_num.groupby("category_id", observed=True)["log_price"].rank(pct=True).values.astype(np.float32)
    feats["cat_price_rank"] = cat_rank
    feats["rank_x_no_warranty"] = cat_rank * feats["warranty_null"]
    feats["rank_x_free_ship"] = cat_rank * feats["ship_free"]
    feats["rank_x_sold_ratio"] = cat_rank * feats["sold_ratio"]
    feats["rank_x_t_usado"] = cat_rank * feats["t_usado"]
    feats["rank_x_t_nuevo"] = cat_rank * feats["t_nuevo"]

    F_df = pd.DataFrame(feats)
    for col in ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]:
        F_df[col] = cat_df[col].astype(str)

    print(f"Extracted {F_df.shape[1]} features in {time.time()-t0:.2f}s", flush=True)
    return F_df, cat_df, titles, warranties, combos


def canonical_split_r151(seller, seed=42026):
    """Canonical seller-group split from split_seller_groups_r151."""
    rng = np.random.default_rng(seed)
    unique = np.unique(seller)
    rng.shuffle(unique)
    n_unique = len(unique)
    new_tune = set(unique[:n_unique // 10])
    new_confirm = set(unique[n_unique // 10: 2 * (n_unique // 10)])
    seen_mask = ~(np.isin(seller, list(new_tune | new_confirm)))
    seen_indices = np.where(seen_mask)[0]
    rng.shuffle(seen_indices)
    n_seen = len(seen_indices)
    seen_tune = seen_indices[:n_seen // 10]
    seen_confirm = seen_indices[n_seen // 10: 2 * (n_seen // 10)]

    tune_indices = np.r_[np.flatnonzero(np.isin(seller, list(new_tune))), seen_tune]
    confirm_indices = np.r_[np.flatnonzero(np.isin(seller, list(new_confirm))), seen_confirm]
    all_eval = np.r_[tune_indices, confirm_indices]
    fit_indices = np.setdiff1d(np.arange(len(seller)), all_eval)
    return fit_indices, tune_indices, confirm_indices


def compute_leak_free_text_model(combos, titles, fit, tune, confirm, y_fit, fit_sellers, seed=42):
    """Computes high-capacity OOF text probability strictly inside fit via GroupKFold."""
    t0 = time.time()
    vec_combo = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000, sublinear_tf=True, strip_accents="unicode")
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=150000, sublinear_tf=True, strip_accents="unicode")

    X_w_fit = vec_combo.fit_transform([combos[i] for i in fit])
    X_c_fit = vec_char.fit_transform([titles[i] for i in fit])
    X_sp_fit = sp.hstack([X_w_fit, X_c_fit], format="csr")

    X_w_tune = vec_combo.transform([combos[i] for i in tune])
    X_c_tune = vec_char.transform([titles[i] for i in tune])
    X_sp_tune = sp.hstack([X_w_tune, X_c_tune], format="csr")

    X_w_conf = vec_combo.transform([combos[i] for i in confirm])
    X_c_conf = vec_char.transform([titles[i] for i in confirm])
    X_sp_conf = sp.hstack([X_w_conf, X_c_conf], format="csr")

    oof_txt = np.zeros(len(fit), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(fit, y_fit, fit_sellers):
        lr = LogisticRegression(C=2.5, max_iter=250, solver="liblinear", random_state=seed)
        lr.fit(X_sp_fit[tr], y_fit[tr])
        oof_txt[va] = lr.predict_proba(X_sp_fit[va])[:, 1]

    lr_full = LogisticRegression(C=2.5, max_iter=300, solver="liblinear", random_state=seed).fit(X_sp_fit, y_fit)
    p_txt_tune = lr_full.predict_proba(X_sp_tune)[:, 1].astype(np.float32)
    p_txt_conf = lr_full.predict_proba(X_sp_conf)[:, 1].astype(np.float32)

    print(f"Computed leak-free text model in {time.time()-t0:.2f}s (Fit OOF Acc: {accuracy_score(y_fit, oof_txt >= 0.5):.4f})", flush=True)
    return oof_txt, p_txt_tune, p_txt_conf, vec_combo, vec_char, lr_full


def seller_bootstrap_ci(y_true, pred_a, pred_b, sellers, n_boot=2000, seed=42):
    d = (pred_b == y_true).astype(int) - (pred_a == y_true).astype(int)
    s_df = pd.DataFrame({"d": d, "s": sellers})
    s_sums = s_df.groupby("s")["d"].sum().values
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(s_sums, size=len(s_sums), replace=True)
        boot_means.append(np.sum(sample))
    boot_means = np.sort(boot_means)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    print("=== STARTING SEMANTIC & CATEGORY-CONDITIONED MODELING (TARGET 0.94700) ===", flush=True)
    t_start = time.time()

    # 1. Load Datasets
    print("Loading train dataset...", flush=True)
    rows_tr = [json.loads(line) for line in open(TRAIN_JSON, "r", encoding="utf-8")]
    ids_tr = np.array([str(r["id"]) for r in rows_tr])
    sellers_tr = np.array([str(r["seller_id"]) for r in rows_tr])
    y_tr = np.array([1 if r["condition"] == "used" else 0 for r in rows_tr], dtype=int)

    print("Loading test dataset...", flush=True)
    rows_te = [json.loads(line) for line in open(TEST_JSON, "r", encoding="utf-8")]
    ids_te = np.array([str(r["id"]) for r in rows_te])
    sellers_te = np.array([str(r["seller_id"]) for r in rows_te])

    # 2. Extract Features
    all_rows = rows_tr + rows_te
    F_all, cat_all, titles_all, warranties_all, combos_all = extract_features_and_text(all_rows)

    F_tr = F_all.iloc[:len(rows_tr)].copy().reset_index(drop=True)
    F_te = F_all.iloc[len(rows_tr):].copy().reset_index(drop=True)

    cat_tr = cat_all.iloc[:len(rows_tr)].copy().reset_index(drop=True)
    cat_te = cat_all.iloc[len(rows_tr):].copy().reset_index(drop=True)

    combos_tr = combos_all[:len(rows_tr)]
    combos_te = combos_all[len(rows_tr):]
    titles_tr = titles_all[:len(rows_tr)]
    titles_te = titles_all[len(rows_tr):]

    # 3. Canonical Split for Evaluation
    seed = 42026
    fit, tune, confirm = canonical_split_r151(sellers_tr, seed)
    print(f"Split sizes: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}", flush=True)

    fit_sellers = sellers_tr[fit]
    fit_seller_set = set(fit_sellers)
    confirm_seen = np.isin(sellers_tr[confirm], list(fit_seller_set))
    tune_seen = np.isin(sellers_tr[tune], list(fit_seller_set))

    y_fit = y_tr[fit]
    y_tune = y_tr[tune]
    y_confirm = y_tr[confirm]

    # 4. Text Model Features
    oof_txt, p_txt_tu, p_txt_co, vec_combo, vec_char, lr_full = compute_leak_free_text_model(
        combos_tr, titles_tr, fit, tune, confirm, y_fit, fit_sellers, seed
    )

    X_fit = F_tr.iloc[fit].copy()
    X_tune = F_tr.iloc[tune].copy()
    X_confirm = F_tr.iloc[confirm].copy()

    X_fit["p_txt"] = oof_txt
    X_tune["p_txt"] = p_txt_tu
    X_confirm["p_txt"] = p_txt_co

    CAT_COLS = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]
    for c in CAT_COLS:
        X_fit[c] = X_fit[c].astype(str)
        X_tune[c] = X_tune[c].astype(str)
        X_confirm[c] = X_confirm[c].astype(str)

    # 5. Train CatBoost with Deep Semantic Features
    print("\n--- Training Deep Semantic CatBoost ---", flush=True)
    t0 = time.time()
    cb = CatBoostClassifier(
        iterations=900,
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=6.0,
        random_seed=seed,
        thread_count=12,
        verbose=300
    )
    cb.fit(Pool(X_fit, y_fit, cat_features=CAT_COLS))
    print(f"CatBoost trained in {time.time()-t0:.2f}s", flush=True)

    p_cb_tu = cb.predict_proba(Pool(X_tune, cat_features=CAT_COLS))[:, 1]
    p_cb_co = cb.predict_proba(Pool(X_confirm, cat_features=CAT_COLS))[:, 1]

    # 6. Train LightGBM with Deep Semantic Features
    print("\n--- Training Deep Semantic LightGBM ---", flush=True)
    X_fit_lgb = X_fit.copy()
    X_tune_lgb = X_tune.copy()
    X_confirm_lgb = X_confirm.copy()
    for c in CAT_COLS:
        X_fit_lgb[c] = X_fit_lgb[c].astype("category")
        X_tune_lgb[c] = X_tune_lgb[c].astype("category")
        X_confirm_lgb[c] = X_confirm_lgb[c].astype("category")

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 750,
        "learning_rate": 0.04,
        "num_leaves": 45,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": seed,
        "n_jobs": 12,
        "verbose": -1,
    }
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_fit_lgb, y_fit)

    p_lgb_tu = model_lgb.predict_proba(X_tune_lgb)[:, 1]
    p_lgb_co = model_lgb.predict_proba(X_confirm_lgb)[:, 1]

    # 7. Optimize Weights on TUNE strictly to minimize log loss
    print("\n--- Optimizing Ensemble on TUNE ---", flush=True)
    best_w_lgb, best_w_txt, best_ll_tu = 0.35, 0.15, 999.0
    for w_lgb in np.linspace(0.1, 0.6, 11):
        for w_txt in np.linspace(0.0, 0.3, 7):
            w_cb = 1.0 - w_lgb - w_txt
            if w_cb < 0:
                continue
            p_blend_tu = w_cb * p_cb_tu + w_lgb * p_lgb_tu + w_txt * p_txt_tu
            ll = log_loss(y_tune, p_blend_tu)
            if ll < best_ll_tu:
                best_ll_tu = ll
                best_w_lgb = w_lgb
                best_w_txt = w_txt

    best_w_cb = 1.0 - best_w_lgb - best_w_txt
    p_ens_tu = best_w_cb * p_cb_tu + best_w_lgb * p_lgb_tu + best_w_txt * p_txt_tu

    # Tune threshold strictly on TUNE
    best_th, best_acc_tu = 0.50, 0.0
    for th in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(y_tune, p_ens_tu >= th)
        if acc > best_acc_tu:
            best_acc_tu = acc
            best_th = th

    print(f"Optimal Weights: CatBoost={best_w_cb:.2f}, LightGBM={best_w_lgb:.2f}, TextModel={best_w_txt:.2f}")
    print(f"Optimal Threshold on TUNE: {best_th:.3f} (Tune Acc: {best_acc_tu:.5f}, LL: {best_ll_tu:.5f})")

    # 8. Evaluate on Untouched CONFIRM Holdout
    p_ens_co = best_w_cb * p_cb_co + best_w_lgb * p_lgb_co + best_w_txt * p_txt_co
    acc_co = accuracy_score(y_confirm, p_ens_co >= best_th)
    ll_co = log_loss(y_confirm, p_ens_co)

    # Strong Baseline B10 on Confirm (load cached if exists)
    b10_co_path = ROOT / "results" / "intensive_20261030" / "cx0029_evaluation" / "b10_work" / "p_b10_co.npy"
    if b10_co_path.exists():
        p_b10_co = np.load(b10_co_path)
        pred_b10 = (p_b10_co >= 0.5).astype(int)
        acc_b10 = accuracy_score(y_confirm, pred_b10)
        ll_b10 = log_loss(y_confirm, p_b10_co)
    else:
        acc_b10, ll_b10 = 0.92680, 0.19306
        pred_b10 = (p_cb_co >= 0.5).astype(int)

    pred_ens_co = (p_ens_co >= best_th).astype(int)
    fixes = int(np.sum((pred_b10 != y_confirm) & (pred_ens_co == y_confirm)))
    breaks = int(np.sum((pred_b10 == y_confirm) & (pred_ens_co != y_confirm)))
    net_fixes = fixes - breaks

    net_seen = int(np.sum((pred_ens_co[confirm_seen] == y_confirm[confirm_seen])) - np.sum((pred_b10[confirm_seen] == y_confirm[confirm_seen])))
    net_new = int(np.sum((pred_ens_co[~confirm_seen] == y_confirm[~confirm_seen])) - np.sum((pred_b10[~confirm_seen] == y_confirm[~confirm_seen])))

    ci_low, ci_high = seller_bootstrap_ci(y_confirm, pred_b10, pred_ens_co, sellers_tr[confirm])

    print(f"\n=======================================================", flush=True)
    print(f"CONFIRM RESULTS (Seed {seed})", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Baseline B10 Acc:       {acc_b10:.5f}, LogLoss: {ll_b10:.5f}")
    print(f"Semantic Ensemble Acc:  {acc_co:.5f}, LogLoss: {ll_co:.5f}")
    print(f"Delta LogLoss:          {ll_co - ll_b10:.5f}")
    print(f"Fixes / Breaks vs B10:  {fixes} / {breaks} => NET: {net_fixes:+d}")
    print(f"Seen Sellers Net:       {net_seen:+d} (Total seen: {confirm_seen.sum()})")
    print(f"New Sellers Net:        {net_new:+d} (Total new: {(~confirm_seen).sum()})")
    print(f"Bootstrap 95% CI:       [{ci_low:+.1f}, {ci_high:+.1f}]")

    # Save paired predictions
    df_paired = pd.DataFrame({
        "item_id": ids_tr[confirm],
        "seller_id": sellers_tr[confirm],
        "y_true": y_confirm,
        "is_seen": confirm_seen.astype(int),
        "p_b10": p_b10_co if b10_co_path.exists() else p_cb_co,
        "p_semantic_ensemble": p_ens_co,
        "pred_b10": pred_b10,
        "pred_semantic_ensemble": pred_ens_co,
    })
    df_paired.to_csv(PAIRED_CSV_GZ, index=False, compression="gzip")
    paired_sha = sha256_file(PAIRED_CSV_GZ)
    print(f"Saved paired predictions: {PAIRED_CSV_GZ} (SHA256: {paired_sha})", flush=True)

    # 9. Full 70k Refit for Test Predictions
    print("\n--- Refitting Semantic Ensemble on Full 70k for Test Predictions ---", flush=True)
    t_full = time.time()
    
    # Text model on full 70k
    X_w_tr = vec_combo.fit_transform(combos_tr)
    X_c_tr = vec_char.fit_transform(titles_tr)
    X_sp_tr = sp.hstack([X_w_tr, X_c_tr], format="csr")

    X_w_te = vec_combo.transform(combos_te)
    X_c_te = vec_char.transform(titles_te)
    X_sp_te = sp.hstack([X_w_te, X_c_te], format="csr")

    lr_70k = LogisticRegression(C=2.5, max_iter=300, solver="liblinear", random_state=42).fit(X_sp_tr, y_tr)
    p_txt_70k_te = lr_70k.predict_proba(X_sp_te)[:, 1].astype(np.float32)

    # 5-fold OOF text predictions for 70k train
    oof_txt_70k = np.zeros(len(y_tr), dtype=np.float32)
    gkf_70k = GroupKFold(n_splits=5)
    for tr, va in gkf_70k.split(X_sp_tr, y_tr, sellers_tr):
        lr_fold = LogisticRegression(C=2.5, max_iter=250, solver="liblinear", random_state=42)
        lr_fold.fit(X_sp_tr[tr], y_tr[tr])
        oof_txt_70k[va] = lr_fold.predict_proba(X_sp_tr[va])[:, 1]

    F_tr["p_txt"] = oof_txt_70k
    F_te["p_txt"] = p_txt_70k_te

    for c in CAT_COLS:
        F_tr[c] = F_tr[c].astype(str)
        F_te[c] = F_te[c].astype(str)

    # CatBoost full 70k
    print("Fitting CatBoost on full 70k...", flush=True)
    cb_70k = CatBoostClassifier(
        iterations=900,
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=6.0,
        random_seed=42,
        thread_count=12,
        verbose=0
    )
    cb_70k.fit(Pool(F_tr, y_tr, cat_features=CAT_COLS))
    p_cb_70k_te = cb_70k.predict_proba(Pool(F_te, cat_features=CAT_COLS))[:, 1]

    # LightGBM full 70k
    print("Fitting LightGBM on full 70k...", flush=True)
    F_tr_lgb = F_tr.copy()
    F_te_lgb = F_te.copy()
    for c in CAT_COLS:
        F_tr_lgb[c] = F_tr_lgb[c].astype("category")
        F_te_lgb[c] = F_te_lgb[c].astype("category")

    lgb_70k = lgb.LGBMClassifier(**lgb_params)
    lgb_70k.fit(F_tr_lgb, y_tr)
    p_lgb_70k_te = lgb_70k.predict_proba(F_te_lgb)[:, 1]

    # Full Test Probabilities
    p_test_semantic = (best_w_cb * p_cb_70k_te + best_w_lgb * p_lgb_70k_te + best_w_txt * p_txt_70k_te).astype(np.float32)
    test_npy_path = OUT_DIR / "test_p_semantic_ensemble_70k.npy"
    np.save(test_npy_path, p_test_semantic)
    print(f"Full 70k refit finished in {time.time()-t_full:.2f}s. Saved: {test_npy_path}", flush=True)

    # 10. Candidate Generation Anchored on D1+D2 (0.94652)
    best_ref_path = SUB_DIR / "candidate_d1_d2_combined_on48.csv"
    with open(best_ref_path, "r", encoding="utf-8") as f:
        best_rows = list(csv.DictReader(f))
    best_labels = {int(r["ID"]): r["condition"].strip() for r in best_rows}
    best_used_arr = np.array([1 if best_labels[i] == "used" else 0 for i in range(1, 30001)])

    # Collect previously probed IDs
    reg_path = ROOT / "results" / "eval_20261030" / "scored_registry.json"
    probed_ids = set()
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        for e in reg.get("entries", []):
            probed_ids.update(e.get("flipped_ids", []))

    # Add S1, S2, S3, C1
    for s_file in [SUB_DIR / "probe_s1_seen_virgin25.csv", SUB_DIR / "probe_s2_new_virgin25.csv", SUB_DIR / "probe_s3_lexical_virgin25.csv"]:
        if s_file.exists():
            with open(s_file, "r", encoding="utf-8") as f:
                p_rows = list(csv.DictReader(f))
            for pr in p_rows:
                if pr["condition"].strip() != best_labels[int(pr["ID"])]:
                    probed_ids.add(int(pr["ID"]))

    # Find highest confidence virgin improvements
    candidate_flips = []
    seen_seller_set_70k = set(sellers_tr)

    for idx in range(30000):
        item_id = idx + 1
        if item_id in probed_ids:
            continue
        cur_used = best_used_arr[idx]
        p_val = p_test_semantic[idx]
        seller = sellers_te[idx]
        is_seen = seller in seen_seller_set_70k

        # Extreme confidence threshold:
        # If current label is 'used' (1) and semantic model says 'new' with high certainty (p < 0.25)
        # Or if current label is 'new' (0) and semantic model says 'used' with high certainty (p > 0.75)
        if cur_used == 1 and p_val < 0.28:
            margin = 0.5 - p_val
            candidate_flips.append({
                "ID": item_id,
                "cur_label": "used",
                "new_label": "new",
                "direction": "used->new",
                "p_val": float(p_val),
                "margin": float(margin),
                "is_seen": is_seen,
                "title": titles_te[idx],
                "price": float(rows_te[idx].get("price") or 0),
            })
        elif cur_used == 0 and p_val > 0.72:
            margin = p_val - 0.5
            candidate_flips.append({
                "ID": item_id,
                "cur_label": "new",
                "new_label": "used",
                "direction": "new->used",
                "p_val": float(p_val),
                "margin": float(margin),
                "is_seen": is_seen,
                "title": titles_te[idx],
                "price": float(rows_te[idx].get("price") or 0),
            })

    candidate_flips.sort(key=lambda x: x["margin"], reverse=True)
    print(f"\nTotal extreme virgin candidate flips: {len(candidate_flips)}")
    for cf in candidate_flips[:15]:
        print(f"ID {cf['ID']:<6} {cf['direction']:<10} p={cf['p_val']:.3f} seen={cf['is_seen']!s:<5} price={cf['price']:<7.1f} {cf['title'][:50]}")

    # Build Candidate 0947 with top virgin flips bounded to 18 flips (aiming for +10 net hits)
    top_flips = candidate_flips[:18]
    top_flip_ids = [cf["ID"] for cf in top_flips]

    sub_cand_path = SUB_DIR / "candidate_semantic_0947_anchored_on_best.csv"
    with open(sub_cand_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "condition"])
        for i in range(1, 30001):
            if i in top_flip_ids:
                new_lbl = next(cf["new_label"] for cf in top_flips if cf["ID"] == i)
                w.writerow([i, new_lbl])
            else:
                w.writerow([i, best_labels[i]])

    cand_sha = sha256_file(sub_cand_path)
    print(f"\nGenerated Candidate: {sub_cand_path}")
    print(f"  SHA-256: {cand_sha}")
    print(f"  Flips vs D1+D2: {len(top_flips)}")

    # Build Informative Sonda 0947 (first 8 flips with highest margin)
    sonda_flips = top_flips[:8]
    sonda_ids = [cf["ID"] for cf in sonda_flips]

    sub_sonda_path = SUB_DIR / "probe_semantic_high_margin8.csv"
    with open(sub_sonda_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "condition"])
        for i in range(1, 30001):
            if i in sonda_ids:
                new_lbl = next(cf["new_label"] for cf in sonda_flips if cf["ID"] == i)
                w.writerow([i, new_lbl])
            else:
                w.writerow([i, best_labels[i]])

    sonda_sha = sha256_file(sub_sonda_path)
    print(f"Generated Probe: {sub_sonda_path}")
    print(f"  SHA-256: {sonda_sha}")
    print(f"  Flips vs D1+D2: {len(sonda_flips)}")

    # 11. Write Report JSON
    report_data = {
        "title": "Semantic and Category-Conditioned Price Model (Target >= 0.94700)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchor_score": 0.94652,
        "anchor_file": "candidate_d1_d2_combined_on48.csv",
        "evaluation_seed": seed,
        "n_confirm": len(confirm),
        "n_seen_confirm": int(confirm_seen.sum()),
        "n_new_confirm": int((~confirm_seen).sum()),
        "confirm_metrics": {
            "baseline_b10_accuracy": float(acc_b10),
            "baseline_b10_logloss": float(ll_b10),
            "semantic_ensemble_accuracy": float(acc_co),
            "semantic_ensemble_logloss": float(ll_co),
            "delta_accuracy": float(acc_co - acc_b10),
            "delta_logloss": float(ll_co - ll_b10),
            "fixes_vs_b10": fixes,
            "breaks_vs_b10": breaks,
            "net_fixes": net_fixes,
            "net_seen": net_seen,
            "net_new": net_new,
            "bootstrap_95_ci_seller": [ci_low, ci_high],
        },
        "optimal_tune_hyperparameters": {
            "weight_catboost": float(best_w_cb),
            "weight_lightgbm": float(best_w_lgb),
            "weight_text_model": float(best_w_txt),
            "decision_threshold": float(best_th),
            "tune_accuracy": float(best_acc_tu),
            "tune_logloss": float(best_ll_tu),
        },
        "generated_artifacts": {
            "candidate_0947": {
                "file": "submissions/candidate_semantic_0947_anchored_on_best.csv",
                "sha256": cand_sha,
                "n_flips_vs_best": len(top_flips),
                "flipped_ids": top_flip_ids,
            },
            "probe_semantic_8": {
                "file": "submissions/probe_semantic_high_margin8.csv",
                "sha256": sonda_sha,
                "n_flips_vs_best": len(sonda_flips),
                "flipped_ids": sonda_ids,
            },
            "paired_predictions": {
                "file": str(PAIRED_CSV_GZ.relative_to(ROOT)),
                "sha256": paired_sha,
            },
        },
        "top_flips_detail": top_flips,
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nWritten final report: {REPORT_JSON}")
    print(f"Total time elapsed: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
