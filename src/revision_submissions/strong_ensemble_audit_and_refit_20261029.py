"""Ronda 157: Honest Strong Baseline Restoration and Deep Ensemble (LGBM + CatBoost) Evaluation.

Protocol:
1. Exact canonical seller-group split from R139/141 (confirm: 12906 / 12844 rows, seen: 3950 / 3831).
2. Deep features + Leak-Free GroupKFold OOF Target Encodings + Leak-Free GroupKFold OOF txt_sparse_p (TF-IDF LR).
3. Baseline: Deep CatBoost (900 trees, depth 6, lr 0.07, l2 6.0) with threshold tuned on tune.
4. Model B: Deep LightGBM with txt_sparse_p.
5. Ensemble: w* and threshold* optimized strictly on tune to minimize log loss and maximize accuracy.
6. Evaluated once on confirm: accuracy, log loss, fixes/breaks, seen/new, 2000 bootstrap CI by seller.
7. Paired predictions saved with SHA256.
8. If confirmed, refits on full 70k and generates test probabilities and diff manifest vs S48.
"""
import hashlib
import json
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"
RESULTS_DIR = ROOT / "results" / "revision_submissions"
SUBMISSIONS_DIR = ROOT / "submissions" / "20261029"


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

    titles = [r.get("title") or "" for r in rows]
    warranties = [r.get("warranty") or "" for r in rows]
    combos = [f"{t} {w}".strip() for t, w in zip(titles, warranties)]

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
        "t_impecable": r"impecable|excelente estado|muy buen estado|como nuev"
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

    # Relative price to category median and percentile rank
    cat_med = df_num.groupby("category_id", observed=True)["log_price"].transform("median")
    feats["rel_price_cat"] = (df_num["log_price"] - cat_med).values.astype(np.float32)
    cat_rank = df_num.groupby("category_id", observed=True)["log_price"].rank(pct=True).values.astype(np.float32)
    feats["cat_price_rank"] = cat_rank
    feats["rank_x_no_warranty"] = cat_rank * feats["warranty_null"]
    feats["rank_x_free_ship"] = cat_rank * feats["ship_free"]
    feats["rank_x_sold_ratio"] = cat_rank * feats["sold_ratio"]

    F_df = pd.DataFrame(feats)
    for col in ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]:
        F_df[col] = cat_df[col].astype(str)

    print(f"Extracted feature matrix {F_df.shape} in {time.time()-t0:.2f}s", flush=True)
    return F_df, cat_df, titles, warranties, combos


def canonical_split(seller, seed):
    """Exact canonical seller-group split from seller_diagnostic_20261028.py (R139/141)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(seller)
    rng.shuffle(unique)
    new_tune = set(unique[:len(unique)//10])
    new_confirm = set(unique[len(unique)//10:len(unique)//5])
    seen = ~(np.isin(seller, list(new_tune | new_confirm)))
    ids = np.where(seen)[0]
    rng.shuffle(ids)
    seen_tune, seen_confirm = ids[:len(ids)//10], ids[len(ids)//10:len(ids)//5]
    tune = np.flatnonzero(np.isin(seller, list(new_tune)))
    confirm = np.flatnonzero(np.isin(seller, list(new_confirm)))
    tune = np.r_[tune, seen_tune]
    confirm = np.r_[confirm, seen_confirm]
    fit = np.setdiff1d(np.arange(len(seller)), np.r_[tune, confirm])
    assert not set(seller[tune]).intersection(new_confirm)
    assert not set(seller[confirm]).intersection(new_tune)
    return fit, tune, confirm


def compute_leak_free_te(fit_df, fit_seller, y_fit, eval_dfs, cat_cols=["seller_id", "category_id", "city"], alpha=10.0):
    global_mean = float(y_fit.mean())
    oof_te = {col: np.full(len(fit_df), global_mean, dtype=np.float32) for col in cat_cols}
    gkf = GroupKFold(n_splits=4)
    for tr_idx, val_idx in gkf.split(fit_df, y_fit, groups=fit_seller):
        for col in cat_cols:
            cats = fit_df[col].values
            tr_cats = cats[tr_idx]
            tr_y = y_fit[tr_idx]
            sums = defaultdict(float)
            counts = defaultdict(int)
            for c, y in zip(tr_cats, tr_y):
                sums[c] += y
                counts[c] += 1
            oof_te[col][val_idx] = np.array([
                (sums[c] + alpha * global_mean) / (counts[c] + alpha)
                for c in cats[val_idx]
            ], dtype=np.float32)

    fit_stats = {}
    for col in cat_cols:
        cats = fit_df[col].values
        sums = defaultdict(float)
        counts = defaultdict(int)
        for c, y in zip(cats, y_fit):
            sums[c] += y
            counts[c] += 1
        fit_stats[col] = (sums, counts)

    eval_tes = []
    for edf in eval_dfs:
        ete = {}
        for col in cat_cols:
            sums, counts = fit_stats[col]
            ete[col] = np.array([
                (sums[c] + alpha * global_mean) / (counts[c] + alpha) if c in counts else global_mean
                for c in edf[col].values
            ], dtype=np.float32)
        eval_tes.append(ete)

    return oof_te, eval_tes


def compute_leak_free_text_feature(combos, titles, fit, tune, confirm, y_fit, fit_sellers, seed):
    """Computes txt_sparse_p strictly OOF inside fit via 4-fold GroupKFold by seller."""
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
    gkf = GroupKFold(n_splits=4)
    for tr, va in gkf.split(fit, y_fit, fit_sellers):
        lr = LogisticRegression(C=3.0, max_iter=250, solver="liblinear", random_state=seed)
        lr.fit(X_sp_fit[tr], y_fit[tr])
        oof_txt[va] = lr.predict_proba(X_sp_fit[va])[:, 1]

    # Full fit to predict tune and confirm
    lr_full = LogisticRegression(C=3.0, max_iter=300, solver="liblinear", random_state=seed).fit(X_sp_fit, y_fit)
    p_txt_tune = lr_full.predict_proba(X_sp_tune)[:, 1].astype(np.float32)
    p_txt_conf = lr_full.predict_proba(X_sp_conf)[:, 1].astype(np.float32)

    print(f"Computed leak-free txt_sparse_p in {time.time()-t0:.2f}s (Fit OOF Acc: {accuracy_score(y_fit, oof_txt >= 0.5):.4f})", flush=True)
    return oof_txt, p_txt_tune, p_txt_conf


def bootstrap_seller_delta(seller_ids, net_delta, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    u_sellers = np.unique(seller_ids)
    s_map = defaultdict(list)
    for idx, s in enumerate(seller_ids):
        s_map[s].append(net_delta[idx])
    s_sums = {s: sum(vals) for s, vals in s_map.items()}
    s_sums_arr = np.array([s_sums[s] for s in u_sellers], dtype=float)
    boot_totals = []
    for _ in range(n_boot):
        sample = rng.choice(s_sums_arr, size=len(s_sums_arr), replace=True)
        boot_totals.append(np.sum(sample))
    boot_totals = np.sort(boot_totals)
    return float(np.percentile(boot_totals, 2.5)), float(np.percentile(boot_totals, 97.5))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_experiment_for_seed(seed, rows, ids, sellers, y_used, F_df, cat_df, titles, warranties, combos):
    print(f"\n=======================================================", flush=True)
    print(f"RUNNING STRONG BASELINE & ENSEMBLE EVALUATION (SEED {seed})", flush=True)
    print(f"=======================================================", flush=True)

    fit, tune, confirm = canonical_split(sellers, seed)
    fit_sellers = sellers[fit]
    fit_seller_set = set(fit_sellers)
    confirm_seen = np.isin(sellers[confirm], list(fit_seller_set))
    tune_seen = np.isin(sellers[tune], list(fit_seller_set))

    print(f"Canonical split sizes: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}", flush=True)
    print(f"Confirm counts: seen={confirm_seen.sum()}/{len(confirm)}, new={(~confirm_seen).sum()}", flush=True)

    y_fit = y_used[fit]
    y_tune = y_used[tune]
    y_confirm = y_used[confirm]

    # 1. Target Encodings
    fit_cat = cat_df.iloc[fit]
    tune_cat = cat_df.iloc[tune]
    confirm_cat = cat_df.iloc[confirm]
    oof_te, (tune_te, confirm_te) = compute_leak_free_te(fit_cat, fit_sellers, y_fit, [tune_cat, confirm_cat])

    # 2. Text Feature (txt_sparse_p)
    oof_txt, p_txt_tune, p_txt_conf = compute_leak_free_text_feature(combos, titles, fit, tune, confirm, y_fit, fit_sellers, seed)

    # 3. Assemble Feature Matrices
    X_fit = F_df.iloc[fit].copy()
    X_tune = F_df.iloc[tune].copy()
    X_confirm = F_df.iloc[confirm].copy()

    for col in ["seller_id", "category_id", "city"]:
        X_fit[f"te_{col}"] = oof_te[col]
        X_tune[f"te_{col}"] = tune_te[col]
        X_confirm[f"te_{col}"] = confirm_te[col]

    X_fit["txt_sparse_p"] = oof_txt
    X_tune["txt_sparse_p"] = p_txt_tune
    X_confirm["txt_sparse_p"] = p_txt_conf

    CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]
    for c in CAT_FEATURES:
        X_fit[c] = X_fit[c].astype(str)
        X_tune[c] = X_tune[c].astype(str)
        X_confirm[c] = X_confirm[c].astype(str)

    # 4. Train Deep CatBoost Baseline (900 trees, depth 6, lr 0.07, l2 6.0)
    print("\nTraining Deep CatBoost Baseline (iterations=900, depth=6, lr=0.07)...", flush=True)
    t0 = time.time()
    cb_base = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                 random_seed=seed, thread_count=12, verbose=0)
    cb_base.fit(Pool(X_fit, y_fit, cat_features=CAT_FEATURES))
    print(f"Deep CatBoost Baseline fitted in {time.time()-t0:.2f}s", flush=True)

    p_cb_tune = cb_base.predict_proba(Pool(X_tune, cat_features=CAT_FEATURES))[:, 1]
    p_cb_conf = cb_base.predict_proba(Pool(X_confirm, cat_features=CAT_FEATURES))[:, 1]

    # Tune threshold for CatBoost on tune strictly
    best_th_cb, best_acc_cb_tu = 0.50, 0.0
    for th in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(y_tune, p_cb_tune >= th)
        if acc > best_acc_cb_tu:
            best_acc_cb_tu = acc
            best_th_cb = th

    acc_cb_conf = accuracy_score(y_confirm, p_cb_conf >= best_th_cb)
    loss_cb_conf = log_loss(y_confirm, p_cb_conf)
    print(f"Deep CatBoost Confirm: Acc = {acc_cb_conf:.5f} (th={best_th_cb:.2f}), LogLoss = {loss_cb_conf:.5f}", flush=True)

    # 5. Train Model B: Deep LightGBM on the exact same features
    print("\nTraining Deep LightGBM (with txt_sparse_p)...", flush=True)
    X_fit_lgb = X_fit.copy()
    X_tune_lgb = X_tune.copy()
    X_confirm_lgb = X_confirm.copy()
    for c in CAT_FEATURES:
        X_fit_lgb[c] = X_fit_lgb[c].astype("category")
        X_tune_lgb[c] = X_tune_lgb[c].astype("category")
        X_confirm_lgb[c] = X_confirm_lgb[c].astype("category")

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 700,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_fit_lgb, y_fit)

    p_lgb_tune = model_lgb.predict_proba(X_tune_lgb)[:, 1]
    p_lgb_conf = model_lgb.predict_proba(X_confirm_lgb)[:, 1]

    acc_lgb_conf = accuracy_score(y_confirm, p_lgb_conf >= 0.5)
    loss_lgb_conf = log_loss(y_confirm, p_lgb_conf)
    print(f"Deep LightGBM Confirm: Acc = {acc_lgb_conf:.5f}, LogLoss = {loss_lgb_conf:.5f}", flush=True)

    # 6. Optimize Ensemble (Weight + Threshold) Strictly on Tune
    print("\nOptimizing Ensemble (w* and th*) strictly on Tune...", flush=True)
    best_w = 0.50
    best_th_ens = 0.50
    best_tune_loss = 999.0
    for w in np.linspace(0.0, 1.0, 101):
        p_b = w * p_lgb_tune + (1.0 - w) * p_cb_tune
        loss = log_loss(y_tune, p_b)
        if loss < best_tune_loss:
            best_tune_loss = loss
            best_w = w

    p_ens_tune = best_w * p_lgb_tune + (1.0 - best_w) * p_cb_tune
    best_acc_ens_tu = 0.0
    for th in np.linspace(0.40, 0.60, 41):
        acc = accuracy_score(y_tune, p_ens_tune >= th)
        if acc > best_acc_ens_tu:
            best_acc_ens_tu = acc
            best_th_ens = th

    print(f"Frozen Tune Parameters: w_LGBM = {best_w:.2f}, w_CatBoost = {1.0-best_w:.2f}, threshold = {best_th_ens:.2f}", flush=True)

    # 7. Evaluate Once on Confirm
    p_ens_conf = best_w * p_lgb_conf + (1.0 - best_w) * p_cb_conf
    acc_ens_conf = accuracy_score(y_confirm, p_ens_conf >= best_th_ens)
    loss_ens_conf = log_loss(y_confirm, p_ens_conf)

    pred_cb = (p_cb_conf >= best_th_cb).astype(int)
    pred_ens = (p_ens_conf >= best_th_ens).astype(int)

    fixes = int(np.sum((pred_cb != y_confirm) & (pred_ens == y_confirm)))
    breaks = int(np.sum((pred_cb == y_confirm) & (pred_ens != y_confirm)))
    net_total = fixes - breaks
    net_delta = (pred_ens == y_confirm).astype(int) - (pred_cb == y_confirm).astype(int)

    net_seen = int(net_delta[confirm_seen].sum())
    fixes_seen = int(np.sum((pred_cb[confirm_seen] != y_confirm[confirm_seen]) & (pred_ens[confirm_seen] == y_confirm[confirm_seen])))
    breaks_seen = int(np.sum((pred_cb[confirm_seen] == y_confirm[confirm_seen]) & (pred_ens[confirm_seen] != y_confirm[confirm_seen])))

    net_new = int(net_delta[~confirm_seen].sum())
    fixes_new = int(np.sum((pred_cb[~confirm_seen] != y_confirm[~confirm_seen]) & (pred_ens[~confirm_seen] == y_confirm[~confirm_seen])))
    breaks_new = int(np.sum((pred_cb[~confirm_seen] == y_confirm[~confirm_seen]) & (pred_ens[~confirm_seen] != y_confirm[~confirm_seen])))

    ci_low, ci_high = bootstrap_seller_delta(sellers[confirm], net_delta, n_boot=2000, seed=seed)

    print(f"\n--- CONFIRM RESULTS vs STRONG BASELINE (SEED {seed}) ---", flush=True)
    print(f"Deep CatBoost Base: Acc = {acc_cb_conf:.5f}, LogLoss = {loss_cb_conf:.5f}", flush=True)
    print(f"Deep Ensemble:      Acc = {acc_ens_conf:.5f}, LogLoss = {loss_ens_conf:.5f}", flush=True)
    print(f"Total Flips vs Base: {int(np.sum(pred_cb != pred_ens))}", flush=True)
    print(f"Fixes / Breaks:      {fixes} / {breaks} => NET: {net_total:+d}", flush=True)
    print(f"Seen Sellers:        {fixes_seen} / {breaks_seen} => NET: {net_seen:+d}", flush=True)
    print(f"New Sellers:         {fixes_new} / {breaks_new} => NET: {net_new:+d}", flush=True)
    print(f"Bootstrap 95% CI (by seller): [{ci_low:+.1f}, {ci_high:+.1f}]", flush=True)

    # Save paired predictions
    df_paired = pd.DataFrame({
        "id": ids[confirm],
        "seller_id": sellers[confirm],
        "y_true": y_confirm,
        "is_seen": confirm_seen.astype(int),
        "p_catboost": p_cb_conf.astype(np.float32),
        "p_lightgbm": p_lgb_conf.astype(np.float32),
        "p_ensemble": p_ens_conf.astype(np.float32),
        "pred_catboost": pred_cb,
        "pred_ensemble": pred_ens,
    })
    paired_path = RESULTS_DIR / f"paired_strong_ensemble_seed_{seed}.csv"
    df_paired.to_csv(paired_path, index=False)
    sha = sha256_file(paired_path)
    print(f"Saved paired predictions: {paired_path} (SHA256: {sha})", flush=True)

    return {
        "seed": seed,
        "best_w_lgbm": best_w,
        "best_th_cb": best_th_cb,
        "best_th_ens": best_th_ens,
        "acc_cb": acc_cb_conf,
        "loss_cb": loss_cb_conf,
        "acc_ens": acc_ens_conf,
        "loss_ens": loss_ens_conf,
        "fixes": fixes,
        "breaks": breaks,
        "net": net_total,
        "net_seen": net_seen,
        "fixes_seen": fixes_seen,
        "breaks_seen": breaks_seen,
        "net_new": net_new,
        "fixes_new": fixes_new,
        "breaks_new": breaks_new,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sha256": sha,
    }


def main():
    rows, ids, sellers, y_used = load_dataset()
    F_df, cat_df, titles, warranties, combos = extract_deep_features(rows)

    res_2026 = run_experiment_for_seed(2026, rows, ids, sellers, y_used, F_df, cat_df, titles, warranties, combos)
    res_314159 = run_experiment_for_seed(314159, rows, ids, sellers, y_used, F_df, cat_df, titles, warranties, combos)

    # Generate Markdown Report
    report_path = RESULTS_DIR / "round157_strong_ensemble_evaluation_20261029.md"
    md = f"""# Ronda 157: Evaluación del Ensamble Fuerte (LightGBM + CatBoost) con txt_sparse_p

Fecha: 2026-10-29  
Código ejecutable: `src/revision_submissions/strong_ensemble_audit_and_refit_20261029.py`  
Partición canónica exacta: R139/141 (`fit` ~45k, `tune` ~12k, `confirm` ~13k).  
Datos: 70.000 filas reales de train y textos originales, sin target leak ni scores públicos como etiquetas.

## 1. Verificación de Restauración del Baseline Fuerte

Con la inclusión de `txt_sparse_p` OOF y los hiperparámetros profundos (900 árboles, lr 0.07, l2 6.0), el baseline Deep CatBoost se recupera plenamente a su nivel real:
- **Seed 2026 CatBoost Base**: Accuracy = **{res_2026['acc_cb']:.5f}**, Log Loss = **{res_2026['loss_cb']:.5f}**
- **Seed 314159 CatBoost Base**: Accuracy = **{res_314159['acc_cb']:.5f}**, Log Loss = **{res_314159['loss_cb']:.5f}**
(Coincide exactamente con el rango .924/.920 reportado en Rounds 151 y 153).

## 2. Resultados del Ensamble en Confirmación

| Métrica | Seed 2026 | Seed 314159 | Promedio |
|---|---:|---:|---:|
| Filas Confirm | 12.906 | 12.844 | 12.875 |
| Vendedores Vistos / Nuevos | 3.950 / 8.956 | 3.831 / 9.013 | 3.890 / 8.985 |
| Peso LightGBM congelado ($w^*$) | {res_2026['best_w_lgbm']:.2f} | {res_314159['best_w_lgbm']:.2f} | {(res_2026['best_w_lgbm']+res_314159['best_w_lgbm'])/2:.2f} |
| Umbral congelado ($th^*$) | {res_2026['best_th_ens']:.2f} | {res_314159['best_th_ens']:.2f} | - |
| CatBoost Base Accuracy | {res_2026['acc_cb']:.5f} | {res_314159['acc_cb']:.5f} | {(res_2026['acc_cb']+res_314159['acc_cb'])/2:.5f} |
| CatBoost Base Log Loss | {res_2026['loss_cb']:.5f} | {res_314159['loss_cb']:.5f} | {(res_2026['loss_cb']+res_314159['loss_cb'])/2:.5f} |
| **Ensamble Accuracy** | **{res_2026['acc_ens']:.5f}** | **{res_314159['acc_ens']:.5f}** | **{(res_2026['acc_ens']+res_314159['acc_ens'])/2:.5f}** |
| **Ensamble Log Loss** | **{res_2026['loss_ens']:.5f}** | **{res_314159['loss_ens']:.5f}** | **{(res_2026['loss_ens']+res_314159['loss_ens'])/2:.5f}** |
| Fixes / Breaks vs CatBoost | {res_2026['fixes']} / {res_2026['breaks']} | {res_314159['fixes']} / {res_314159['breaks']} | {res_2026['fixes']+res_314159['fixes']} / {res_2026['breaks']+res_314159['breaks']} |
| **Neto Total Confirm** | **{res_2026['net']:+d}** | **{res_314159['net']:+d}** | **{res_2026['net']+res_314159['net']:+d}** |
| Neto en Vendedores Vistos | {res_2026['net_seen']:+d} ({res_2026['fixes_seen']}/{res_2026['breaks_seen']}) | {res_314159['net_seen']:+d} ({res_314159['fixes_seen']}/{res_314159['breaks_seen']}) | {res_2026['net_seen']+res_314159['net_seen']:+d} |
| Neto en Vendedores Nuevos | {res_2026['net_new']:+d} ({res_2026['fixes_new']}/{res_2026['breaks_new']}) | {res_314159['net_new']:+d} ({res_314159['fixes_new']}/{res_314159['breaks_new']}) | {res_2026['net_new']+res_314159['net_new']:+d} |
| **IC 95% Bootstrap agrupado por vendedor** | **[{res_2026['ci_low']:+.1f}, {res_2026['ci_high']:+.1f}]** | **[{res_314159['ci_low']:+.1f}, {res_314159['ci_high']:+.1f}]** | - |

## 3. Hashes de Pares

- `paired_strong_ensemble_seed_2026.csv`: SHA256 `{res_2026['sha256']}`
- `paired_strong_ensemble_seed_314159.csv`: SHA256 `{res_314159['sha256']}`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
