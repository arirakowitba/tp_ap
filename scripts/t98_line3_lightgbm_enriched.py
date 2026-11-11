"""T98 Line 3 — Enriched Leak-Free LightGBM Specialist + CX-0164 Corrected Stacked Features Audit.

Protocol & Guarantees (CX-0164 + T98 Specification):
1. Uses ONLY the frozen shared split (`data/splits/t98_dual_regime_outer_seller_group_splits.csv`)
   and pre-parsed feature caches (`data/splits/t98_parsed_train.pkl`, `data/splits/t98_parsed_test.pkl`).
2. For each outer fold k in 0..4:
   - `fit_idx`: rows where `role_fold_{k} == "fit"` (~49,574 rows).
   - `seen_val_idx`: rows where `role_fold_{k} == "seen_val"` (~6,426 rows, causal latest item of multi-item sellers in outer-train k).
   - `unseen_val_idx`: rows where `role_fold_{k} == "unseen_val"` (14,000 rows, outer_fold == k, 100% disjoint sellers).
   - `inner_fold_{k}`: values 0..3 on `fit_idx` (`StratifiedGroupKFold(4)` by `seller_id`, 0 seller overlap).
3. Fixes CX-0164 Stacking & Target Encoding Leakage (`M0_lgb_cx0164_clean`):
   - Inside any training set T (`fit_idx` or `inner_train`), `p_text_oof` is computed strictly Out-of-Fold
     using 4-fold `StratifiedGroupKFold` by `seller_id` inside T (zero in-sample `lr.predict_proba(XT)` leakage).
   - For the **unseen-seller regime** (and inner seller-grouped folds), `seller_te` (`m=5.0`) computed by
     `StratifiedGroupKFold` on `seller_id` equals the fold prior for every row, eliminating the train/val distribution shift
     identified in CX-0164.
   - For the **seen-seller regime** (`seen_val_idx`), we explicitly record and evaluate:
     (a) Causal prior-items `seller_te` (`seller_te_causal_prior`): for each item of seller s ordered by `(start_time, row)` in T,
         smoothed TE over strictly earlier items of seller s in T (sellers' first item gets prior on T; `seen_val_idx` gets
         all historical items in T), plus `seller_hist_cnt = log1p(prior_count)`.
     (b) Row-level 4-fold StratifiedKFold OOF `seller_te` on T matched to full-T `seller_te` on `seen_val_idx` (`M0_seen_rowte`),
         and the unseen-regime model evaluated directly on `seen_val_idx` (`M0_seen_unseen_model`).
4. Builds `M1_lgb_enriched` — Enriched Leak-Free LightGBM Specialist:
   - Fold-local price & relative price features (`log1p(price)`, `log1p(base_price)`, `price / (cat_median_price + 1)`,
     `(log_price - cat_mean_log_price) / (cat_std_log_price + 0.1)`, category-conditional percentile rank of price in T,
     `has_orig_price`, `discount_ratio = price / (orig_price + 1)`).
   - Quantities & commercial signals (`log1p(init_qty)`, `log1p(avail_qty)`, `log1p(sold_qty)`, `sold_rate`,
     `is_qty_1`, `is_sold_0`, `listing_type_id` OOF TE + one-hot for `free`, `bronze`, `silver`, `gold`, `gold_special`, `gold_pro`,
     `buying_mode`, `automatic_relist`, `accepts_mercadopago`, `has_official_store`, `has_video`, `has_catalog`, `has_parent_item`).
   - Shipping, payment & location (`free_shipping`, `local_pick_up`, `ship_mode` OOF TE, `num_non_mp`, `has_efectivo`,
     `has_transferencia`, `has_acordar`, `state_id` OOF TE, `num_pics`, `num_vars`, `num_deals`, `num_tags`,
     `has_dragged_bids`, `has_good_quality_thumb`).
   - Attributes & warranty (`num_attrs`, `warr_has_mes_ano`, `warr_sin_garantia`, `warr_fabrica`, `warranty_len`,
     OOF TE of `brand_val`, `model_val`, `item_cond_attr`).
   - Leak-free seller-grouped OOF Text & SVD features:
     * `p_text_title_tags` (Word(1,2) + Char_wb(3,5) SAGA LR, strictly 4-fold seller-grouped OOF inside T)
     * `p_text_attrs_warranty` (Word(1,2) + Char_wb(3,5) on `attr_text + warranty_str + subtitle`, SAGA LR, strictly 4-fold seller-grouped OOF inside T)
     * `TruncatedSVD(64)` fit only on T (`title + tags_str` TF-IDF).
   - Selects hyperparameters (`num_leaves` in `[55, 95]`, `colsample_bytree` in `[0.7, 0.8]`, `min_child_samples` in `[20, 40]`)
     strictly on the 4 inner seller-group folds of `fit_idx`, freezes to `freeze_line3_outer{k}.json`, and fits 3-seed bagged
     `LGBMClassifier` (`seeds=[42, 2026, 777]`) on `fit_idx`.
5. Status of all outputs: DEVELOPMENT_ONLY. Zero submissions or Kaggle calls.
"""
import gc
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "results" / "t98_line3_lightgbm_enriched_20261111"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = 4
SEEDS = [42, 2026, 777]
LGB_TREES = int(os.environ.get("T98_LGB_TREES", 900))
INNER_TUNE_TREES = int(os.environ.get("T98_INNER_TREES", 700))

USED_KWS = [
    "antiguo", "antigua", "antiguos", "antiguas", "vintage", "retro",
    "impecable", "impecables", "poco uso", "como nuevo", "despiece", "usado", "usada"
]
NEW_KWS = [
    "nuevo", "nueva", "nuevos", "nuevas", "sellado", "en caja",
    "0km", "cero km", "garantia oficial", "de fabrica"
]

HP_GRID = [
    {"num_leaves": nl, "colsample_bytree": cs, "min_child_samples": mcs}
    for nl, cs, mcs in itertools.product([55, 95], [0.7, 0.8], [20, 40])
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def precompute_item_intrinsic_arrays(df: pd.DataFrame) -> dict:
    """Extract purely item-intrinsic features (zero cross-row, target, or global stats)."""
    n = len(df)
    titles = df["title"].fillna("").astype(str)
    tl = titles.str.lower()
    used_kw_cnt = np.array([sum(kw in s for kw in USED_KWS) for s in tl], dtype=np.float32)
    new_kw_cnt = np.array([sum(kw in s for kw in NEW_KWS) for s in tl], dtype=np.float32)

    ind_m0 = np.column_stack([
        used_kw_cnt,
        new_kw_cnt,
        df["title_len"].values.astype(np.float32),
        df["title_words"].values.astype(np.float32),
        df["title_digits"].values.astype(np.float32),
        df["title_uppers"].values.astype(np.float32),
    ]).astype(np.float32)

    corpus_title_tags = (df["title"].fillna("") + " " + df["tags_str"].fillna("")).tolist()
    corpus_attrs_warr = (
        df["attr_text"].fillna("") + " " + df["warranty_str"].fillna("") + " " + df["subtitle"].fillna("")
    ).tolist()

    price = np.maximum(0.0, df["price"].fillna(0.0).values.astype(np.float64))
    base_price = np.maximum(0.0, df["base_price"].fillna(0.0).values.astype(np.float64))
    orig_price = np.maximum(0.0, df["orig_price"].fillna(0.0).values.astype(np.float64))
    has_orig_price = df["has_orig_price"].fillna(0).values.astype(np.float32)

    log_price = np.log1p(price).astype(np.float32)
    log_base_price = np.log1p(base_price).astype(np.float32)
    discount_ratio = np.clip(price / (orig_price + 1.0), 0.0, 10.0).astype(np.float32)
    base_to_price = np.clip(base_price / (price + 1.0), 0.0, 10.0).astype(np.float32)

    init_qty = np.maximum(0.0, df["init_qty"].fillna(0.0).values.astype(np.float64))
    avail_qty = np.maximum(0.0, df["avail_qty"].fillna(0.0).values.astype(np.float64))
    sold_qty = np.maximum(0.0, df["sold_qty"].fillna(0.0).values.astype(np.float64))

    log_init_qty = np.log1p(init_qty).astype(np.float32)
    log_avail_qty = np.log1p(avail_qty).astype(np.float32)
    log_sold_qty = np.log1p(sold_qty).astype(np.float32)
    sold_rate = np.clip(sold_qty / (init_qty + 1.0), 0.0, 10.0).astype(np.float32)
    avail_rate = np.clip(avail_qty / (init_qty + 1.0), 0.0, 10.0).astype(np.float32)
    is_qty_1 = (init_qty == 1.0).astype(np.float32)
    is_sold_0 = (sold_qty == 0.0).astype(np.float32)
    is_sold_pos = (sold_qty > 0.0).astype(np.float32)

    lt = df["listing_type_id"].fillna("MISSING").astype(str).values
    lt_oh = np.column_stack([
        (lt == "free").astype(np.float32),
        (lt == "bronze").astype(np.float32),
        (lt == "silver").astype(np.float32),
        (lt == "gold").astype(np.float32),
        (lt == "gold_special").astype(np.float32),
        (lt == "gold_pro").astype(np.float32),
        (lt == "gold_premium").astype(np.float32),
    ])

    bm = df["buying_mode"].fillna("MISSING").astype(str).values
    bm_oh = np.column_stack([
        (bm == "buy_it_now").astype(np.float32),
        (bm == "classified").astype(np.float32),
        (bm == "auction").astype(np.float32),
    ])

    curr = (df["currency_id"].fillna("ARS").astype(str).values == "USD").astype(np.float32)
    status_active = (df["status"].fillna("active").astype(str).values == "active").astype(np.float32)
    sub_susp = (df["sub_status_str"].fillna("NONE").astype(str).values == "suspended").astype(np.float32)

    pay_s = df["pay_ids"].fillna("").astype(str)
    has_efectivo = np.array(["MLAMO" in s for s in pay_s], dtype=np.float32)
    has_transferencia = np.array(["MLATB" in s for s in pay_s], dtype=np.float32)
    has_acordar = np.array(["MLAWC" in s for s in pay_s], dtype=np.float32)

    tags_s = df["tags_str"].fillna("").astype(str)
    has_dragged_bids = np.array(["dragged_bids_and_visits" in s for s in tags_s], dtype=np.float32)
    has_good_quality_thumb = np.array(["good_quality_thumbnail" in s for s in tags_s], dtype=np.float32)
    has_poor_quality_thumb = np.array(["poor_quality_thumbnail" in s for s in tags_s], dtype=np.float32)

    warr_s = df["warranty_str"].fillna("").astype(str)
    warranty_len = np.array([len(s) for s in warr_s], dtype=np.float32)
    subtitle_len = np.array([len(s) for s in df["subtitle"].fillna("").astype(str)], dtype=np.float32)

    item_cond_s = df["item_cond_attr"].fillna("MISSING").astype(str).str.lower()
    ic_used = np.array(["usad" in s or "used" in s for s in item_cond_s], dtype=np.float32)
    ic_new = np.array(["nuev" in s or "new" in s for s in item_cond_s], dtype=np.float32)

    start_t = df["start_time"].fillna(0).values.astype(np.float64)
    stop_t = df["stop_time"].fillna(0).values.astype(np.float64)
    duration_days = np.clip((stop_t - start_t) / (1000.0 * 86400.0), -365.0, 3650.0).astype(np.float32)

    static_m1 = np.column_stack([
        log_price,
        log_base_price,
        has_orig_price,
        discount_ratio,
        base_to_price,
        log_init_qty,
        log_avail_qty,
        log_sold_qty,
        sold_rate,
        avail_rate,
        is_qty_1,
        is_sold_0,
        is_sold_pos,
        lt_oh,
        bm_oh,
        curr,
        status_active,
        sub_susp,
        df["automatic_relist"].fillna(0).values.astype(np.float32),
        df["accepts_mercadopago"].fillna(0).values.astype(np.float32),
        df["has_official_store"].fillna(0).values.astype(np.float32),
        df["has_video"].fillna(0).values.astype(np.float32),
        df["has_catalog"].fillna(0).values.astype(np.float32),
        df["has_parent_item"].fillna(0).values.astype(np.float32),
        df["free_shipping"].fillna(0).values.astype(np.float32),
        df["local_pick_up"].fillna(0).values.astype(np.float32),
        df["num_non_mp"].fillna(0).values.astype(np.float32),
        has_efectivo,
        has_transferencia,
        has_acordar,
        df["num_pics"].fillna(0).values.astype(np.float32),
        (df["num_pics"].fillna(0).values == 0).astype(np.float32),
        df["num_vars"].fillna(0).values.astype(np.float32),
        df["num_deals"].fillna(0).values.astype(np.float32),
        df["num_tags"].fillna(0).values.astype(np.float32),
        has_dragged_bids,
        has_good_quality_thumb,
        has_poor_quality_thumb,
        df["num_attrs"].fillna(0).values.astype(np.float32),
        df["warr_has_mes_ano"].fillna(0).values.astype(np.float32),
        df["warr_sin_garantia"].fillna(0).values.astype(np.float32),
        df["warr_fabrica"].fillna(0).values.astype(np.float32),
        warranty_len,
        np.log1p(warranty_len),
        subtitle_len,
        ic_used,
        ic_new,
        duration_days,
        ind_m0,
    ]).astype(np.float32)

    return {
        "row": df["row"].values,
        "id": df["id"].values,
        "y": df["y"].values.astype(np.int32),
        "seller_id": df["seller_id"].astype(str).values,
        "category_id": df["category_id"].astype(str).values,
        "listing_type_id": lt,
        "buying_mode": bm,
        "ship_mode": df["ship_mode"].fillna("not_specified").astype(str).values,
        "state_id": df["state_id"].fillna("MISSING").astype(str).values,
        "city_id": df["city_id"].fillna("MISSING").astype(str).values,
        "brand_val": df["brand_val"].fillna("MISSING").astype(str).values,
        "model_val": df["model_val"].fillna("MISSING").astype(str).values,
        "item_cond_attr": df["item_cond_attr"].fillna("MISSING").astype(str).values,
        "price": price,
        "log_price": log_price,
        "num_attrs": df["num_attrs"].fillna(0).values.astype(np.float32),
        "warranty_m0": df["warr_has_mes_ano"].fillna(0).values.astype(np.float32),
        "init_qty_m0": np.log1p(np.maximum(0.0, df["init_qty"].fillna(1.0).values)).astype(np.float32),
        "ind_m0": ind_m0,
        "static_m1": static_m1,
        "start_time": df["start_time"].fillna(0).values.astype(np.int64),
        "corpus_title_tags": corpus_title_tags,
        "corpus_attrs_warr": corpus_attrs_warr,
    }


def te_map(cats: np.ndarray, y: np.ndarray, m: float, prior: float) -> pd.Series:
    s = pd.DataFrame({"k": cats, "y": y}).groupby("k")["y"].agg(["count", "sum"])
    return (s["sum"] + m * prior) / (s["count"] + m)


def compute_oof_te_and_maps(
    cats_T: np.ndarray,
    y_T: np.ndarray,
    splits_T: list,
    m: float,
    prior: float,
    eval_cats_list: list,
) -> tuple:
    """Compute strictly out-of-fold TE on T using `splits_T` and full-T map on each eval array."""
    oof_T = np.full(len(y_T), prior, dtype=np.float32)
    for tr_sub, va_sub in splits_T:
        mp = te_map(cats_T[tr_sub], y_T[tr_sub], m, float(y_T[tr_sub].mean()))
        oof_T[va_sub] = pd.Series(cats_T[va_sub]).map(mp).fillna(float(y_T[tr_sub].mean())).values.astype(np.float32)
    full_mp = te_map(cats_T, y_T, m, prior)
    eval_outs = [
        pd.Series(c_eval).map(full_mp).fillna(prior).values.astype(np.float32)
        for c_eval in eval_cats_list
    ]
    return oof_T, eval_outs


def compute_causal_seller_features(
    sellers_T: np.ndarray,
    y_T: np.ndarray,
    start_T: np.ndarray,
    row_T: np.ndarray,
    m: float,
    prior: float,
    eval_sellers_list: list,
) -> tuple:
    """Compute causal prior-items seller_te and log1p(prior_count) inside T, and full-T historical stats on eval sets.

    Inside T:
      Items are sorted chronologically by `(seller_id, start_time, row)`.
      For the k-th item (0-indexed) of seller s in T:
        prior_count = k
        prior_sum = sum of y for strictly earlier items 0..k-1 of seller s in T
        causal_seller_te = (prior_sum + m * prior) / (prior_count + m)
    On any evaluation set P (specifically `seen_val_idx`, whose items occur strictly after all items of that seller in T):
      prior_count = total items of seller s in T
      prior_sum = sum of y for all items of seller s in T
      causal_seller_te = (prior_sum + m * prior) / (prior_count + m)
    """
    df_s = pd.DataFrame({
        "pos": np.arange(len(y_T)),
        "seller_id": sellers_T,
        "start_time": start_T,
        "row": row_T,
        "y": y_T,
    }).sort_values(["seller_id", "start_time", "row"])

    grp = df_s.groupby("seller_id", sort=False)
    prior_cnt = grp.cumcount().values.astype(np.float32)
    cum_y = grp["y"].cumsum().values.astype(np.float32)
    prior_sum = cum_y - df_s["y"].values.astype(np.float32)

    causal_te_sorted = (prior_sum + m * prior) / (prior_cnt + m)
    causal_cnt_sorted = np.log1p(prior_cnt)

    pos_order = df_s["pos"].values
    causal_te_T = np.empty(len(y_T), dtype=np.float32)
    causal_cnt_T = np.empty(len(y_T), dtype=np.float32)
    causal_te_T[pos_order] = causal_te_sorted
    causal_cnt_T[pos_order] = causal_cnt_sorted

    agg = pd.DataFrame({"k": sellers_T, "y": y_T}).groupby("k")["y"].agg(["count", "sum"])
    full_te_map = (agg["sum"] + m * prior) / (agg["count"] + m)
    full_cnt_map = np.log1p(agg["count"])

    eval_te_outs = []
    eval_cnt_outs = []
    for s_eval in eval_sellers_list:
        s_ser = pd.Series(s_eval)
        eval_te_outs.append(s_ser.map(full_te_map).fillna(prior).values.astype(np.float32))
        eval_cnt_outs.append(s_ser.map(full_cnt_map).fillna(0.0).values.astype(np.float32))

    return causal_te_T, causal_cnt_T, eval_te_outs, eval_cnt_outs


def compute_fold_local_price_features(
    price_T: np.ndarray,
    log_price_T: np.ndarray,
    cat_T: np.ndarray,
    eval_price_list: list,
    eval_log_price_list: list,
    eval_cat_list: list,
) -> tuple:
    """Compute fold-local relative price statistics strictly from T (`fit` or `inner_train`)."""
    s_price = pd.Series(price_T)
    s_lp = pd.Series(log_price_T)

    med_map = s_price.groupby(cat_T).median()
    mean_lp_map = s_lp.groupby(cat_T).mean()
    std_lp_map = s_lp.groupby(cat_T).std().fillna(0.5)
    cnt_map = s_price.groupby(cat_T).count()

    global_med = float( np.median(price_T) )
    global_mean_lp = float( np.mean(log_price_T) )
    global_std_lp = float( np.std(log_price_T) )

    # Category-conditional percentile rank computed strictly on T
    sorted_prices_by_cat = {}
    df_cat_p = pd.DataFrame({"c": cat_T, "p": price_T})
    for c_val, grp in df_cat_p.groupby("c", sort=False):
        sorted_prices_by_cat[c_val] = np.sort(grp["p"].values)

    def _transform_one(p_arr: np.ndarray, lp_arr: np.ndarray, c_arr: np.ndarray) -> tuple:
        c_ser = pd.Series(c_arr)
        med = c_ser.map(med_map).fillna(100.0).values
        pr_m0 = np.clip(p_arr / (med + 1.0), 0.0, 50.0).astype(np.float32)

        med_m1 = c_ser.map(med_map).fillna(global_med).values
        pr_m1 = np.clip(p_arr / (med_m1 + 1.0), 0.0, 50.0).astype(np.float32)

        mn_lp = c_ser.map(mean_lp_map).fillna(global_mean_lp).values
        sd_lp = c_ser.map(std_lp_map).fillna(global_std_lp).values
        z_lp = np.clip((lp_arr - mn_lp) / (sd_lp + 0.1), -10.0, 10.0).astype(np.float32)

        cat_freq = np.log1p(c_ser.map(cnt_map).fillna(0.0).values).astype(np.float32)

        pct_rank = np.full(len(p_arr), 0.5, dtype=np.float32)
        df_tmp = pd.DataFrame({"c": c_arr, "p": p_arr, "idx": np.arange(len(p_arr))})
        for c_val, grp in df_tmp.groupby("c", sort=False):
            ref = sorted_prices_by_cat.get(c_val)
            if ref is not None and len(ref) > 0:
                left = np.searchsorted(ref, grp["p"].values, side="left")
                right = np.searchsorted(ref, grp["p"].values, side="right")
                pct_rank[grp["idx"].values] = (0.5 * (left + right) / len(ref)).astype(np.float32)

        m1_price_block = np.column_stack([pr_m1, z_lp, pct_rank, cat_freq]).astype(np.float32)
        return pr_m0, m1_price_block

    pr_m0_T, m1_price_T = _transform_one(price_T, log_price_T, cat_T)
    pr_m0_evals, m1_price_evals = [], []
    for p_e, lp_e, c_e in zip(eval_price_list, eval_log_price_list, eval_cat_list):
        pm0, pm1 = _transform_one(p_e, lp_e, c_e)
        pr_m0_evals.append(pm0)
        m1_price_evals.append(pm1)

    return pr_m0_T, m1_price_T, pr_m0_evals, m1_price_evals


def compute_leak_free_text_and_svd(
    T_indices: np.ndarray,
    eval_indices_list: list,
    y_T: np.ndarray,
    splits_T: list,
    corpus_title_tags: list,
    corpus_attrs_warr: list,
    cache_path: Path,
) -> dict:
    """Compute strictly leak-free seller-grouped OOF `p_text_title_tags` and `p_text_attrs_warranty` on T,
    plus full-T SAGA LR predictions and TruncatedSVD(64) on T and all eval sets.
    """
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    t0 = time.time()
    c1_T = [corpus_title_tags[i] for i in T_indices]
    c2_T = [corpus_attrs_warr[i] for i in T_indices]

    vw1 = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
    vc1 = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)
    X1_T = hstack([vw1.fit_transform(c1_T), vc1.fit_transform(c1_T)]).tocsr()

    vw2 = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, min_df=3, sublinear_tf=True)
    vc2 = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=35000, min_df=5, sublinear_tf=True)
    X2_T = hstack([vw2.fit_transform(c2_T), vc2.fit_transform(c2_T)]).tocsr()

    # Strictly Out-of-Fold seller-grouped predictions inside T (CX-0164 Item 2 & Action C)
    ptx1_oof_T = np.zeros(len(T_indices), dtype=np.float32)
    ptx2_oof_T = np.zeros(len(T_indices), dtype=np.float32)
    for tr_sub, va_sub in splits_T:
        lr1_sub = LogisticRegression(C=1.5, max_iter=800, solver="saga", random_state=42)
        lr1_sub.fit(X1_T[tr_sub], y_T[tr_sub])
        ptx1_oof_T[va_sub] = lr1_sub.predict_proba(X1_T[va_sub])[:, 1].astype(np.float32)

        lr2_sub = LogisticRegression(C=1.0, max_iter=800, solver="saga", random_state=42)
        lr2_sub.fit(X2_T[tr_sub], y_T[tr_sub])
        ptx2_oof_T[va_sub] = lr2_sub.predict_proba(X2_T[va_sub])[:, 1].astype(np.float32)

    # Fit full-T text models & SVD(64) on X1_T only
    lr1_full = LogisticRegression(C=1.5, max_iter=800, solver="saga", random_state=42)
    lr1_full.fit(X1_T, y_T)
    lr2_full = LogisticRegression(C=1.0, max_iter=800, solver="saga", random_state=42)
    lr2_full.fit(X2_T, y_T)

    svd = TruncatedSVD(n_components=64, random_state=42)
    svd_T = svd.fit_transform(X1_T).astype(np.float32)

    out = {
        "ptx1_oof_T": ptx1_oof_T,
        "ptx2_oof_T": ptx2_oof_T,
        "svd_T": svd_T,
        "vocab_w1": np.array([len(vw1.vocabulary_)], dtype=np.int32),
        "vocab_c1": np.array([len(vc1.vocabulary_)], dtype=np.int32),
        "vocab_w2": np.array([len(vw2.vocabulary_)], dtype=np.int32),
        "vocab_c2": np.array([len(vc2.vocabulary_)], dtype=np.int32),
        "svd_var": np.array([float(svd.explained_variance_ratio_.sum())], dtype=np.float32),
        "seconds": np.array([round(time.time() - t0, 2)], dtype=np.float32),
    }

    for idx_e, E_indices in enumerate(eval_indices_list):
        c1_E = [corpus_title_tags[i] for i in E_indices]
        c2_E = [corpus_attrs_warr[i] for i in E_indices]
        X1_E = hstack([vw1.transform(c1_E), vc1.transform(c1_E)]).tocsr()
        X2_E = hstack([vw2.transform(c2_E), vc2.transform(c2_E)]).tocsr()
        out[f"ptx1_eval_{idx_e}"] = lr1_full.predict_proba(X1_E)[:, 1].astype(np.float32)
        out[f"ptx2_eval_{idx_e}"] = lr2_full.predict_proba(X2_E)[:, 1].astype(np.float32)
        out[f"svd_eval_{idx_e}"] = svd.transform(X1_E).astype(np.float32)

    np.savez_compressed(cache_path, **out)
    del X1_T, X2_T, vw1, vc1, vw2, vc2, svd
    gc.collect()
    return out


def build_fold_matrices(
    T_indices: np.ndarray,
    eval_indices_list: list,
    splits_T: list,
    tr_data: dict,
    eval_data_list: list,
    cache_path: Path,
) -> dict:
    """Construct leak-free M0 and M1 feature matrices for T and each evaluation set in `eval_indices_list`."""
    y_T = tr_data["y"][T_indices]
    prior = float(y_T.mean())
    sel_T = tr_data["seller_id"][T_indices]
    cat_T = tr_data["category_id"][T_indices]

    # 1. Leak-free seller-grouped OOF Text & SVD features
    corpus_tt_all = tr_data["corpus_title_tags"]
    corpus_aw_all = tr_data["corpus_attrs_warr"]

    # Since eval_data_list may include test_data (which has its own corpus list), handle corpus lookup cleanly
    t0 = time.time()
    if cache_path.exists():
        tx = np.load(cache_path, allow_pickle=True)
        tx = {k: tx[k] for k in tx.files}
    else:
        c1_T = [tr_data["corpus_title_tags"][i] for i in T_indices]
        c2_T = [tr_data["corpus_attrs_warr"][i] for i in T_indices]

        vw1 = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=3, sublinear_tf=True)
        vc1 = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=5, sublinear_tf=True)
        X1_T = hstack([vw1.fit_transform(c1_T), vc1.fit_transform(c1_T)]).tocsr()

        vw2 = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, min_df=3, sublinear_tf=True)
        vc2 = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=35000, min_df=5, sublinear_tf=True)
        X2_T = hstack([vw2.fit_transform(c2_T), vc2.fit_transform(c2_T)]).tocsr()

        ptx1_oof_T = np.zeros(len(T_indices), dtype=np.float32)
        ptx2_oof_T = np.zeros(len(T_indices), dtype=np.float32)
        for tr_sub, va_sub in splits_T:
            lr1_sub = LogisticRegression(C=1.5, max_iter=800, solver="saga", random_state=42)
            lr1_sub.fit(X1_T[tr_sub], y_T[tr_sub])
            ptx1_oof_T[va_sub] = lr1_sub.predict_proba(X1_T[va_sub])[:, 1].astype(np.float32)

            lr2_sub = LogisticRegression(C=1.0, max_iter=800, solver="saga", random_state=42)
            lr2_sub.fit(X2_T[tr_sub], y_T[tr_sub])
            ptx2_oof_T[va_sub] = lr2_sub.predict_proba(X2_T[va_sub])[:, 1].astype(np.float32)

        lr1_full = LogisticRegression(C=1.5, max_iter=800, solver="saga", random_state=42)
        lr1_full.fit(X1_T, y_T)
        lr2_full = LogisticRegression(C=1.0, max_iter=800, solver="saga", random_state=42)
        lr2_full.fit(X2_T, y_T)

        svd = TruncatedSVD(n_components=64, random_state=42)
        svd_T = svd.fit_transform(X1_T).astype(np.float32)

        tx = {
            "ptx1_oof_T": ptx1_oof_T,
            "ptx2_oof_T": ptx2_oof_T,
            "svd_T": svd_T,
            "vocab_w1": np.array([len(vw1.vocabulary_)], dtype=np.int32),
            "vocab_c1": np.array([len(vc1.vocabulary_)], dtype=np.int32),
            "vocab_w2": np.array([len(vw2.vocabulary_)], dtype=np.int32),
            "vocab_c2": np.array([len(vc2.vocabulary_)], dtype=np.int32),
            "svd_var": np.array([float(svd.explained_variance_ratio_.sum())], dtype=np.float32),
            "seconds": np.array([round(time.time() - t0, 2)], dtype=np.float32),
        }
        for idx_e, (E_indices, e_data) in enumerate(zip(eval_indices_list, eval_data_list)):
            c1_E = [e_data["corpus_title_tags"][i] for i in E_indices]
            c2_E = [e_data["corpus_attrs_warr"][i] for i in E_indices]
            X1_E = hstack([vw1.transform(c1_E), vc1.transform(c1_E)]).tocsr()
            X2_E = hstack([vw2.transform(c2_E), vc2.transform(c2_E)]).tocsr()
            tx[f"ptx1_eval_{idx_e}"] = lr1_full.predict_proba(X1_E)[:, 1].astype(np.float32)
            tx[f"ptx2_eval_{idx_e}"] = lr2_full.predict_proba(X2_E)[:, 1].astype(np.float32)
            tx[f"svd_eval_{idx_e}"] = svd.transform(X1_E).astype(np.float32)

        np.savez_compressed(cache_path, **tx)
        del X1_T, X2_T, vw1, vc1, vw2, vc2, svd
        gc.collect()

    # 2. Fold-local price features
    eval_prices = [e_data["price"][E_idx] for E_idx, e_data in zip(eval_indices_list, eval_data_list)]
    eval_lps = [e_data["log_price"][E_idx] for E_idx, e_data in zip(eval_indices_list, eval_data_list)]
    eval_cats = [e_data["category_id"][E_idx] for E_idx, e_data in zip(eval_indices_list, eval_data_list)]
    eval_sellers = [e_data["seller_id"][E_idx] for E_idx, e_data in zip(eval_indices_list, eval_data_list)]

    pr_m0_T, m1_price_T, pr_m0_evals, m1_price_evals = compute_fold_local_price_features(
        tr_data["price"][T_indices],
        tr_data["log_price"][T_indices],
        cat_T,
        eval_prices,
        eval_lps,
        eval_cats,
    )

    # 3. Seller-grouped OOF Target Encodings (CX-0164 compliant)
    cte_T, cte_evals = compute_oof_te_and_maps(cat_T, y_T, splits_T, 10.0, prior, eval_cats)

    # Under seller-grouped splits_T, seller_id is disjoint across splits_T, so seller_te on T equals fold prior!
    ste_unseen_T, ste_unseen_evals = compute_oof_te_and_maps(sel_T, y_T, splits_T, 5.0, prior, eval_sellers)
    # Force exact prior on unseen eval sets when evaluating the unseen-seller regime model
    ste_unseen_prior_T = np.full(len(T_indices), prior, dtype=np.float32)
    ste_unseen_prior_evals = [np.full(len(E_idx), prior, dtype=np.float32) for E_idx in eval_indices_list]

    # Causal prior-items seller_te (for the seen-seller regime!)
    ste_causal_T, scnt_causal_T, ste_causal_evals, scnt_causal_evals = compute_causal_seller_features(
        sel_T,
        y_T,
        tr_data["start_time"][T_indices],
        tr_data["row"][T_indices],
        5.0,
        prior,
        eval_sellers,
    )

    # Also compute 4-fold row-level StratifiedKFold OOF seller_te on T (diagnostic comparison for seen_val_idx)
    skf_splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(np.zeros(len(T_indices)), y_T))
    ste_row_T, ste_row_evals = compute_oof_te_and_maps(sel_T, y_T, skf_splits, 5.0, prior, eval_sellers)

    # Additional categorical OOF TEs for M1 (all computed via seller-grouped splits_T)
    m1_te_cols = [
        ("listing_type_id", 10.0),
        ("ship_mode", 10.0),
        ("state_id", 10.0),
        ("city_id", 10.0),
        ("brand_val", 10.0),
        ("model_val", 10.0),
        ("item_cond_attr", 10.0),
    ]
    m1_te_T_list = [cte_T]
    m1_te_evals_list = [[cte_evals[i]] for i in range(len(eval_indices_list))]
    for col_name, m_val in m1_te_cols:
        col_T = tr_data[col_name][T_indices]
        col_evals = [e_data[col_name][E_idx] for E_idx, e_data in zip(eval_indices_list, eval_data_list)]
        oof_c, ev_c = compute_oof_te_and_maps(col_T, y_T, splits_T, m_val, prior, col_evals)
        m1_te_T_list.append(oof_c)
        for i in range(len(eval_indices_list)):
            m1_te_evals_list[i].append(ev_c[i])

    m1_te_block_T = np.column_stack(m1_te_T_list).astype(np.float32)
    m1_te_block_evals = [np.column_stack(lst).astype(np.float32) for lst in m1_te_evals_list]

    # 4. Assemble M0 and M1 feature matrices
    # M0 Unseen Regime (CX-0164 clean baseline)
    X_m0_unseen_T = np.column_stack([
        tr_data["log_price"][T_indices],
        pr_m0_T,
        tr_data["num_attrs"][T_indices],
        tr_data["warranty_m0"][T_indices],
        tr_data["init_qty_m0"][T_indices],
        cte_T,
        ste_unseen_T,
        tr_data["ind_m0"][T_indices],
        tx["ptx1_oof_T"],
        tx["svd_T"],
    ]).astype(np.float32)

    X_m0_unseen_evals = []
    for i, (E_idx, e_data) in enumerate(zip(eval_indices_list, eval_data_list)):
        X_m0_unseen_evals.append(np.column_stack([
            e_data["log_price"][E_idx],
            pr_m0_evals[i],
            e_data["num_attrs"][E_idx],
            e_data["warranty_m0"][E_idx],
            e_data["init_qty_m0"][E_idx],
            cte_evals[i],
            ste_unseen_prior_evals[i],
            e_data["ind_m0"][E_idx],
            tx[f"ptx1_eval_{i}"],
            tx[f"svd_eval_{i}"],
        ]).astype(np.float32))

    # M0 Seen Regime (Causal prior-items seller_te)
    X_m0_seen_causal_T = X_m0_unseen_T.copy()
    X_m0_seen_causal_T[:, 6] = ste_causal_T
    X_m0_seen_causal_evals = []
    for i in range(len(eval_indices_list)):
        arr = X_m0_unseen_evals[i].copy()
        arr[:, 6] = ste_causal_evals[i]
        X_m0_seen_causal_evals.append(arr)

    # M0 Seen Regime diagnostic alternative (Row OOF seller_te on T matched to full-T seller_te on eval)
    X_m0_seen_rowte_T = X_m0_unseen_T.copy()
    X_m0_seen_rowte_T[:, 6] = ste_row_T
    X_m0_seen_rowte_evals = []
    for i in range(len(eval_indices_list)):
        arr = X_m0_unseen_evals[i].copy()
        arr[:, 6] = ste_row_evals[i]
        X_m0_seen_rowte_evals.append(arr)

    # M1 Enriched Unseen Regime
    ptx_diff_T = (tx["ptx1_oof_T"] - tx["ptx2_oof_T"]).astype(np.float32)
    ptx_mean_T = (0.5 * (tx["ptx1_oof_T"] + tx["ptx2_oof_T"])).astype(np.float32)
    X_m1_unseen_T = np.column_stack([
        tr_data["static_m1"][T_indices],
        m1_price_T,
        m1_te_block_T,
        ste_unseen_T,
        np.zeros(len(T_indices), dtype=np.float32),  # seller_hist_cnt = 0 for unseen regime
        tx["ptx1_oof_T"],
        tx["ptx2_oof_T"],
        ptx_diff_T,
        ptx_mean_T,
        tx["svd_T"],
    ]).astype(np.float32)

    X_m1_unseen_evals = []
    for i, (E_idx, e_data) in enumerate(zip(eval_indices_list, eval_data_list)):
        ptx1_e = tx[f"ptx1_eval_{i}"]
        ptx2_e = tx[f"ptx2_eval_{i}"]
        X_m1_unseen_evals.append(np.column_stack([
            e_data["static_m1"][E_idx],
            m1_price_evals[i],
            m1_te_block_evals[i],
            ste_unseen_prior_evals[i],
            np.zeros(len(E_idx), dtype=np.float32),
            ptx1_e,
            ptx2_e,
            (ptx1_e - ptx2_e).astype(np.float32),
            (0.5 * (ptx1_e + ptx2_e)).astype(np.float32),
            tx[f"svd_eval_{i}"],
        ]).astype(np.float32))

    # M1 Enriched Seen Regime (Causal prior-items seller_te + seller_hist_cnt)
    ste_col_idx = tr_data["static_m1"].shape[1] + m1_price_T.shape[1] + m1_te_block_T.shape[1]
    X_m1_seen_causal_T = X_m1_unseen_T.copy()
    X_m1_seen_causal_T[:, ste_col_idx] = ste_causal_T
    X_m1_seen_causal_T[:, ste_col_idx + 1] = scnt_causal_T
    X_m1_seen_causal_evals = []
    for i in range(len(eval_indices_list)):
        arr = X_m1_unseen_evals[i].copy()
        arr[:, ste_col_idx] = ste_causal_evals[i]
        arr[:, ste_col_idx + 1] = scnt_causal_evals[i]
        X_m1_seen_causal_evals.append(arr)

    # M1 Seen Regime with Row OOF seller_te
    X_m1_seen_rowte_T = X_m1_unseen_T.copy()
    X_m1_seen_rowte_T[:, ste_col_idx] = ste_row_T
    X_m1_seen_rowte_T[:, ste_col_idx + 1] = scnt_causal_T
    X_m1_seen_rowte_evals = []
    for i in range(len(eval_indices_list)):
        arr = X_m1_unseen_evals[i].copy()
        arr[:, ste_col_idx] = ste_row_evals[i]
        arr[:, ste_col_idx + 1] = scnt_causal_evals[i]
        X_m1_seen_rowte_evals.append(arr)

    return {
        "y_T": y_T,
        "X_m0_unseen_T": X_m0_unseen_T,
        "X_m0_unseen_evals": X_m0_unseen_evals,
        "X_m0_seen_causal_T": X_m0_seen_causal_T,
        "X_m0_seen_causal_evals": X_m0_seen_causal_evals,
        "X_m0_seen_rowte_T": X_m0_seen_rowte_T,
        "X_m0_seen_rowte_evals": X_m0_seen_rowte_evals,
        "X_m1_unseen_T": X_m1_unseen_T,
        "X_m1_unseen_evals": X_m1_unseen_evals,
        "X_m1_seen_causal_T": X_m1_seen_causal_T,
        "X_m1_seen_causal_evals": X_m1_seen_causal_evals,
        "X_m1_seen_rowte_T": X_m1_seen_rowte_T,
        "X_m1_seen_rowte_evals": X_m1_seen_rowte_evals,
        "sha_ptx1_oof": sha256_array(tx["ptx1_oof_T"]),
        "sha_ptx2_oof": sha256_array(tx["ptx2_oof_T"]),
        "sha_cte_oof": sha256_array(cte_T),
        "sha_ste_unseen_oof": sha256_array(ste_unseen_T),
        "sha_ste_causal_oof": sha256_array(ste_causal_T),
        "sha_ste_row_oof": sha256_array(ste_row_T),
        "sha_m1_te_block": sha256_array(m1_te_block_T),
        "ptx1_oof_acc": float(accuracy_score(y_T, tx["ptx1_oof_T"] >= 0.5)),
        "ptx2_oof_acc": float(accuracy_score(y_T, tx["ptx2_oof_T"] >= 0.5)),
        "vocab_w1": int(tx["vocab_w1"][0]),
        "vocab_c1": int(tx["vocab_c1"][0]),
        "vocab_w2": int(tx["vocab_w2"][0]),
        "vocab_c2": int(tx["vocab_c2"][0]),
        "svd_var": float(tx["svd_var"][0]),
    }


def fit_predict_lgb_bagged(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval_list: list,
    hp: dict,
    seeds: list,
    n_estimators: int,
) -> list:
    preds_list = [np.zeros(len(X_e), dtype=np.float64) for X_e in X_eval_list]
    for sd in seeds:
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.025,
            num_leaves=hp["num_leaves"],
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=hp["colsample_bytree"],
            min_child_samples=hp["min_child_samples"],
            random_state=sd,
            n_jobs=N_JOBS,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        for i, X_e in enumerate(X_eval_list):
            preds_list[i] += model.predict_proba(X_e)[:, 1] / len(seeds)
    return [p.astype(np.float32) for p in preds_list]


def boot_two_way(diff: np.ndarray, sellers: np.ndarray, cats: np.ndarray, n_boot: int = 2000, seed: int = 20261111) -> np.ndarray:
    rng = np.random.default_rng(seed)
    su, si = np.unique(sellers, return_inverse=True)
    cu, ci = np.unique(cats, return_inverse=True)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w = rng.poisson(1.0, len(su))[si] * rng.poisson(1.0, len(cu))[ci]
        out[b] = (w * diff).sum() / w.sum()
    return out


def paired_comparison(
    y_true: np.ndarray,
    p_cand: np.ndarray,
    p_base: np.ndarray,
    folds: np.ndarray,
    sellers: np.ndarray,
    cats: np.ndarray,
    name: str,
) -> dict:
    n = len(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)
    diff = c_cand - c_base

    p_c_clip = np.clip(p_cand, 1e-7, 1.0 - 1e-7)
    p_b_clip = np.clip(p_base, 1e-7, 1.0 - 1e-7)
    ll_cand_i = -(y_true * np.log(p_c_clip) + (1.0 - y_true) * np.log(1.0 - p_c_clip))
    ll_base_i = -(y_true * np.log(p_b_clip) + (1.0 - y_true) * np.log(1.0 - p_b_clip))
    ll_diff = ll_cand_i - ll_base_i

    fold_deltas = []
    for k in sorted(np.unique(folds)):
        mk = folds == k
        fold_deltas.append(int(diff[mk].sum()))

    bs_hits = boot_two_way(diff.astype(np.float64), sellers, cats, 2000, 20261111) * n
    bs_ll = boot_two_way(ll_diff.astype(np.float64), sellers, cats, 2000, 20261112)

    return {
        "comparison": name,
        "n_rows": int(n),
        "cand_hits": int(c_cand.sum()),
        "base_hits": int(c_base.sum()),
        "cand_acc": round(float(c_cand.mean()), 6),
        "base_acc": round(float(c_base.mean()), 6),
        "delta_hits": int(diff.sum()),
        "fixes": int((diff == 1).sum()),
        "breaks": int((diff == -1).sum()),
        "fold_deltas": fold_deltas,
        "folds_positive": int(sum(d > 0 for d in fold_deltas)),
        "folds_zero": int(sum(d == 0 for d in fold_deltas)),
        "worst_fold_delta": int(min(fold_deltas)),
        "cand_logloss": round(float(ll_cand_i.mean()), 6),
        "base_logloss": round(float(ll_base_i.mean()), 6),
        "delta_logloss": round(float(ll_diff.mean()), 6),
        "boot_delta_hits_median": round(float(np.median(bs_hits)), 2),
        "boot_delta_hits_q05": round(float(np.quantile(bs_hits, 0.05)), 2),
        "boot_delta_hits_q95": round(float(np.quantile(bs_hits, 0.95)), 2),
        "boot_frac_gt0": round(float((bs_hits > 0).mean()), 4),
        "boot_delta_logloss_median": round(float(np.median(bs_ll)), 6),
        "boot_delta_logloss_q05": round(float(np.quantile(bs_ll, 0.05)), 6),
        "boot_delta_logloss_q95": round(float(np.quantile(bs_ll, 0.95)), 6),
    }


def main():
    t_start = time.time()
    log("Loading frozen T98 splits and pre-parsed train/test caches...")
    df_train = pd.read_pickle(SPLIT_DIR / "t98_parsed_train.pkl")
    df_test = pd.read_pickle(SPLIT_DIR / "t98_parsed_test.pkl")
    df_splits = pd.read_csv(SPLIT_DIR / "t98_dual_regime_outer_seller_group_splits.csv")

    assert len(df_train) == 70000 and len(df_splits) == 70000 and len(df_test) == 30000
    assert np.array_equal(df_train["row"].values, df_splits["row"].values)

    tr_data = precompute_item_intrinsic_arrays(df_train)
    te_data = precompute_item_intrinsic_arrays(df_test)
    del df_train, df_test
    gc.collect()

    log(f"Precomputed item-intrinsic arrays: M1 static cols={tr_data['static_m1'].shape[1]}")

    # Global containers for unseen 70k OOF, seen 32,092 OOF across 5 folds, and 30k test predictions
    oof_m0_unseen = np.zeros(70000, dtype=np.float32)
    oof_m1_unseen = np.zeros(70000, dtype=np.float32)

    seen_rows_records = []
    test_preds_per_fold = {
        "p_m0_unseen_test": [],
        "p_m0_seen_causal_test": [],
        "p_m0_seen_rowte_test": [],
        "p_m1_unseen_test": [],
        "p_m1_seen_causal_test": [],
        "p_m1_seen_rowte_test": [],
        "p_m0_dual_test": [],
        "p_m1_dual_test": [],
        "test_in_fit_seller": [],
    }

    provenance_rows = []
    per_fold_rows = []
    frozen_list = []

    hp_m0 = {"num_leaves": 55, "colsample_bytree": 0.80, "min_child_samples": 20}
    test_all_idx = np.arange(30000, dtype=np.int64)

    for k in range(5):
        t_fold = time.time()
        role_k = df_splits[f"role_fold_{k}"].values
        inner_k = df_splits[f"inner_fold_{k}"].values

        fit_idx = np.where(role_k == "fit")[0]
        seen_val_idx = np.where(role_k == "seen_val")[0]
        unseen_val_idx = np.where(role_k == "unseen_val")[0]

        # Verify strict seller disjointness between fit_idx and unseen_val_idx
        fit_sellers_set = set(tr_data["seller_id"][fit_idx])
        unseen_sellers_set = set(tr_data["seller_id"][unseen_val_idx])
        seen_sellers_set = set(tr_data["seller_id"][seen_val_idx])
        assert len(fit_sellers_set & unseen_sellers_set) == 0, f"Seller overlap in outer fold {k}!"
        assert seen_sellers_set.issubset(fit_sellers_set), f"Seen sellers not in fit_idx in fold {k}!"

        log(
            f"=== Outer Fold {k+1}/5: fit={len(fit_idx)}, seen_val={len(seen_val_idx)}, "
            f"unseen_val={len(unseen_val_idx)} ==="
        )

        fold_npz_path = OUT_DIR / f"fold_{k}_predictions.npz"
        freeze_path = OUT_DIR / f"freeze_line3_outer{k}.json"
        prov_cache_path = CACHE_DIR / f"fold_{k}_meta.json"

        # Build 4 inner fold splits inside fit_idx
        inner_splits_local = []
        for j in range(4):
            va_loc = np.where(inner_k[fit_idx] == j)[0]
            tr_loc = np.where(inner_k[fit_idx] != j)[0]
            assert len(set(tr_data["seller_id"][fit_idx[tr_loc]]) & set(tr_data["seller_id"][fit_idx[va_loc]])) == 0
            inner_splits_local.append((tr_loc, va_loc))

        # Step 1: Inner 4-fold OOF inside fit_idx (both M0 and M1 hyperparameter grid selection)
        p_m0_fit_oof = np.zeros(len(fit_idx), dtype=np.float32)
        hp_grid_oof = [np.zeros(len(fit_idx), dtype=np.float32) for _ in HP_GRID]

        for j, (tr_loc, va_loc) in enumerate(inner_splits_local):
            Ti = fit_idx[tr_loc]
            Pi = fit_idx[va_loc]

            # 4-fold seller-grouped split inside Ti for leak-free inner OOF text & TE
            sgkf_inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2000 + 10 * k + j)
            splits_Ti = list(sgkf_inner.split(np.zeros(len(Ti)), tr_data["y"][Ti], tr_data["seller_id"][Ti]))

            inner_pred_cache = CACHE_DIR / f"outer{k}_inner{j}_preds.npz"
            inner_tx_cache = CACHE_DIR / f"outer{k}_inner{j}_tx.npz"

            if inner_pred_cache.exists():
                z_in = np.load(inner_pred_cache, allow_pickle=True)
                p_m0_fit_oof[va_loc] = z_in["p_m0"]
                for g_idx in range(len(HP_GRID)):
                    hp_grid_oof[g_idx][va_loc] = z_in[f"p_m1_g{g_idx}"]
                d_in = json.loads(str(z_in["diag"]))
                provenance_rows.append(d_in)
                log(f"  [Fold {k+1} Inner {j+1}/4] loaded cached predictions")
                continue

            t_in = time.time()
            mats_in = build_fold_matrices(
                T_indices=Ti,
                eval_indices_list=[Pi],
                splits_T=splits_Ti,
                tr_data=tr_data,
                eval_data_list=[tr_data],
                cache_path=inner_tx_cache,
            )

            # Fit M0 on Ti (3 seeds, 900 trees)
            p_m0_j = fit_predict_lgb_bagged(
                mats_in["X_m0_unseen_T"],
                mats_in["y_T"],
                mats_in["X_m0_unseen_evals"],
                hp_m0,
                SEEDS,
                LGB_TREES,
            )[0]
            p_m0_fit_oof[va_loc] = p_m0_j

            # Evaluate HP_GRID on Ti -> Pi (seed=42, INNER_TUNE_TREES for fast inner grid selection)
            save_dict = {"p_m0": p_m0_j}
            for g_idx, hp_cand in enumerate(HP_GRID):
                p_g = fit_predict_lgb_bagged(
                    mats_in["X_m1_unseen_T"],
                    mats_in["y_T"],
                    mats_in["X_m1_unseen_evals"],
                    hp_cand,
                    [42],
                    INNER_TUNE_TREES,
                )[0]
                hp_grid_oof[g_idx][va_loc] = p_g
                save_dict[f"p_m1_g{g_idx}"] = p_g

            d_in = {
                "model": f"outer{k}_inner{j}",
                "outer_fold": k + 1,
                "inner_fold": j + 1,
                "n_fit": int(len(Ti)),
                "n_pred": int(len(Pi)),
                "fit_rows_sha": sha256_array(np.sort(Ti)),
                "pred_rows_sha": sha256_array(np.sort(Pi)),
                "sha_ptx1_oof": mats_in["sha_ptx1_oof"],
                "sha_ptx2_oof": mats_in["sha_ptx2_oof"],
                "sha_cte_oof": mats_in["sha_cte_oof"],
                "sha_ste_unseen_oof": mats_in["sha_ste_unseen_oof"],
                "sha_ste_causal_oof": mats_in["sha_ste_causal_oof"],
                "sha_m1_te_block": mats_in["sha_m1_te_block"],
                "ptx1_oof_acc": round(mats_in["ptx1_oof_acc"], 5),
                "ptx2_oof_acc": round(mats_in["ptx2_oof_acc"], 5),
                "vocab_w1": mats_in["vocab_w1"],
                "vocab_c1": mats_in["vocab_c1"],
                "vocab_w2": mats_in["vocab_w2"],
                "vocab_c2": mats_in["vocab_c2"],
                "svd_var": round(mats_in["svd_var"], 5),
                "seconds": round(time.time() - t_in, 1),
            }
            save_dict["diag"] = json.dumps(d_in)
            np.savez_compressed(inner_pred_cache, **save_dict)
            provenance_rows.append(d_in)
            log(
                f"  [Fold {k+1} Inner {j+1}/4] M0 acc={accuracy_score(tr_data['y'][Pi], p_m0_j >= 0.5):.4f} "
                f"({d_in['seconds']}s)"
            )
            del mats_in
            gc.collect()

        # Select best HP on inner-OOF of fit_idx (strictly before touching seen_val_idx or unseen_val_idx!)
        y_fit = tr_data["y"][fit_idx]
        hp_scores = []
        for g_idx, hp_cand in enumerate(HP_GRID):
            p_oof_g = np.clip(hp_grid_oof[g_idx], 1e-7, 1.0 - 1e-7)
            ll_g = float(log_loss(y_fit, p_oof_g))
            acc_g = float(accuracy_score(y_fit, p_oof_g >= 0.5))
            hits_g = int(((p_oof_g >= 0.5) == y_fit).sum())
            hp_scores.append({
                "grid_idx": g_idx,
                "hp": hp_cand,
                "inner_oof_logloss": round(ll_g, 6),
                "inner_oof_acc": round(acc_g, 6),
                "inner_oof_hits": hits_g,
            })

        # Sort by inner_oof_logloss ascending (primary), inner_oof_hits descending (tie-breaker)
        hp_scores_sorted = sorted(hp_scores, key=lambda x: (x["inner_oof_logloss"], -x["inner_oof_hits"]))
        best_entry = hp_scores_sorted[0]
        best_hp = best_entry["hp"]
        best_g_idx = best_entry["grid_idx"]

        # Compute full 3-seed bagged inner-OOF for the winning HP (`p_m1_fit_oof`)
        best_inner_3seed_cache = CACHE_DIR / f"outer{k}_best_m1_fit_oof_3seed.npz"
        if best_inner_3seed_cache.exists():
            p_m1_fit_oof = np.load(best_inner_3seed_cache)["p_m1_fit_oof"]
        else:
            p_m1_fit_oof = np.zeros(len(fit_idx), dtype=np.float32)
            for j, (tr_loc, va_loc) in enumerate(inner_splits_local):
                Ti = fit_idx[tr_loc]
                Pi = fit_idx[va_loc]
                sgkf_inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2000 + 10 * k + j)
                splits_Ti = list(sgkf_inner.split(np.zeros(len(Ti)), tr_data["y"][Ti], tr_data["seller_id"][Ti]))
                mats_in = build_fold_matrices(
                    T_indices=Ti,
                    eval_indices_list=[Pi],
                    splits_T=splits_Ti,
                    tr_data=tr_data,
                    eval_data_list=[tr_data],
                    cache_path=CACHE_DIR / f"outer{k}_inner{j}_tx.npz",
                )
                p_m1_fit_oof[va_loc] = fit_predict_lgb_bagged(
                    mats_in["X_m1_unseen_T"],
                    mats_in["y_T"],
                    mats_in["X_m1_unseen_evals"],
                    best_hp,
                    SEEDS,
                    LGB_TREES,
                )[0]
                del mats_in
                gc.collect()
            np.savez_compressed(best_inner_3seed_cache, p_m1_fit_oof=p_m1_fit_oof)

        m0_inner_ll = float(log_loss(y_fit, np.clip(p_m0_fit_oof, 1e-7, 1.0 - 1e-7)))
        m0_inner_hits = int(((p_m0_fit_oof >= 0.5) == y_fit).sum())
        m1_inner_ll = float(log_loss(y_fit, np.clip(p_m1_fit_oof, 1e-7, 1.0 - 1e-7)))
        m1_inner_hits = int(((p_m1_fit_oof >= 0.5) == y_fit).sum())

        freeze_obj = {
            "outer_fold": k + 1,
            "selected_hp": best_hp,
            "n_estimators": LGB_TREES,
            "seeds": SEEDS,
            "m0_inner_oof_hits": m0_inner_hits,
            "m0_inner_oof_acc": round(m0_inner_hits / len(y_fit), 6),
            "m0_inner_oof_logloss": round(m0_inner_ll, 6),
            "m1_3seed_inner_oof_hits": m1_inner_hits,
            "m1_3seed_inner_oof_acc": round(m1_inner_hits / len(y_fit), 6),
            "m1_3seed_inner_oof_logloss": round(m1_inner_ll, 6),
            "hp_grid_results_1seed": hp_scores,
            "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        freeze_path.write_text(json.dumps(freeze_obj, indent=2), encoding="utf-8")
        frozen_list.append(freeze_obj)
        log(
            f"  [Fold {k+1} Freeze] Selected HP={best_hp} | Inner-OOF M0={m0_inner_hits} "
            f"(ll={m0_inner_ll:.5f}) -> M1={m1_inner_hits} (ll={m1_inner_ll:.5f}, "
            f"delta={m1_inner_hits - m0_inner_hits:+d})"
        )

        # Step 2: Fit outer models on full `fit_idx` (~49,574 rows) using `inner_splits_local` for OOF text/TE inside `fit_idx`
        t_out = time.time()
        outer_tx_cache = CACHE_DIR / f"outer{k}_full_tx.npz"
        mats_out = build_fold_matrices(
            T_indices=fit_idx,
            eval_indices_list=[seen_val_idx, unseen_val_idx, test_all_idx],
            splits_T=inner_splits_local,
            tr_data=tr_data,
            eval_data_list=[tr_data, tr_data, te_data],
            cache_path=outer_tx_cache,
        )

        # Fit M0 Unseen Regime model -> evaluate on seen_val, unseen_val, test
        p_m0_unseen_seen, p_m0_unseen_val, p_m0_unseen_test = fit_predict_lgb_bagged(
            mats_out["X_m0_unseen_T"],
            mats_out["y_T"],
            mats_out["X_m0_unseen_evals"],
            hp_m0,
            SEEDS,
            LGB_TREES,
        )

        # Fit M0 Seen Causal-Prior-TE model -> evaluate on seen_val, unseen_val, test
        p_m0_causal_seen, p_m0_causal_unseen, p_m0_causal_test = fit_predict_lgb_bagged(
            mats_out["X_m0_seen_causal_T"],
            mats_out["y_T"],
            mats_out["X_m0_seen_causal_evals"],
            hp_m0,
            SEEDS,
            LGB_TREES,
        )

        # Fit M0 Seen Row-OOF-TE model -> evaluate on seen_val, unseen_val, test
        p_m0_rowte_seen, p_m0_rowte_unseen, p_m0_rowte_test = fit_predict_lgb_bagged(
            mats_out["X_m0_seen_rowte_T"],
            mats_out["y_T"],
            mats_out["X_m0_seen_rowte_evals"],
            hp_m0,
            SEEDS,
            LGB_TREES,
        )

        # Fit M1 Enriched Unseen Regime model -> evaluate on seen_val, unseen_val, test
        p_m1_unseen_seen, p_m1_unseen_val, p_m1_unseen_test = fit_predict_lgb_bagged(
            mats_out["X_m1_unseen_T"],
            mats_out["y_T"],
            mats_out["X_m1_unseen_evals"],
            best_hp,
            SEEDS,
            LGB_TREES,
        )

        # Fit M1 Enriched Seen Causal-Prior-TE model -> evaluate on seen_val, unseen_val, test
        p_m1_causal_seen, p_m1_causal_unseen, p_m1_causal_test = fit_predict_lgb_bagged(
            mats_out["X_m1_seen_causal_T"],
            mats_out["y_T"],
            mats_out["X_m1_seen_causal_evals"],
            best_hp,
            SEEDS,
            LGB_TREES,
        )

        # Fit M1 Enriched Seen Row-OOF-TE model -> evaluate on seen_val, unseen_val, test
        p_m1_rowte_seen, p_m1_rowte_unseen, p_m1_rowte_test = fit_predict_lgb_bagged(
            mats_out["X_m1_seen_rowte_T"],
            mats_out["y_T"],
            mats_out["X_m1_seen_rowte_evals"],
            best_hp,
            SEEDS,
            LGB_TREES,
        )

        # Regime-aware routing for test items: if seller in fit_sellers_set -> use causal seen model, else unseen model
        test_in_fit = np.isin(te_data["seller_id"], list(fit_sellers_set))
        p_m0_dual_test = np.where(test_in_fit, p_m0_causal_test, p_m0_unseen_test).astype(np.float32)
        p_m1_dual_test = np.where(test_in_fit, p_m1_causal_test, p_m1_unseen_test).astype(np.float32)

        # Primary seen_val prediction uses causal prior-items seller_te (leak-free temporal seller history)
        p_m0_seen_val = p_m0_causal_seen
        p_m1_seen_val = p_m1_causal_seen

        # Save per-fold predictions .npz
        np.savez_compressed(
            fold_npz_path,
            fit_idx=fit_idx,
            seen_val_idx=seen_val_idx,
            unseen_val_idx=unseen_val_idx,
            p_m0_fit_oof=p_m0_fit_oof,
            p_m1_fit_oof=p_m1_fit_oof,
            p_m0_seen_val=p_m0_seen_val,
            p_m1_seen_val=p_m1_seen_val,
            p_m0_seen_val_unseen_model=p_m0_unseen_seen,
            p_m1_seen_val_unseen_model=p_m1_unseen_seen,
            p_m0_seen_val_rowte_model=p_m0_rowte_seen,
            p_m1_seen_val_rowte_model=p_m1_rowte_seen,
            p_m0_unseen_val=p_m0_unseen_val,
            p_m1_unseen_val=p_m1_unseen_val,
            p_m0_test=p_m0_dual_test,
            p_m1_test=p_m1_dual_test,
            p_m0_unseen_test=p_m0_unseen_test,
            p_m1_unseen_test=p_m1_unseen_test,
            p_m0_seen_causal_test=p_m0_causal_test,
            p_m1_seen_causal_test=p_m1_causal_test,
            p_m0_seen_rowte_test=p_m0_rowte_test,
            p_m1_seen_rowte_test=p_m1_rowte_test,
            test_in_fit_seller=test_in_fit.astype(np.int8),
        )

        oof_m0_unseen[unseen_val_idx] = p_m0_unseen_val
        oof_m1_unseen[unseen_val_idx] = p_m1_unseen_val

        for idx_pos, r_idx in enumerate(seen_val_idx):
            seen_rows_records.append({
                "outer_fold": k + 1,
                "row": int(r_idx),
                "id": tr_data["id"][r_idx],
                "seller_id": tr_data["seller_id"][r_idx],
                "category_id": tr_data["category_id"][r_idx],
                "y_true": int(tr_data["y"][r_idx]),
                "p_m0_seen_val": float(p_m0_seen_val[idx_pos]),
                "p_m1_seen_val": float(p_m1_seen_val[idx_pos]),
                "p_m0_seen_unseen_model": float(p_m0_unseen_seen[idx_pos]),
                "p_m1_seen_unseen_model": float(p_m1_unseen_seen[idx_pos]),
                "p_m0_seen_rowte_model": float(p_m0_rowte_seen[idx_pos]),
                "p_m1_seen_rowte_model": float(p_m1_rowte_seen[idx_pos]),
            })

        test_preds_per_fold["p_m0_unseen_test"].append(p_m0_unseen_test)
        test_preds_per_fold["p_m0_seen_causal_test"].append(p_m0_causal_test)
        test_preds_per_fold["p_m0_seen_rowte_test"].append(p_m0_rowte_test)
        test_preds_per_fold["p_m1_unseen_test"].append(p_m1_unseen_test)
        test_preds_per_fold["p_m1_seen_causal_test"].append(p_m1_causal_test)
        test_preds_per_fold["p_m1_seen_rowte_test"].append(p_m1_rowte_test)
        test_preds_per_fold["p_m0_dual_test"].append(p_m0_dual_test)
        test_preds_per_fold["p_m1_dual_test"].append(p_m1_dual_test)
        test_preds_per_fold["test_in_fit_seller"].append(test_in_fit.astype(np.float32))

        d_out = {
            "model": f"outer{k}_full_fit",
            "outer_fold": k + 1,
            "inner_fold": 0,
            "n_fit": int(len(fit_idx)),
            "n_pred": int(len(unseen_val_idx)),
            "fit_rows_sha": sha256_array(np.sort(fit_idx)),
            "pred_rows_sha": sha256_array(np.sort(unseen_val_idx)),
            "sha_ptx1_oof": mats_out["sha_ptx1_oof"],
            "sha_ptx2_oof": mats_out["sha_ptx2_oof"],
            "sha_cte_oof": mats_out["sha_cte_oof"],
            "sha_ste_unseen_oof": mats_out["sha_ste_unseen_oof"],
            "sha_ste_causal_oof": mats_out["sha_ste_causal_oof"],
            "sha_m1_te_block": mats_out["sha_m1_te_block"],
            "ptx1_oof_acc": round(mats_out["ptx1_oof_acc"], 5),
            "ptx2_oof_acc": round(mats_out["ptx2_oof_acc"], 5),
            "vocab_w1": mats_out["vocab_w1"],
            "vocab_c1": mats_out["vocab_c1"],
            "vocab_w2": mats_out["vocab_w2"],
            "vocab_c2": mats_out["vocab_c2"],
            "svd_var": round(mats_out["svd_var"], 5),
            "seconds": round(time.time() - t_out, 1),
        }
        provenance_rows.append(d_out)

        # Per-fold evaluation metrics
        y_u = tr_data["y"][unseen_val_idx]
        y_s = tr_data["y"][seen_val_idx]

        c_m0_u = ((p_m0_unseen_val >= 0.5) == y_u)
        c_m1_u = ((p_m1_unseen_val >= 0.5) == y_u)
        c_m0_s = ((p_m0_seen_val >= 0.5) == y_s)
        c_m1_s = ((p_m1_seen_val >= 0.5) == y_s)

        c_m0_s_unseen_m = ((p_m0_unseen_seen >= 0.5) == y_s)
        c_m1_s_unseen_m = ((p_m1_unseen_seen >= 0.5) == y_s)
        c_m0_s_rowte_m = ((p_m0_rowte_seen >= 0.5) == y_s)
        c_m1_s_rowte_m = ((p_m1_rowte_seen >= 0.5) == y_s)

        fold_row = {
            "status": "DEVELOPMENT_ONLY",
            "outer_fold": k + 1,
            "n_fit": int(len(fit_idx)),
            "n_seen_val": int(len(seen_val_idx)),
            "n_unseen_val": int(len(unseen_val_idx)),
            "selected_num_leaves": best_hp["num_leaves"],
            "selected_colsample_bytree": best_hp["colsample_bytree"],
            "selected_min_child_samples": best_hp["min_child_samples"],
            "fit_oof_hits_m0": m0_inner_hits,
            "fit_oof_hits_m1": m1_inner_hits,
            "fit_oof_delta_m1_vs_m0": m1_inner_hits - m0_inner_hits,
            "fit_oof_ll_m0": round(m0_inner_ll, 5),
            "fit_oof_ll_m1": round(m1_inner_ll, 5),
            "unseen_hits_m0": int(c_m0_u.sum()),
            "unseen_hits_m1": int(c_m1_u.sum()),
            "unseen_acc_m0": round(float(c_m0_u.mean()), 5),
            "unseen_acc_m1": round(float(c_m1_u.mean()), 5),
            "unseen_fixes_m1_vs_m0": int((c_m1_u & ~c_m0_u).sum()),
            "unseen_breaks_m1_vs_m0": int((~c_m1_u & c_m0_u).sum()),
            "unseen_delta_m1_vs_m0": int(c_m1_u.sum()) - int(c_m0_u.sum()),
            "unseen_ll_m0": round(float(log_loss(y_u, np.clip(p_m0_unseen_val, 1e-7, 1.0 - 1e-7))), 5),
            "unseen_ll_m1": round(float(log_loss(y_u, np.clip(p_m1_unseen_val, 1e-7, 1.0 - 1e-7))), 5),
            "seen_causal_hits_m0": int(c_m0_s.sum()),
            "seen_causal_hits_m1": int(c_m1_s.sum()),
            "seen_causal_acc_m0": round(float(c_m0_s.mean()), 5),
            "seen_causal_acc_m1": round(float(c_m1_s.mean()), 5),
            "seen_causal_fixes_m1_vs_m0": int((c_m1_s & ~c_m0_s).sum()),
            "seen_causal_breaks_m1_vs_m0": int((~c_m1_s & c_m0_s).sum()),
            "seen_causal_delta_m1_vs_m0": int(c_m1_s.sum()) - int(c_m0_s.sum()),
            "seen_causal_ll_m0": round(float(log_loss(y_s, np.clip(p_m0_seen_val, 1e-7, 1.0 - 1e-7))), 5),
            "seen_causal_ll_m1": round(float(log_loss(y_s, np.clip(p_m1_seen_val, 1e-7, 1.0 - 1e-7))), 5),
            "seen_unseen_model_hits_m0": int(c_m0_s_unseen_m.sum()),
            "seen_unseen_model_hits_m1": int(c_m1_s_unseen_m.sum()),
            "seen_unseen_model_ll_m0": round(float(log_loss(y_s, np.clip(p_m0_unseen_seen, 1e-7, 1.0 - 1e-7))), 5),
            "seen_unseen_model_ll_m1": round(float(log_loss(y_s, np.clip(p_m1_unseen_seen, 1e-7, 1.0 - 1e-7))), 5),
            "seen_rowte_model_hits_m0": int(c_m0_s_rowte_m.sum()),
            "seen_rowte_model_hits_m1": int(c_m1_s_rowte_m.sum()),
            "seen_rowte_model_ll_m0": round(float(log_loss(y_s, np.clip(p_m0_rowte_seen, 1e-7, 1.0 - 1e-7))), 5),
            "seen_rowte_model_ll_m1": round(float(log_loss(y_s, np.clip(p_m1_rowte_seen, 1e-7, 1.0 - 1e-7))), 5),
            "fold_seconds": round(time.time() - t_fold, 1),
        }
        per_fold_rows.append(fold_row)
        log(
            f"  [Fold {k+1} Outer] Unseen: M0={fold_row['unseen_hits_m0']} -> M1={fold_row['unseen_hits_m1']} "
            f"({fold_row['unseen_delta_m1_vs_m0']:+d}, ll {fold_row['unseen_ll_m0']:.5f}->{fold_row['unseen_ll_m1']:.5f}) | "
            f"Seen(causal): M0={fold_row['seen_causal_hits_m0']} -> M1={fold_row['seen_causal_hits_m1']} "
            f"({fold_row['seen_causal_delta_m1_vs_m0']:+d}, ll {fold_row['seen_causal_ll_m0']:.5f}->{fold_row['seen_causal_ll_m1']:.5f})"
        )
        del mats_out
        gc.collect()

    # Save provenance & per-fold CSVs
    df_prov = pd.DataFrame(provenance_rows)
    df_prov["status"] = "DEVELOPMENT_ONLY"
    df_prov.to_csv(OUT_DIR / "model_provenance.csv", index=False)

    # Load T97 outer validation predictions for paired comparisons
    t97_df = pd.read_csv(ROOT / "results/seller_group_nested_podium_20261111/paired_outer_val_predictions.csv")
    assert np.array_equal(t97_df["row"].values, tr_data["row"])
    p_t97_lgb = t97_df["p_lgb"].values.astype(np.float32)
    p_t97_polA = t97_df["p_A_podium_recipe"].values.astype(np.float32)

    # Augment per_fold_rows with T97 comparisons on unseen_val
    for k in range(5):
        uk = np.where(df_splits["outer_fold"].values == k)[0]
        yu = tr_data["y"][uk]
        c_m1 = (oof_m1_unseen[uk] >= 0.5) == yu
        c_m0 = (oof_m0_unseen[uk] >= 0.5) == yu
        c_t97_lgb = (p_t97_lgb[uk] >= 0.5) == yu
        c_t97_polA = (p_t97_polA[uk] >= 0.5) == yu

        per_fold_rows[k]["unseen_hits_t97_lgb"] = int(c_t97_lgb.sum())
        per_fold_rows[k]["unseen_hits_t97_polA"] = int(c_t97_polA.sum())
        per_fold_rows[k]["unseen_delta_m0_vs_t97_lgb"] = int(c_m0.sum()) - int(c_t97_lgb.sum())
        per_fold_rows[k]["unseen_delta_m1_vs_t97_lgb"] = int(c_m1.sum()) - int(c_t97_lgb.sum())
        per_fold_rows[k]["unseen_delta_m1_vs_t97_polA"] = int(c_m1.sum()) - int(c_t97_polA.sum())

    df_per_fold = pd.DataFrame(per_fold_rows)
    df_per_fold.to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)

    # Save oof_unseen_70k.csv
    df_oof_unseen = pd.DataFrame({
        "row": tr_data["row"],
        "id": tr_data["id"],
        "seller_id": tr_data["seller_id"],
        "category_id": tr_data["category_id"],
        "outer_fold": df_splits["outer_fold"].values + 1,
        "y_true": tr_data["y"],
        "p_m0_unseen": oof_m0_unseen,
        "p_m1_unseen": oof_m1_unseen,
        "p_t97_lgb": p_t97_lgb,
        "p_t97_policy_A": p_t97_polA,
    })
    oof_unseen_path = OUT_DIR / "oof_unseen_70k.csv"
    df_oof_unseen.to_csv(oof_unseen_path, index=False)

    # Save oof_seen_all_folds.csv (32,092 rows across 5 folds)
    df_oof_seen = pd.DataFrame(seen_rows_records)
    # Also attach T97 outer-val predictions on those rows for reference
    row_idx_seen = df_oof_seen["row"].values
    df_oof_seen["p_t97_lgb_outer_oof"] = p_t97_lgb[row_idx_seen]
    df_oof_seen["p_t97_policy_A_outer_oof"] = p_t97_polA[row_idx_seen]
    oof_seen_path = OUT_DIR / "oof_seen_all_folds.csv"
    df_oof_seen.to_csv(oof_seen_path, index=False)

    # Save test_predictions_30k.csv and test_disagreements_vs_s4_probs_only.csv (ZERO inferred labels!)
    p_m0_test_mean = np.mean(test_preds_per_fold["p_m0_dual_test"], axis=0)
    p_m1_test_mean = np.mean(test_preds_per_fold["p_m1_dual_test"], axis=0)
    p_m0_unseen_test_mean = np.mean(test_preds_per_fold["p_m0_unseen_test"], axis=0)
    p_m1_unseen_test_mean = np.mean(test_preds_per_fold["p_m1_unseen_test"], axis=0)
    p_m0_causal_test_mean = np.mean(test_preds_per_fold["p_m0_seen_causal_test"], axis=0)
    p_m1_causal_test_mean = np.mean(test_preds_per_fold["p_m1_seen_causal_test"], axis=0)
    test_seen_frac = np.mean(test_preds_per_fold["test_in_fit_seller"], axis=0)

    s4_test_df = pd.read_csv(ROOT / "results/ultimate_podium_engine_23h/test_probabilities_podium_engine.csv")
    assert len(s4_test_df) == 30000

    df_test_preds = pd.DataFrame({
        "ID": s4_test_df["ID"].values,
        "p_m0_test": p_m0_test_mean,
        "p_m1_test": p_m1_test_mean,
        "p_m0_unseen_test": p_m0_unseen_test_mean,
        "p_m1_unseen_test": p_m1_unseen_test_mean,
        "p_m0_seen_causal_test": p_m0_causal_test_mean,
        "p_m1_seen_causal_test": p_m1_causal_test_mean,
        "fit_seller_fold_frac": test_seen_frac,
    })
    test_preds_path = OUT_DIR / "test_predictions_30k.csv"
    df_test_preds.to_csv(test_preds_path, index=False)

    # Filter test rows where M1 or M0 probability crosses 0.5 relative to S4 p_optimal_blend or p_lgb,
    # saving ONLY ID and model probabilities (zero inferred binary/string labels!).
    s4_p_blend = s4_test_df["p_optimal_blend"].values
    s4_p_lgb = s4_test_df["p_lgb"].values
    disagree_mask = (
        ((p_m1_test_mean >= 0.5) != (s4_p_blend >= 0.5))
        | ((p_m1_unseen_test_mean >= 0.5) != (s4_p_blend >= 0.5))
        | ((p_m1_test_mean >= 0.5) != (s4_p_lgb >= 0.5))
        | ((p_m1_test_mean >= 0.5) != (p_m0_test_mean >= 0.5))
    )
    df_disagree = pd.DataFrame({
        "ID": s4_test_df["ID"].values[disagree_mask],
        "p_m0_test": p_m0_test_mean[disagree_mask],
        "p_m1_test": p_m1_test_mean[disagree_mask],
        "p_m0_unseen_test": p_m0_unseen_test_mean[disagree_mask],
        "p_m1_unseen_test": p_m1_unseen_test_mean[disagree_mask],
        "p_m0_seen_causal_test": p_m0_causal_test_mean[disagree_mask],
        "p_m1_seen_causal_test": p_m1_causal_test_mean[disagree_mask],
        "p_s4_lgb": s4_p_lgb[disagree_mask],
        "p_s4_optimal_blend": s4_p_blend[disagree_mask],
    })
    disagree_path = OUT_DIR / "test_disagreements_vs_s4_probs_only.csv"
    df_disagree.to_csv(disagree_path, index=False)

    # Paired evaluations on unseen_val (70,000 rows) and seen_val (32,092 rows across 5 folds)
    y_all = tr_data["y"]
    sellers_all = tr_data["seller_id"]
    cats_all = tr_data["category_id"]
    folds_all = df_splits["outer_fold"].values + 1

    unseen_comparisons = {
        "M1_vs_M0_unseen_70k": paired_comparison(
            y_all, oof_m1_unseen, oof_m0_unseen, folds_all, sellers_all, cats_all, "M1_lgb_enriched vs M0_lgb_cx0164_clean (Unseen 70k)"
        ),
        "M0_vs_T97_LGB_unseen_70k": paired_comparison(
            y_all, oof_m0_unseen, p_t97_lgb, folds_all, sellers_all, cats_all, "M0_lgb_cx0164_clean vs T97_LightGBM (Unseen 70k)"
        ),
        "M1_vs_T97_LGB_unseen_70k": paired_comparison(
            y_all, oof_m1_unseen, p_t97_lgb, folds_all, sellers_all, cats_all, "M1_lgb_enriched vs T97_LightGBM (Unseen 70k)"
        ),
        "M1_vs_T97_PolicyA_unseen_70k": paired_comparison(
            y_all, oof_m1_unseen, p_t97_polA, folds_all, sellers_all, cats_all, "M1_lgb_enriched vs T97_Policy_A (Unseen 70k)"
        ),
        "M0_vs_T97_PolicyA_unseen_70k": paired_comparison(
            y_all, oof_m0_unseen, p_t97_polA, folds_all, sellers_all, cats_all, "M0_lgb_cx0164_clean vs T97_Policy_A (Unseen 70k)"
        ),
    }

    y_seen_all = df_oof_seen["y_true"].values
    s_seen_all = df_oof_seen["seller_id"].values
    c_seen_all = df_oof_seen["category_id"].values
    f_seen_all = df_oof_seen["outer_fold"].values
    p_m0_seen_all = df_oof_seen["p_m0_seen_val"].values
    p_m1_seen_all = df_oof_seen["p_m1_seen_val"].values
    p_m0_seen_unseen_m_all = df_oof_seen["p_m0_seen_unseen_model"].values
    p_m1_seen_unseen_m_all = df_oof_seen["p_m1_seen_unseen_model"].values
    p_m0_seen_rowte_m_all = df_oof_seen["p_m0_seen_rowte_model"].values
    p_m1_seen_rowte_m_all = df_oof_seen["p_m1_seen_rowte_model"].values

    seen_comparisons = {
        "M1_causal_vs_M0_causal_seen_32092": paired_comparison(
            y_seen_all, p_m1_seen_all, p_m0_seen_all, f_seen_all, s_seen_all, c_seen_all,
            "M1_lgb_enriched (causal TE) vs M0_lgb_cx0164_clean (causal TE) (Seen 32,092)"
        ),
        "M1_unseen_model_vs_M0_unseen_model_seen_32092": paired_comparison(
            y_seen_all, p_m1_seen_unseen_m_all, p_m0_seen_unseen_m_all, f_seen_all, s_seen_all, c_seen_all,
            "M1_lgb_enriched (unseen-regime model) vs M0_lgb_cx0164_clean (unseen-regime model) (Seen 32,092)"
        ),
        "M1_rowte_vs_M0_rowte_seen_32092": paired_comparison(
            y_seen_all, p_m1_seen_rowte_m_all, p_m0_seen_rowte_m_all, f_seen_all, s_seen_all, c_seen_all,
            "M1_lgb_enriched (row OOF TE) vs M0_lgb_cx0164_clean (row OOF TE) (Seen 32,092)"
        ),
        "M1_causal_vs_M1_unseen_model_seen_32092": paired_comparison(
            y_seen_all, p_m1_seen_all, p_m1_seen_unseen_m_all, f_seen_all, s_seen_all, c_seen_all,
            "M1_lgb_enriched (causal seller TE) vs M1_lgb_enriched (prior seller TE) (Seen 32,092)"
        ),
        "M1_causal_vs_T97_LGB_outer_oof_seen_32092": paired_comparison(
            y_seen_all, p_m1_seen_all, df_oof_seen["p_t97_lgb_outer_oof"].values, f_seen_all, s_seen_all, c_seen_all,
            "M1_lgb_enriched (causal TE) vs T97_LightGBM (outer unseen OOF on same items) (Seen 32,092)"
        ),
        "M1_causal_vs_T97_PolicyA_outer_oof_seen_32092": paired_comparison(
            y_seen_all, p_m1_seen_all, df_oof_seen["p_t97_policy_A_outer_oof"].values, f_seen_all, s_seen_all, c_seen_all,
            "M1_lgb_enriched (causal TE) vs T97_Policy_A (outer unseen OOF on same items) (Seen 32,092)"
        ),
    }

    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git_head = "unknown"

    manifest = {
        "status": "DEVELOPMENT_ONLY",
        "task": "T98 Line 3 — Enriched Leak-Free LightGBM Specialist + CX-0164 Corrected Stacked Features Audit",
        "submission_generated": False,
        "kaggle_api_used": False,
        "git_head_at_run": git_head,
        "script": "scripts/t98_line3_lightgbm_enriched.py",
        "script_sha256": sha256_file(Path(__file__)),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "cx0164_compliance": {
            "in_sample_text_feature_eliminated": True,
            "text_oof_method": "4-fold StratifiedGroupKFold by seller_id strictly inside each training set T",
            "unseen_seller_te_method": "4-fold StratifiedGroupKFold by seller_id inside T (equals fold prior everywhere, eliminating train/val distribution shift)",
            "seen_seller_te_methods_recorded": [
                "causal_prior_items_te (chronological prior items of seller in T; first item gets prior, seen_val gets all T items)",
                "row_skf4_oof_te (4-fold StratifiedKFold inside T matched to full-T seller_te on seen_val)",
                "unseen_regime_prior_te (seller_te fixed to fold prior on both T and seen_val)",
            ],
            "outer_fit_size_matched_to_t98_dual_regime": True,
        },
        "frozen_outer_folds": frozen_list,
        "unseen_70k_comparisons": unseen_comparisons,
        "seen_32092_comparisons": seen_comparisons,
        "artifacts": {
            "oof_unseen_70k_csv": str(oof_unseen_path.relative_to(ROOT)),
            "oof_unseen_70k_sha256": sha256_file(oof_unseen_path),
            "oof_seen_all_folds_csv": str(oof_seen_path.relative_to(ROOT)),
            "oof_seen_all_folds_sha256": sha256_file(oof_seen_path),
            "test_predictions_30k_csv": str(test_preds_path.relative_to(ROOT)),
            "test_predictions_30k_sha256": sha256_file(test_preds_path),
            "test_disagreements_vs_s4_probs_only_csv": str(disagree_path.relative_to(ROOT)),
            "test_disagreements_vs_s4_probs_only_sha256": sha256_file(disagree_path),
            "per_fold_metrics_csv": str((OUT_DIR / "per_fold_metrics.csv").relative_to(ROOT)),
            "per_fold_metrics_sha256": sha256_file(OUT_DIR / "per_fold_metrics.csv"),
            "model_provenance_csv": str((OUT_DIR / "model_provenance.csv").relative_to(ROOT)),
            "model_provenance_sha256": sha256_file(OUT_DIR / "model_provenance.csv"),
        },
        "runtime_minutes": round((time.time() - t_start) / 60.0, 2),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("=== SUMMARY OF PAIRED COMPARISONS ===")
    log(json.dumps({"unseen_70k": unseen_comparisons, "seen_32092": seen_comparisons}, indent=2))
    log(f"ALL DONE in {manifest['runtime_minutes']} min. Outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
