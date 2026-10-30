"""Shared Hash-Verified Feature & Split Cache for Intensive Multi-Agent Work (2026-10-30).

Provides:
1. Fast loading (<1s) of pre-extracted 70k train and 30k test features, categoricals,
   text representations (both historical `combos_hist` and variant `combos_valname`),
   IDs, seller IDs, and labels (`y_used`, where used=1, new=0).
2. Both:
   - Exact historical R151 101-feature matrix (`F_r151_tr`) matching `extract_deep_features` bit-for-bit.
   - Strictly inductive group aggregation function `compute_inductive_group_features(F_row_tr, cat_tr, fit_idx, eval_dfs_and_cats)`
     so that external validation (Agent 3) and 70k->30k test inference never leak group statistics across splits.
"""
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import (
    extract_deep_features, lst
)

TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
CACHE_DIR = ROOT / "results" / "cache_20261030"
CACHE_FILE = CACHE_DIR / "shared_feature_cache.pkl"
MANIFEST_FILE = CACHE_DIR / "cache_manifest.json"

CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]

GROUP_AGG_COLS = [
    "freq_seller_id", "freq_category_id", "freq_city",
    "seller_n_cats", "seller_mean_logprice", "seller_mean_init_q",
    "sell_mean_w_null", "sell_mean_t_usado", "sell_mean_t_nuevo",
    "sell_mean_ship_free", "sell_mean_auto_relist", "sell_mean_n_pics",
    "rel_price_cat"
]


def compute_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_array_sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def build_valname_combos(rows):
    combos = []
    for r in rows:
        parts = [r.get("title") or ""]
        if r.get("warranty"):
            parts.append(str(r["warranty"]))
        attrs = r.get("attributes")
        if isinstance(attrs, list):
            for a in attrs:
                v = a.get("value_name")
                if v:
                    parts.append(str(v))
        combos.append(" ".join(parts))
    return combos


def compute_inductive_group_features(F_base_tr, cat_tr, fit_idx, target_list):
    """Computes the 13 unsupervised seller/category/city features strictly from `fit_idx` of train,
    and applies them inductively to each `(F_base_target, cat_target)` in `target_list`.
    For unseen sellers in target splits, seller-level means fall back to the item's own row value
    (since a singleton seller of size 1 has seller_mean == item_value!).
    """
    fit_cat = cat_tr.iloc[fit_idx].reset_index(drop=True)
    fit_num = F_base_tr.iloc[fit_idx].copy().reset_index(drop=True)
    fit_num["seller_id"] = fit_cat["seller_id"]
    fit_num["category_id"] = fit_cat["category_id"]

    # 1. Frequencies in fit
    freq_maps = {}
    for col in ["seller_id", "category_id", "city"]:
        freq_maps[col] = fit_cat[col].value_counts()

    # 2. Seller aggregations in fit
    g_seller = fit_num.groupby("seller_id", observed=True)
    seller_stats = pd.DataFrame({
        "seller_n_cats": g_seller["category_id"].nunique(),
        "seller_mean_logprice": g_seller["log_price"].mean(),
        "seller_mean_init_q": g_seller["sold_ratio"].mean(),
        "sell_mean_w_null": g_seller["warranty_null"].mean(),
        "sell_mean_t_usado": g_seller["t_usado"].mean(),
        "sell_mean_t_nuevo": g_seller["t_nuevo"].mean(),
        "sell_mean_ship_free": g_seller["ship_free"].mean(),
        "sell_mean_auto_relist": g_seller["automatic_relist"].mean(),
        "sell_mean_n_pics": g_seller["n_pics"].mean(),
    })

    # 3. Category median log_price in fit
    cat_med_map = fit_num.groupby("category_id", observed=True)["log_price"].median()
    global_med_logprice = float(fit_num["log_price"].median())

    results = []
    for F_target, cat_target in target_list:
        out = F_target.copy().reset_index(drop=True)
        c_df = cat_target.reset_index(drop=True)

        # Frequencies (unseen in fit -> 1.0)
        for col in ["seller_id", "category_id", "city"]:
            out[f"freq_{col}"] = c_df[col].map(freq_maps[col]).fillna(1.0).values.astype(np.float32)

        # Seller stats (unseen seller -> singleton fallback = item's own feature value)
        s_series = c_df["seller_id"]
        out["seller_n_cats"] = s_series.map(seller_stats["seller_n_cats"]).fillna(1.0).values.astype(np.float32)

        fallback_pairs = [
            ("seller_mean_logprice", "log_price"),
            ("seller_mean_init_q", "sold_ratio"),
            ("sell_mean_w_null", "warranty_null"),
            ("sell_mean_t_usado", "t_usado"),
            ("sell_mean_t_nuevo", "t_nuevo"),
            ("sell_mean_ship_free", "ship_free"),
            ("sell_mean_auto_relist", "automatic_relist"),
            ("sell_mean_n_pics", "n_pics"),
        ]
        for s_col, item_col in fallback_pairs:
            mapped = s_series.map(seller_stats[s_col])
            out[s_col] = mapped.fillna(out[item_col]).values.astype(np.float32)

        # Relative price to category median
        c_med = c_df["category_id"].map(cat_med_map).fillna(global_med_logprice).values.astype(np.float32)
        out["rel_price_cat"] = (out["log_price"].values - c_med).astype(np.float32)

        # Ensure exact column order matching F_base_tr
        for col in CAT_FEATURES:
            out[col] = c_df[col].astype(str).values
        out = out[list(F_base_tr.columns)]
        results.append(out)

    return results


def build_or_load_cache(force_rebuild=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and MANIFEST_FILE.exists() and not force_rebuild:
        t0 = time.time()
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded shared feature cache in {time.time()-t0:.2f}s", flush=True)
        return data

    t0 = time.time()
    print("Building shared feature cache from raw JSONL files...", flush=True)
    train_sha = compute_file_sha256(TRAIN_JSON)
    test_sha = compute_file_sha256(TEST_JSON)

    rows_tr = [json.loads(line) for line in open(TRAIN_JSON, "r", encoding="utf-8")]
    rows_te = [json.loads(line) for line in open(TEST_JSON, "r", encoding="utf-8")]

    ids_tr = np.array([str(r["id"]) for r in rows_tr])
    sellers_tr = np.array([str(r["seller_id"]) for r in rows_tr])
    y_tr = np.array([1 if r.get("condition") == "used" else 0 for r in rows_tr], dtype=int)

    ids_te = np.arange(1, len(rows_te) + 1, dtype=int)
    item_ids_te = np.array([str(r["id"]) for r in rows_te])
    sellers_te = np.array([str(r["seller_id"]) for r in rows_te])

    # Extract exact R151 features on 70k train
    F_r151_tr, cat_tr, titles_tr, warranties_tr, combos_hist_tr = extract_deep_features(rows_tr)
    combos_valname_tr = build_valname_combos(rows_tr)

    # Extract base features on 30k test, then overwrite group agg columns inductively from 70k train!
    F_raw_te, cat_te, titles_te, warranties_te, combos_hist_te = extract_deep_features(rows_te)
    combos_valname_te = build_valname_combos(rows_te)

    # Inductive 70k -> 30k test feature matrix (using only 70k train for seller/cat/city stats)
    all_tr_idx = np.arange(len(rows_tr))
    F_ind_te = compute_inductive_group_features(
        F_r151_tr, cat_tr, all_tr_idx, [(F_raw_te, cat_te)]
    )[0]

    cache_data = {
        "train_sha256": train_sha,
        "test_sha256": test_sha,
        "ids_tr": ids_tr,
        "sellers_tr": sellers_tr,
        "y_tr": y_tr,
        "F_r151_tr": F_r151_tr,
        "cat_tr": cat_tr,
        "titles_tr": titles_tr,
        "warranties_tr": warranties_tr,
        "combos_hist_tr": combos_hist_tr,
        "combos_valname_tr": combos_valname_tr,
        "ids_te": ids_te,
        "item_ids_te": item_ids_te,
        "sellers_te": sellers_te,
        "F_ind_te": F_ind_te,
        "cat_te": cat_te,
        "titles_te": titles_te,
        "warranties_te": warranties_te,
        "combos_hist_te": combos_hist_te,
        "combos_valname_te": combos_valname_te,
        "feature_columns_101": list(F_r151_tr.columns),
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_json_sha256": train_sha,
        "test_json_sha256": test_sha,
        "n_train": len(rows_tr),
        "n_test": len(rows_te),
        "n_features_base": F_r151_tr.shape[1],
        "feature_columns": list(F_r151_tr.columns),
        "y_tr_mean_used": float(y_tr.mean()),
        "y_tr_sum_used": int(y_tr.sum()),
        "y_tr_sha256": compute_array_sha256(y_tr),
        "sellers_tr_sha256": compute_array_sha256(sellers_tr.astype("U")),
        "sellers_te_sha256": compute_array_sha256(sellers_te.astype("U")),
        "cache_file_sha256": compute_file_sha256(CACHE_FILE),
        "build_time_seconds": round(time.time() - t0, 2),
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Built and saved shared cache in {time.time()-t0:.2f}s (SHA256: {manifest['cache_file_sha256'][:12]}...)", flush=True)
    return cache_data


if __name__ == "__main__":
    build_or_load_cache(force_rebuild=True)
