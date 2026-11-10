#!/usr/bin/env python3
"""Specialized High-Capacity CatBoost Classifier (depth=8, lr=0.04, l2=5.0, iters=700).

Author: ML Tree Specialist
Date: 2026-11-10

Validation & Leakage Protocol:
- Outer StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped strictly by seller_id on 70,000 train items.
- Exactly ZERO seller overlap between train and validation.
- ZERO seller features, ZERO seller target encoding (100% item-intrinsic features).
- Out-of-fold category price statistics computed strictly inside each outer-train fold.
- Resource control: Limit memory to < 4 GB RAM via sequential fold processing and gc.collect().

Outputs:
- results/agent_catboost_depth8_20261110/oof_catboost.csv
- results/agent_catboost_depth8_20261110/test_probabilities_catboost.csv
- results/agent_catboost_depth8_20261110/summary_report.json
"""

import gc
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
OUT_DIR = ROOT / "results" / "agent_catboost_depth8_20261110"
SVD_CACHE = ROOT / "results" / "deep_xgboost_depth9_20261110" / "svd_200_features.npz"


def log(msg: str):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)


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


def extract_intrinsic_features(rows):
    """Extract item intrinsic features (STRICTLY ZERO SELLER_ID)."""
    t0 = time.time()
    feats = {}
    N = len(rows)

    # 1. Price and Quantities
    price = np.array([float(r.get("price") or 0.0) for r in rows], dtype=np.float32)
    base_price = np.array([float(r.get("base_price") or 0.0) for r in rows], dtype=np.float32)
    init_q = np.array([float(r.get("initial_quantity") or 0.0) for r in rows], dtype=np.float32)
    avail_q = np.array([float(r.get("available_quantity") or 0.0) for r in rows], dtype=np.float32)
    sold_q = np.array([float(r.get("sold_quantity") or 0.0) for r in rows], dtype=np.float32)

    feats["price"] = price
    feats["base_price"] = base_price
    feats["initial_quantity"] = init_q
    feats["available_quantity"] = avail_q
    feats["sold_quantity"] = sold_q
    feats["log_price"] = np.log1p(np.maximum(0.0, price))
    feats["log_base_price"] = np.log1p(np.maximum(0.0, base_price))
    feats["log_iq"] = np.log1p(np.maximum(0.0, init_q))
    feats["log_aq"] = np.log1p(np.maximum(0.0, avail_q))
    feats["log_sq"] = np.log1p(np.maximum(0.0, sold_q))
    feats["price_eq_base"] = (price == base_price).astype(np.float32)
    feats["price_diff"] = (base_price - price).astype(np.float32)
    feats["discount_rate"] = np.clip((base_price - price) / (base_price + 1e-4), -1.0, 1.0).astype(np.float32)
    feats["sold_ratio"] = (sold_q / (init_q + 1.0)).astype(np.float32)
    feats["avail_ratio"] = (avail_q / (init_q + 1.0)).astype(np.float32)
    feats["init_minus_avail"] = (init_q - avail_q).astype(np.float32)
    feats["price_cents"] = ((price * 100) % 100).round().astype(np.float32)
    feats["price_round100"] = ((price % 100) == 0).astype(np.float32)
    feats["price_round10"] = ((price % 10) == 0).astype(np.float32)
    feats["price_end9"] = ((np.round(price) % 10) == 9).astype(np.float32)

    # 2. Structural Listing Metadata
    feats["has_original_price"] = np.array([1.0 if r.get("original_price") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["official_store"] = np.array([1.0 if r.get("official_store_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_video"] = np.array([1.0 if r.get("video_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_catalog"] = np.array([1.0 if r.get("catalog_product_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_parent"] = np.array([1.0 if r.get("parent_item_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["automatic_relist"] = np.array([1.0 if r.get("automatic_relist") else 0.0 for r in rows], dtype=np.float32)
    feats["accepts_mp"] = np.array([1.0 if r.get("accepts_mercadopago") else 0.0 for r in rows], dtype=np.float32)
    feats["n_deals"] = np.array([float(len(lst(r.get("deal_ids")))) for r in rows], dtype=np.float32)
    feats["currency_usd"] = np.array([1.0 if r.get("currency_id") == "USD" else 0.0 for r in rows], dtype=np.float32)

    # 3. Tags
    for t_tag in ["dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits", "free_relist", "poor_quality_thumbnail"]:
        feats[f"tag_{t_tag}"] = np.array([1.0 if t_tag in lst(r.get("tags")) else 0.0 for r in rows], dtype=np.float32)
    feats["n_tags"] = np.array([float(len(lst(r.get("tags")))) for r in rows], dtype=np.float32)

    # 4. Payment Methods
    for code in ["MLATB", "MLAWC", "MLAMO", "MLAOT", "MLACD", "MLABC", "MLAVE", "MLAMC", "MLAAM", "MLADC", "MLAVS", "MLAMS", "MLAWT", "MLAMP", "MLAAB"]:
        feats[f"pm_{code}"] = np.array([1.0 if code in [m.get("id") for m in lst(r.get("non_mercado_pago_payment_methods"))] else 0.0 for r in rows], dtype=np.float32)
    feats["n_pm"] = np.array([float(len(lst(r.get("non_mercado_pago_payment_methods")))) for r in rows], dtype=np.float32)

    # 5. Shipping Metadata
    shipping = [r.get("shipping") or {} for r in rows]
    feats["free_shipping"] = np.array([1.0 if sh.get("free_shipping") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_local_pick_up"] = np.array([1.0 if sh.get("local_pick_up") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_dims"] = np.array([1.0 if sh.get("dimensions") is not None else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_n_methods"] = np.array([float(len(lst(sh.get("methods")))) for sh in shipping], dtype=np.float32)
    feats["ship_n_tags"] = np.array([float(len(lst(sh.get("tags")))) for sh in shipping], dtype=np.float32)

    ship_modes = [str(sh.get("mode") or "") for sh in shipping]
    feats["ship_mode_me1"] = np.array([1.0 if sm == "me1" else 0.0 for sm in ship_modes], dtype=np.float32)
    feats["ship_mode_me2"] = np.array([1.0 if sm == "me2" else 0.0 for sm in ship_modes], dtype=np.float32)
    feats["ship_mode_custom"] = np.array([1.0 if sm == "custom" else 0.0 for sm in ship_modes], dtype=np.float32)
    feats["ship_mode_not_specified"] = np.array([1.0 if sm == "not_specified" else 0.0 for sm in ship_modes], dtype=np.float32)

    # 6. Pictures
    n_pics, pic0_w, pic0_h, pic_max_area, pic_mean_area, pic_square = [], [], [], [], [], []
    pic_prefix, pic_code, pic_code_min, pic_code_max, thumb_code = [], [], [], [], []
    for r in rows:
        pics = lst(r.get("pictures"))
        n_pics.append(float(len(pics)))
        if pics:
            p0 = dims(pics[0])
            pic0_w.append(float(p0[0]))
            pic0_h.append(float(p0[1]))
            pic_square.append(1.0 if p0[0] == p0[1] and p0[0] > 0 else 0.0)
            areas = [dims(p)[0] * dims(p)[1] for p in pics]
            pic_max_area.append(float(max(areas)))
            pic_mean_area.append(float(np.mean(areas)))
            pnum0 = picnum(pics[0])
            pic_prefix.append(float(pnum0[0]))
            pic_code.append(float(pnum0[1]))
            all_codes = [picnum(p)[1] for p in pics if picnum(p)[1] >= 0]
            pic_code_min.append(float(min(all_codes)) if all_codes else -1.0)
            pic_code_max.append(float(max(all_codes)) if all_codes else -1.0)
        else:
            pic0_w.append(-1.0)
            pic0_h.append(-1.0)
            pic_square.append(0.0)
            pic_max_area.append(-1.0)
            pic_mean_area.append(-1.0)
            pic_prefix.append(-1.0)
            pic_code.append(-1.0)
            pic_code_min.append(-1.0)
            pic_code_max.append(-1.0)
        u = r.get("thumbnail") or ""
        m = re.search(r"_(\d+)-", u)
        thumb_code.append(float(m.group(1)) if m else -1.0)

    feats["n_pics"] = np.array(n_pics, dtype=np.float32)
    feats["pic0_w"] = np.array(pic0_w, dtype=np.float32)
    feats["pic0_h"] = np.array(pic0_h, dtype=np.float32)
    feats["pic_max_area"] = np.array(pic_max_area, dtype=np.float32)
    feats["pic_mean_area"] = np.array(pic_mean_area, dtype=np.float32)
    feats["pic_square"] = np.array(pic_square, dtype=np.float32)
    feats["pic_prefix"] = np.array(pic_prefix, dtype=np.float32)
    feats["pic_code"] = np.array(pic_code, dtype=np.float32)
    feats["pic_code_min"] = np.array(pic_code_min, dtype=np.float32)
    feats["pic_code_max"] = np.array(pic_code_max, dtype=np.float32)
    feats["thumb_code"] = np.array(thumb_code, dtype=np.float32)

    # 7. Attributes & Variations
    n_attrs = np.array([float(len(lst(r.get("attributes")))) for r in rows], dtype=np.float32)
    feats["n_attributes"] = n_attrs
    feats["has_attributes"] = (n_attrs > 0).astype(np.float32)
    feats["n_variations"] = np.array([float(len(lst(r.get("variations")))) for r in rows], dtype=np.float32)
    feats["var_sold"] = np.array([float(sum(v.get("sold_quantity", 0) or 0 for v in lst(r.get("variations")))) for r in rows], dtype=np.float32)

    for an in ["ITEM_CONDITION", "BRAND", "MODEL", "COLOR", "LINE", "MATERIAL", "LENGTH", "POWER", "VOLTAGE"]:
        feats[f"has_attr_{an.lower()}"] = np.array([
            1.0 if any(a.get("id") == an for a in lst(r.get("attributes")) if isinstance(a, dict)) else 0.0
            for r in rows
        ], dtype=np.float32)

    # 8. Dates and Durations
    ref = pd.Timestamp("2013-01-01", tz="UTC")
    dates_created = pd.to_datetime([r.get("date_created") for r in rows], utc=True, errors="coerce")
    dates_updated = pd.to_datetime([r.get("last_updated") for r in rows], utc=True, errors="coerce")
    st = pd.to_datetime([r.get("start_time") for r in rows], unit="ms", utc=True, errors="coerce")
    sp = pd.to_datetime([r.get("stop_time") for r in rows], unit="ms", utc=True, errors="coerce")

    feats["created_days"] = np.nan_to_num((dates_created - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["updated_days"] = np.nan_to_num((dates_updated - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["start_days"] = np.nan_to_num((st - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["stop_days"] = np.nan_to_num((sp - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["dur_days"] = np.nan_to_num((sp - st).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["upd_minus_created"] = np.nan_to_num((dates_updated - dates_created).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    feats["start_minus_created"] = np.nan_to_num((st - dates_created).total_seconds(), nan=0.0).astype(np.float32)
    feats["created_hour"] = np.nan_to_num(dates_created.hour, nan=0.0).astype(np.float32)
    feats["created_dow"] = np.nan_to_num(dates_created.dayofweek, nan=0.0).astype(np.float32)
    feats["created_sec"] = np.nan_to_num(dates_created.second, nan=0.0).astype(np.float32)

    # 9. Warranty Features
    warranties = [str(r.get("warranty") or "") for r in rows]
    feats["has_warranty"] = np.array([1.0 if bool(r.get("warranty")) else 0.0 for r in rows], dtype=np.float32)
    feats["warranty_null"] = 1.0 - feats["has_warranty"]
    feats["warranty_len"] = np.array([float(len(w)) for w in warranties], dtype=np.float32)
    for kw, pat in {
        "w_sin": r"\bsin\b", "w_no": r"\bno\b", "w_meses": r"mes", "w_anio": r"a[ñn]o",
        "w_fabrica": r"f[aá]brica", "w_oficial": r"oficial", "w_usado": r"usad", "w_nuevo": r"nuev",
        "w_dias": r"d[ií]as", "w_garantia": r"garant", "w_reputacion": r"reputaci|calificaci",
        "w_prueba": r"prueba|probad|funciona"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, w.lower())) else 0.0 for w in warranties], dtype=np.float32)

    # 10. Listing Types
    listing_types = [str(r.get("listing_type_id") or "") for r in rows]
    feats["is_free"] = np.array([1.0 if lt == "free" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_bronze"] = np.array([1.0 if lt == "bronze" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_silver"] = np.array([1.0 if lt == "silver" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold"] = np.array([1.0 if "gold" in lt else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_special"] = np.array([1.0 if lt == "gold_special" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_pro"] = np.array([1.0 if lt == "gold_pro" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_premium"] = np.array([1.0 if lt == "gold_premium" else 0.0 for lt in listing_types], dtype=np.float32)

    # 11. Buying Modes
    buying_modes = [str(r.get("buying_mode") or "") for r in rows]
    feats["is_buy_it_now"] = np.array([1.0 if bm == "buy_it_now" else 0.0 for bm in buying_modes], dtype=np.float32)
    feats["is_auction"] = np.array([1.0 if bm == "auction" else 0.0 for bm in buying_modes], dtype=np.float32)

    # 12. Text Features & Lexical Cues
    titles = [str(r.get("title") or "") for r in rows]
    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_char_count"] = np.array([float(len(t.replace(" ", ""))) for t in titles], dtype=np.float32)
    feats["title_words"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["title_upper_frac"] = np.array([sum(ch.isupper() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_digits"] = np.array([float(sum(ch.isdigit() for ch in t)) for t in titles], dtype=np.float32)
    feats["title_digit_frac"] = np.array([sum(ch.isdigit() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_exclamation_cnt"] = np.array([float(t.count("!")) for t in titles], dtype=np.float32)
    feats["title_punctuation_cnt"] = np.array([float(sum(ch in string.punctuation for ch in t)) for t in titles], dtype=np.float32)

    used_patterns = {
        "t_usado": r"\busad[oa]s?\b",
        "t_segunda_mano": r"\bsegunda mano\b",
        "t_reparar": r"\b(reparar|repuestos?|para repuesto|desarme|despiece|falla|rot[oa]s?|defecto)\b",
        "t_detalles": r"\b(detalles?|detalles? de uso|leves detalles)\b",
        "t_vintage": r"\b(vintage|retro|antigu[oa]s?|antiguedades|colecci[oó]n|lote)\b",
        "t_buen_estado": r"\b(buen estado|muy buen estado|impecable estado)\b",
        "t_libro_usado": r"\b(libro|manual|revista|fasc[ií]culo)\b",
    }
    new_patterns = {
        "t_nuevo": r"\bnuev[oa]s?\b",
        "t_sellado": r"\b(sellad[oa]s?|en blister|cerrad[oa] en caja|caja cerrada)\b",
        "t_cero_km": r"\b(0\s*km|cero\s*km)\b",
        "t_a_estrenar": r"\b(a estrenar|sin uso|impecable sin uso|nuevo sin uso|nuevo sellado)\b",
        "t_dummy": r"\b(dummy|ficticio)\b",
        "t_oficial_fabrica": r"\b(oficial|f[aá]brica|original)\b",
        "t_artesanal": r"\b(artesanal|hech[oa] a mano|souvenir)\b",
    }

    used_cues_total = np.zeros(N, dtype=np.float32)
    for kw, pat in used_patterns.items():
        reg = re.compile(pat, re.IGNORECASE)
        val = np.array([1.0 if bool(reg.search(t)) else 0.0 for t in titles], dtype=np.float32)
        feats[kw] = val
        used_cues_total += val

    new_cues_total = np.zeros(N, dtype=np.float32)
    for kw, pat in new_patterns.items():
        reg = re.compile(pat, re.IGNORECASE)
        val = np.array([1.0 if bool(reg.search(t)) else 0.0 for t in titles], dtype=np.float32)
        feats[kw] = val
        new_cues_total += val

    feats["used_cues_count"] = used_cues_total
    feats["new_cues_count"] = new_cues_total
    feats["net_cue_score"] = new_cues_total - used_cues_total

    categories = [str(r.get("category_id") or "MISSING") for r in rows]
    df_base = pd.DataFrame(feats)
    log(f"Extracted {df_base.shape[1]} intrinsic item features for {N} records in {time.time()-t0:.2f}s (ZERO seller features)")
    return df_base, pd.Series(categories, name="category_id")


def compute_leak_free_price_features(tr_price, tr_cat, eval_price, eval_cat):
    """Compute strictly leak-free category price statistics on tr and map to eval."""
    price_df = pd.DataFrame({"cat": tr_cat.values, "price": tr_price.values})
    grp = price_df.groupby("cat")["price"]
    med_mapping = grp.median().to_dict()
    std_mapping = grp.std().to_dict()
    freq_mapping = grp.count().to_dict()

    global_med = float(tr_price.median())
    global_std = float(tr_price.std()) if float(tr_price.std()) > 0 else 1.0

    eval_p_vals = eval_price.values if hasattr(eval_price, "values") else eval_price
    eval_med = eval_cat.map(med_mapping).fillna(global_med).values.astype(np.float32)
    eval_std = eval_cat.map(std_mapping).fillna(global_std).values.astype(np.float32)
    eval_std = np.where(eval_std <= 0.0, global_std, eval_std)
    eval_freq = np.log1p(eval_cat.map(freq_mapping).fillna(0.0).values.astype(np.float32))

    rel_price_med = ((eval_p_vals - eval_med) / (eval_med + 1.0)).astype(np.float32)
    cat_price_zscore = np.clip((eval_p_vals - eval_med) / (eval_std + 1e-4), -10.0, 10.0).astype(np.float32)

    fold_price_arr = np.column_stack([
        eval_med,
        eval_std,
        rel_price_med,
        cat_price_zscore,
        eval_freq
    ]).astype(np.float32)

    return fold_price_arr


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("=== HIGH-CAPACITY CATBOOST SPECIALIST (DEPTH=8, LR=0.04, L2=5.0, ITERS=700) ===")
    log("=== STRATIFIED GROUP K-FOLD (5 SPLITS) GROUPED STRICTLY BY SELLER_ID ===")
    log("=" * 80)

    # 1. Load Data
    log("Loading raw JSONLines train (70,000) and test (30,000)...")
    train_rows = []
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))
    test_rows = []
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))

    n_train = len(train_rows)
    n_test = len(test_rows)
    assert n_train == 70000, f"Expected 70,000 train items, got {n_train}"
    assert n_test == 30000, f"Expected 30,000 test items, got {n_test}"

    # Target: 1 for used, 0 for new
    y_train = np.array([1 if r["condition"] == "used" else 0 for r in train_rows], dtype=np.int32)
    sellers_train = np.array([str(r.get("seller_id") or "unknown") for r in train_rows])
    train_ids = [str(r.get("id") or "") for r in train_rows]
    test_ids = [str(r.get("id") or "") for r in test_rows]

    log(f"Train data loaded: {n_train} rows | Used condition count: {int(y_train.sum())} ({y_train.mean()*100:.2f}%)")
    log(f"Test data loaded:  {n_test} rows")
    log(f"Unique sellers in train: {len(np.unique(sellers_train))}")

    # 2. Extract Base Intrinsic Features
    log("Extracting intrinsic item features (ZERO seller leakage)...")
    all_rows = train_rows + test_rows
    del train_rows, test_rows
    gc.collect()

    df_base_all, cat_series_all = extract_intrinsic_features(all_rows)
    del all_rows
    gc.collect()

    df_base_tr = df_base_all.iloc[:n_train].reset_index(drop=True)
    df_base_te = df_base_all.iloc[n_train:].reset_index(drop=True)
    cat_series_tr = cat_series_all.iloc[:n_train].reset_index(drop=True)
    cat_series_te = cat_series_all.iloc[n_train:].reset_index(drop=True)
    del df_base_all, cat_series_all
    gc.collect()

    # 3. Load SVD Features
    log(f"Loading cached 200-dim SVD features from {SVD_CACHE}...")
    assert SVD_CACHE.exists(), f"SVD cache not found at {SVD_CACHE}"
    svd_data = np.load(SVD_CACHE)
    X_tr_svd = svd_data["X_tr_svd"].astype(np.float32)
    X_te_svd = svd_data["X_te_svd"].astype(np.float32)
    assert X_tr_svd.shape == (n_train, 200)
    assert X_te_svd.shape == (n_test, 200)

    # Combine base intrinsic features and SVD into dense arrays
    log("Stacking intrinsic tabular features and 200 SVD text features...")
    X_tr_base = np.column_stack([df_base_tr.values.astype(np.float32), X_tr_svd])
    X_te_base = np.column_stack([df_base_te.values.astype(np.float32), X_te_svd])
    del X_tr_svd, X_te_svd, svd_data
    gc.collect()

    price_tr = df_base_tr["price"]
    price_te = df_base_te["price"]

    # 4. StratifiedGroupKFold grouped strictly by seller_id
    log("Setting up StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped by seller_id...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(X_tr_base, y_train, groups=sellers_train))

    oof_p = np.zeros(n_train, dtype=np.float32)
    test_p_folds = np.zeros((5, n_test), dtype=np.float32)
    fold_metrics = []

    # Verify zero seller leakage across all folds
    for f_idx, (trn_idx, val_idx) in enumerate(splits):
        trn_sellers = set(sellers_train[trn_idx])
        val_sellers = set(sellers_train[val_idx])
        overlap = len(trn_sellers.intersection(val_sellers))
        assert overlap == 0, f"FATAL: Fold {f_idx+1} has seller overlap of {overlap}!"
    log("Verified: Exactly ZERO seller overlap across all 5 folds.")

    # 5. Sequential Fold Training
    for fold, (trn_idx, val_idx) in enumerate(splits):
        t_fold0 = time.time()
        log("\n" + "=" * 60)
        log(f"--- FOLD {fold + 1}/5 (Train: {len(trn_idx)}, Val: {len(val_idx)}) ---")
        log("=" * 60)

        # Leak-free category price statistics
        tr_p = price_tr.iloc[trn_idx]
        tr_c = cat_series_tr.iloc[trn_idx]
        va_p = price_tr.iloc[val_idx]
        va_c = cat_series_tr.iloc[val_idx]

        fold_trn_price = compute_leak_free_price_features(tr_p, tr_c, tr_p, tr_c)
        fold_val_price = compute_leak_free_price_features(tr_p, tr_c, va_p, va_c)
        fold_test_price = compute_leak_free_price_features(tr_p, tr_c, price_te, cat_series_te)

        X_trn = np.column_stack([X_tr_base[trn_idx], fold_trn_price])
        X_val = np.column_stack([X_tr_base[val_idx], fold_val_price])
        X_te = np.column_stack([X_te_base, fold_test_price])

        if fold == 0:
            log(f"Total intrinsic features: {X_trn.shape[1]} (Base={df_base_tr.shape[1]}, SVD=200, PriceStats=5)")

        clf = CatBoostClassifier(
            iterations=700,
            depth=8,
            learning_rate=0.04,
            l2_leaf_reg=5.0,
            thread_count=14,
            random_seed=42 + fold,
            loss_function="Logloss",
            eval_metric="Logloss",
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=100,
            allow_writing_files=False,
        )

        clf.fit(
            X_trn,
            y_train[trn_idx],
            eval_set=(X_val, y_train[val_idx]),
        )

        p_val = clf.predict_proba(X_val)[:, 1].astype(np.float32)
        oof_p[val_idx] = p_val
        p_te = clf.predict_proba(X_te)[:, 1].astype(np.float32)
        test_p_folds[fold] = p_te

        acc = float(accuracy_score(y_train[val_idx], (p_val >= 0.5).astype(int)))
        auc = float(roc_auc_score(y_train[val_idx], p_val))
        ll = float(log_loss(y_train[val_idx], np.clip(p_val, 1e-7, 1.0 - 1e-7)))
        best_iter = int(clf.get_best_iteration())
        elapsed = time.time() - t_fold0

        log(f"Fold {fold + 1} Finished ({elapsed:.1f}s) | Best Iter: {best_iter} | Acc@0.5: {acc:.5f} | ROC-AUC: {auc:.5f} | LogLoss: {ll:.5f}")
        fold_metrics.append({
            "fold": fold + 1,
            "accuracy": acc,
            "roc_auc": auc,
            "logloss": ll,
            "best_iteration": best_iter,
            "elapsed_seconds": elapsed
        })

        del X_trn, X_val, X_te, fold_trn_price, fold_val_price, fold_test_price, clf
        gc.collect()

    # 6. Global Out-of-Fold Evaluation
    global_acc = float(accuracy_score(y_train, (oof_p >= 0.5).astype(int)))
    global_auc = float(roc_auc_score(y_train, oof_p))
    global_ll = float(log_loss(y_train, np.clip(oof_p, 1e-7, 1.0 - 1e-7)))
    n_correct = int(((oof_p >= 0.5).astype(int) == y_train).sum())
    test_p_mean = test_p_folds.mean(axis=0).astype(np.float32)
    total_time = time.time() - t_start

    log("\n" + "=" * 80)
    log("=== GLOBAL CATBOOST DEPTH 8 (700 ITERS) RESULTS ===")
    log(f"  OOF Accuracy @ 0.50: {global_acc:.5f} ({n_correct} / {n_train})")
    log(f"  OOF ROC-AUC:         {global_auc:.5f}")
    log(f"  OOF LogLoss:         {global_ll:.5f}")
    log(f"  Total Duration:      {total_time:.1f}s ({total_time / 60.0:.2f} min)")
    log("=" * 80)

    # 7. Save Clean Artifacts
    oof_csv_path = OUT_DIR / "oof_catboost.csv"
    oof_df = pd.DataFrame({
        "id": np.arange(1, n_train + 1),
        "p_used": oof_p,
        "p_used_catboost": oof_p,
        "seller_id": sellers_train,
        "target_condition_used": y_train,
        "item_id": train_ids,
    })
    oof_df.to_csv(oof_csv_path, index=False)
    log(f"Saved OOF predictions to {oof_csv_path} (shape: {oof_df.shape})")

    test_csv_path = OUT_DIR / "test_probabilities_catboost.csv"
    test_df = pd.DataFrame({
        "id": np.arange(1, n_test + 1),
        "p_used": test_p_mean,
        "p_used_catboost": test_p_mean,
        "item_id": test_ids,
    })
    test_df.to_csv(test_csv_path, index=False)
    log(f"Saved test probabilities to {test_csv_path} (shape: {test_df.shape})")

    summary = {
        "model_name": "Specialized High-Capacity CatBoost Classifier (depth=8, l2=5.0, lr=0.04, iters=700)",
        "hyperparameters": {
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "learning_rate": 0.04,
            "iterations": 700,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "early_stopping_rounds": 100,
        },
        "cv_protocol": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped by seller_id",
        "zero_seller_leakage": True,
        "num_features": 348,
        "n_train": n_train,
        "n_test": n_test,
        "global_accuracy": global_acc,
        "global_roc_auc": global_auc,
        "global_logloss": global_ll,
        "hits": n_correct,
        "total_duration_seconds": total_time,
        "fold_metrics": fold_metrics,
        "output_files": {
            "oof_csv": str(oof_csv_path),
            "test_csv": str(test_csv_path),
        }
    }

    summary_path = OUT_DIR / "summary_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved detailed summary report to {summary_path}")


if __name__ == "__main__":
    main()
