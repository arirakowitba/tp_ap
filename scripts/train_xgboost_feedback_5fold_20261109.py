#!/usr/bin/env python3
"""XGBoost Hist Diversity Specialist with Deconvolved Pattern Feedback Features (5-Fold Stratified).

Author: Antigravity
Date: 2026-11-09

Specifications:
- Incorporates 17 Deconvolved Pattern Feedback Features and Bayesian Log-Odds Composite Scores.
- Full 5-Fold Stratified Cross-Validation on 70,000 train items (Leak-Free).
- Out-of-fold target encodings, text SVD 64 components, nested Naive Bayes text signal.
- Model: XGBClassifier(tree_method="hist", max_depth=7, learning_rate=0.035, colsample_bytree=0.75, subsample=0.85, n_estimators=1400)
- Outputs to results/xgboost_feedback_5fold/:
  - oof_predictions.csv
  - test_probabilities.npy
  - test_probabilities.csv
  - metrics.json
"""

import argparse
import gc
import json
import os
import re
import string
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ZIP = ROOT / "data" / "train_data.zip"
DEFAULT_TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
DEFAULT_OUT_DIR = ROOT / "results" / "xgboost_feedback_5fold"


def log(msg):
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


def extract_base_features(rows):
    """Extract intrinsic item features invariant to cross-validation fold."""
    t0 = time.time()
    feats = {}
    N = len(rows)

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
    feats["log_price"] = np.log1p(np.maximum(0, price))
    feats["log_base_price"] = np.log1p(np.maximum(0, base_price))
    feats["log_iq"] = np.log1p(np.maximum(0, init_q))
    feats["log_aq"] = np.log1p(np.maximum(0, avail_q))
    feats["log_sq"] = np.log1p(np.maximum(0, sold_q))
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

    shipping = [r.get("shipping") or {} for r in rows]
    feats["ship_local_pick_up"] = np.array([1.0 if sh.get("local_pick_up") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_free"] = np.array([1.0 if sh.get("free_shipping") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_dims"] = np.array([1.0 if sh.get("dimensions") is not None else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_n_methods"] = np.array([float(len(lst(sh.get("methods")))) for sh in shipping], dtype=np.float32)
    feats["ship_n_tags"] = np.array([float(len(lst(sh.get("tags")))) for sh in shipping], dtype=np.float32)

    ship_modes = [str(sh.get("mode") or "") for sh in shipping]
    feats["ship_mode_me1"] = np.array([1.0 if sm == "me1" else 0.0 for sm in ship_modes], dtype=np.float32)
    feats["ship_mode_me2"] = np.array([1.0 if sm == "me2" else 0.0 for sm in ship_modes], dtype=np.float32)
    feats["ship_mode_custom"] = np.array([1.0 if sm == "custom" else 0.0 for sm in ship_modes], dtype=np.float32)
    feats["ship_mode_not_specified"] = np.array([1.0 if sm == "not_specified" else 0.0 for sm in ship_modes], dtype=np.float32)

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

    n_attrs = np.array([float(len(lst(r.get("attributes")))) for r in rows], dtype=np.float32)
    feats["n_attributes"] = n_attrs
    feats["has_attributes"] = (n_attrs > 0).astype(np.float32)

    attr_names = ["ITEM_CONDITION", "BRAND", "MODEL", "COLOR", "LINE", "MATERIAL", "LENGTH", "POWER", "VOLTAGE"]
    for an in attr_names:
        feats[f"has_attr_{an.lower()}"] = np.array([
            1.0 if any(a.get("id") == an for a in lst(r.get("attributes")) if isinstance(a, dict)) else 0.0
            for r in rows
        ], dtype=np.float32)

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

    warranties = [str(r.get("warranty") or "") for r in rows]
    feats["warranty_null"] = np.array([1.0 if not r.get("warranty") else 0.0 for r in rows], dtype=np.float32)
    feats["warranty_len"] = np.array([float(len(w)) for w in warranties], dtype=np.float32)
    for kw, pat in {
        "w_sin": r"\bsin\b", "w_no": r"\bno\b", "w_meses": r"mes", "w_anio": r"a[ñn]o",
        "w_fabrica": r"f[aá]brica", "w_oficial": r"oficial", "w_usado": r"usad", "w_nuevo": r"nuev",
        "w_dias": r"d[ií]as", "w_garantia": r"garant", "w_reputacion": r"reputaci|calificaci",
        "w_prueba": r"prueba|probad|funciona"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, w.lower())) else 0.0 for w in warranties], dtype=np.float32)

    listing_types = [str(r.get("listing_type_id") or "") for r in rows]
    feats["is_free"] = np.array([1.0 if lt == "free" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_bronze"] = np.array([1.0 if lt == "bronze" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_silver"] = np.array([1.0 if lt == "silver" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold"] = np.array([1.0 if "gold" in lt else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_special"] = np.array([1.0 if lt == "gold_special" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_pro"] = np.array([1.0 if lt == "gold_pro" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_premium"] = np.array([1.0 if lt == "gold_premium" else 0.0 for lt in listing_types], dtype=np.float32)

    buying_modes = [str(r.get("buying_mode") or "") for r in rows]
    feats["is_buy_it_now"] = np.array([1.0 if bm == "buy_it_now" else 0.0 for bm in buying_modes], dtype=np.float32)
    feats["is_auction"] = np.array([1.0 if bm == "auction" else 0.0 for bm in buying_modes], dtype=np.float32)

    titles = [str(r.get("title") or "") for r in rows]
    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_char_count"] = np.array([float(len(t.replace(" ", ""))) for t in titles], dtype=np.float32)
    feats["title_words"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["title_upper_frac"] = np.array([sum(ch.isupper() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_digits"] = np.array([float(sum(ch.isdigit() for ch in t)) for t in titles], dtype=np.float32)
    feats["title_digit_frac"] = np.array([sum(ch.isdigit() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_exclamation_cnt"] = np.array([float(t.count("!")) for t in titles], dtype=np.float32)
    feats["title_punctuation_cnt"] = np.array([float(sum(ch in string.punctuation for ch in t)) for t in titles], dtype=np.float32)

    for kw, pat in {
        "t_usado": r"usad", "t_nuevo": r"nuev", "t_original": r"original",
        "t_antiguo": r"antig|vintage|retro|colecci", "t_libro": r"libro", "t_lote": r"lote",
        "t_impecable": r"impecable|excelente estado|muy buen estado",
        "t_reparar": r"reparar|repuesto|detalle|defecto|falla|rot[oa]",
        "t_retro": r"\bretro\b", "t_vintage": r"\bvintage\b",
        "t_sellado": r"sellad[oa]|cerrad[oa]|en blister|en caja|0\s*km|cero\s*km",
        "t_estrenar": r"estrenar|sin uso|impecable"
    }.items():
        feats[kw] = np.array([1.0 if bool(re.search(pat, t.lower())) else 0.0 for t in titles], dtype=np.float32)

    # 13. DECONVOLVED PATTERN FEEDBACK FEATURES
    feedback_cues = {
        "cue_impecable_nuevo": (r"\b(impecable sin uso|nuevo sellado|nuevo sin uso)\b", -3.5),
        "cue_a_estrenar": (r"\b(a estrenar)\b", -3.0),
        "cue_artesanal": (r"\b(artesanal|hech[oa] a mano|souvenirs?|cotill[oó]n|centro de mesa)\b", -2.0),
        "cue_sellado": (r"\b(sellad[oa]s?|en blister|cerrad[oa] en caja|caja cerrada)\b", -1.9),
        "cue_0km": (r"\b(0km|0 km)\b", -1.5),
        "cue_disfraz": (r"\b(disfraz|disfraces)\b", -1.6),
        "cue_amiibo": (r"\b(amiibo)\b", -2.0),
        "cue_dummy": (r"\b(dummy|ficticio)\b", -3.0),
        "cue_detalles_uso": (r"\b(detalles? de uso|leves detalles|buen estado|muy buen estado|impecable estado)\b", 3.4),
        "cue_segunda_mano": (r"\b(segunda mano|usad[oa]s?)\b", 2.5),
        "cue_lote_revistas": (r"\b(lote de|colecci[oó]n de|revistas?|fasc[ií]culos?)\b", 2.2),
        "cue_antiguo_vintage": (r"\b(antigu[oa]s?|antiguedades|vintage|retro)\b", 1.7),
        "cue_desarme": (r"\b(desarme|despiece|repuestos?|para repuesto)\b", 0.5),
    }

    full_texts = [f"{t.lower()} {w.lower()}" for t, w in zip(titles, warranties)]
    composite_score = np.zeros(N, dtype=np.float32)

    for cname, (cpat, cweight) in feedback_cues.items():
        creg = re.compile(cpat, re.IGNORECASE)
        cmask = np.array([1.0 if creg.search(ft) else 0.0 for ft in full_texts], dtype=np.float32)
        feats[cname] = cmask
        composite_score += cmask * cweight

    feats["cue_composite_score"] = composite_score
    feats["cue_antiguo_x_nowarranty"] = feats["cue_antiguo_vintage"] * feats["warranty_null"]
    feats["cue_sellado_x_logprice"] = feats["cue_sellado"] * feats["log_price"]
    feats["cue_a_estrenar_x_logprice"] = feats["cue_a_estrenar"] * feats["log_price"]

    cat_df = pd.DataFrame({
        "category_id": [str(r.get("category_id") or "MISSING") for r in rows],
        "listing_type_id": [str(r.get("listing_type_id") or "MISSING") for r in rows],
        "buying_mode": [str(r.get("buying_mode") or "MISSING") for r in rows],
        "status": [str(r.get("status") or "MISSING") for r in rows],
        "seller_id": [str(r.get("seller_id") or "MISSING") for r in rows],
        "ship_mode": [str(sh.get("mode") or "MISSING") for sh in shipping],
        "state": [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "MISSING") for r in rows],
        "city": [str(((r.get("seller_address") or {}).get("city") or {}).get("name") or "MISSING").lower().strip() for r in rows],
    })

    df_base = pd.DataFrame(feats)
    log(f"Extracted {df_base.shape[1]} features (including {len(feedback_cues)+4} feedback features) in {time.time()-t0:.2f}s")
    return df_base, cat_df


def compute_fold_features(df_train_base, cat_train, y_train, train_idx, val_idx, df_test_base=None, cat_test=None):
    """Compute strictly leak-free target encodings and category aggregations."""
    tr_base = df_train_base.iloc[train_idx].copy()
    val_base = df_train_base.iloc[val_idx].copy()
    te_base = df_test_base.copy() if df_test_base is not None else None

    tr_cat = cat_train.iloc[train_idx].copy()
    val_cat = cat_train.iloc[val_idx].copy()
    te_cat = cat_test.copy() if cat_test is not None else None

    y_tr = y_train[train_idx]
    global_prior = float(y_tr.mean())

    tr_cat["y"] = y_tr
    for col, m_weight in [("category_id", 20.0), ("seller_id", 10.0)]:
        agg = tr_cat.groupby(col)["y"].agg(["count", "mean"])
        smoothed = (agg["count"] * agg["mean"] + m_weight * global_prior) / (agg["count"] + m_weight)
        mapping = smoothed.to_dict()
        cnt_mapping = agg["count"].to_dict()

        tr_base[f"{col}_te"] = tr_cat[col].map(mapping).fillna(global_prior).astype(np.float32)
        val_base[f"{col}_te"] = val_cat[col].map(mapping).fillna(global_prior).astype(np.float32)
        if te_base is not None:
            te_base[f"{col}_te"] = te_cat[col].map(mapping).fillna(global_prior).astype(np.float32)

        tr_base[f"{col}_cnt"] = np.log1p(tr_cat[col].map(cnt_mapping).fillna(0.0)).astype(np.float32)
        val_base[f"{col}_cnt"] = np.log1p(val_cat[col].map(cnt_mapping).fillna(0.0)).astype(np.float32)
        if te_base is not None:
            te_base[f"{col}_cnt"] = np.log1p(te_cat[col].map(cnt_mapping).fillna(0.0)).astype(np.float32)

    tr_cat["price"] = tr_base["price"].values
    cat_price_med = tr_cat.groupby("category_id")["price"].median().to_dict()
    cat_price_std = tr_cat.groupby("category_id")["price"].std().to_dict()
    global_price_med = float(tr_base["price"].median())

    for df, c_df in [(tr_base, tr_cat), (val_base, val_cat)] + ([(te_base, te_cat)] if te_base is not None else []):
        meds = c_df["category_id"].map(cat_price_med).fillna(global_price_med)
        stds = c_df["category_id"].map(cat_price_std).fillna(1.0).replace(0.0, 1.0)
        df["rel_price_cat"] = (df["price"] / (meds + 1.0)).astype(np.float32)
        df["price_zscore_cat"] = ((df["price"] - meds) / stds).clip(-5.0, 5.0).astype(np.float32)

    return tr_base, val_base, te_base


def main():
    parser = argparse.ArgumentParser(description="XGBoost Feedback 5-Fold Trainer")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20261109)
    parser.add_argument("--n-estimators", type=int, default=1400)
    args = parser.parse_args()

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 80)
    log("=== XGBOOST HIST DIVERSITY WITH DECONVOLVED PATTERN FEEDBACK (5-FOLD) ===")
    log("=" * 80)

    # 1. Load Datasets
    with zipfile.ZipFile(DEFAULT_TRAIN_ZIP) as z:
        train_rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    with open(DEFAULT_TEST_JSON, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    y_train = np.array([int(r["condition"] == "used") for r in train_rows], dtype=np.int32)
    train_ids = [str(r["id"]) for r in train_rows]
    test_ids = [str(r["id"]) for r in test_rows]
    log(f"Loaded Train: {len(train_rows)} rows (Used rate: {y_train.mean():.4f}), Test: {len(test_rows)} rows.")

    # 2. Extract Base Features + Feedback Features
    df_tr_base, cat_tr = extract_base_features(train_rows)
    df_te_base, cat_te = extract_base_features(test_rows)

    # 3. Load Cached SVD 64 components
    svd_cache_path = ROOT / "results" / "xgboost_diversity_5fold" / "svd_64_features.npz"
    assert svd_cache_path.exists(), f"Missing SVD cache: {svd_cache_path}"
    svd_data = np.load(svd_cache_path)
    if "X_tr_svd" in svd_data:
        svd_tr = svd_data["X_tr_svd"].astype(np.float32)
        svd_te = svd_data["X_te_svd"].astype(np.float32)
    else:
        svd_tr = svd_data["svd_train"].astype(np.float32)
        svd_te = svd_data["svd_test"].astype(np.float32)
    log(f"Loaded cached SVD 64 components: Train {svd_tr.shape}, Test {svd_te.shape}")

    for k in range(64):
        df_tr_base[f"svd_{k}"] = svd_tr[:, k]
        df_te_base[f"svd_{k}"] = svd_te[:, k]

    # 4. Load Cached Nested Naive Bayes Text Log-Odds
    nb_cache_path = ROOT / "results" / "xgboost_diversity_5fold" / "nested_nb_probabilities.npz"
    assert nb_cache_path.exists(), f"Missing NB cache: {nb_cache_path}"
    nb_data = np.load(nb_cache_path)
    if "nb_prob_tr" in nb_data:
        df_tr_base["nb_prob_mean"] = nb_data["nb_prob_tr"].astype(np.float32)
        df_te_base["nb_prob_mean"] = nb_data["nb_prob_te"].astype(np.float32)
    else:
        df_tr_base["nb_prob_mean"] = nb_data["nb_oof"].astype(np.float32)
        df_te_base["nb_prob_mean"] = nb_data["nb_test"].astype(np.float32)
    eps = 1e-5
    df_tr_base["nb_log_odds"] = np.log(np.clip(df_tr_base["nb_prob_mean"], eps, 1 - eps) / (1.0 - np.clip(df_tr_base["nb_prob_mean"], eps, 1 - eps))).astype(np.float32)
    df_te_base["nb_log_odds"] = np.log(np.clip(df_te_base["nb_prob_mean"], eps, 1 - eps) / (1.0 - np.clip(df_te_base["nb_prob_mean"], eps, 1 - eps))).astype(np.float32)
    log("Loaded cached Naive Bayes text signals.")

    # Frequency encode categoricals for XGBoost
    for c in ["listing_type_id", "buying_mode", "status", "ship_mode", "state"]:
        freq = cat_tr[c].value_counts().to_dict()
        df_tr_base[f"{c}_freq"] = cat_tr[c].map(freq).fillna(0).astype(np.float32)
        df_te_base[f"{c}_freq"] = cat_te[c].map(freq).fillna(0).astype(np.float32)

    # 5. 5-Fold Stratified CV
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    oof_preds = np.zeros(len(y_train), dtype=np.float32)
    test_preds = np.zeros(len(test_rows), dtype=np.float32)

    xgb_params = {
        "tree_method": "hist",
        "max_depth": 7,
        "learning_rate": 0.035,
        "colsample_bytree": 0.75,
        "subsample": 0.85,
        "min_child_weight": 20,
        "eval_metric": "logloss",
        "n_jobs": 6,
    }

    fold_metrics = []
    log("\n--- STARTING 5-FOLD CROSS-VALIDATION TRAINING ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_tr_base, y_train), 1):
        t_fold = time.time()
        log(f"\n--- FOLD {fold}/{args.n_splits} ---")

        tr_df, val_df, te_df = compute_fold_features(
            df_tr_base, cat_tr, y_train, train_idx, val_idx, df_te_base, cat_te
        )

        model = xgb.XGBClassifier(
            **xgb_params,
            n_estimators=args.n_estimators,
            early_stopping_rounds=40,
            random_state=args.random_state + fold,
        )

        model.fit(
            tr_df,
            y_train[train_idx],
            eval_set=[(tr_df, y_train[train_idx]), (val_df, y_train[val_idx])],
            verbose=False,
        )

        p_val = model.predict_proba(val_df)[:, 1]
        oof_preds[val_idx] = p_val

        p_test_fold = model.predict_proba(te_df)[:, 1]
        test_preds += p_test_fold / args.n_splits

        best_iter = model.best_iteration
        acc = accuracy_score(y_train[val_idx], p_val >= 0.5)
        ll = log_loss(y_train[val_idx], np.clip(p_val, 1e-7, 1.0 - 1e-7))
        auc = roc_auc_score(y_train[val_idx], p_val)
        log(f"Fold {fold} Finished ({time.time()-t_fold:.1f}s) | Trees: {best_iter} | Acc: {acc:.5f} | LogLoss: {ll:.5f} | AUC: {auc:.5f}")

        fold_metrics.append({
            "fold": fold,
            "accuracy": float(acc),
            "logloss": float(ll),
            "auc": float(auc),
            "best_iteration": int(best_iter),
        })

    # Overall OOF Evaluation
    oof_acc_05 = accuracy_score(y_train, oof_preds >= 0.5)
    oof_hits = int(((oof_preds >= 0.5) == y_train).sum())
    oof_ll = log_loss(y_train, np.clip(oof_preds, 1e-7, 1.0 - 1e-7))
    oof_auc = roc_auc_score(y_train, oof_preds)

    log("\n" + "=" * 80)
    log("=== OVERALL 5-FOLD OUT-OF-FOLD XGBOOST FEEDBACK BENCHMARK RESULTS ===")
    log("=" * 80)
    log(f"OOF Accuracy (t=0.50): {oof_acc_05:.5f} ({oof_hits} / 70,000 hits)")
    log(f"OOF LogLoss:          {oof_ll:.5f}")
    log(f"OOF ROC-AUC:          {oof_auc:.5f}")

    # Save OOF Predictions
    df_oof = pd.DataFrame({
        "id": train_ids,
        "item_id": train_ids,
        "seller_id": cat_tr["seller_id"],
        "y_true": y_train,
        "prob_used": oof_preds,
        "pred_condition": np.where(oof_preds >= 0.5, "used", "new"),
    })
    df_oof.to_csv(DEFAULT_OUT_DIR / "oof_predictions.csv", index=False)
    log(f"Saved OOF predictions to {DEFAULT_OUT_DIR / 'oof_predictions.csv'}")

    # Save Test Predictions
    np.save(DEFAULT_OUT_DIR / "test_probabilities.npy", test_preds)
    df_test = pd.DataFrame({
        "ID": np.arange(1, len(test_preds) + 1),
        "id": test_ids,
        "item_id": test_ids,
        "prob_used": test_preds,
        "condition": np.where(test_preds >= 0.5, "used", "new"),
    })
    df_test.to_csv(DEFAULT_OUT_DIR / "test_probabilities.csv", index=False)
    log(f"Saved test probabilities to {DEFAULT_OUT_DIR / 'test_probabilities.npy'}")

    # Save Summary JSON
    metrics_summary = {
        "model": "XGBoost_Feedback_5Fold",
        "num_features": int(tr_df.shape[1]),
        "oof_accuracy_05": float(oof_acc_05),
        "oof_hits": int(oof_hits),
        "oof_logloss": float(oof_ll),
        "oof_auc": float(oof_auc),
        "fold_metrics": fold_metrics,
    }
    with open(DEFAULT_OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    log(f"Saved metrics summary to {DEFAULT_OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
