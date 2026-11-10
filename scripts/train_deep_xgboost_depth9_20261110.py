#!/usr/bin/env python3
"""Deep XGBoost Depth 9 Specialist (5-Fold Stratified).

Author: Antigravity
Date: 2026-11-10

Specifications:
1. Load data from data/ap_2026_q_2/raw/train_data.jsonlines (70,000 items) and test_data.jsonlines (30,000 items).
2. Features to engineer (STRICTLY LEAK-FREE, ZERO seller_id features):
   - 200 TruncatedSVD components over 150,000 TF-IDF word and character n-grams.
   - Price features: log(price+1), category price median, category price std,
     relative price to median (price - med) / (med + 1), category price z-score.
   - Lexical indicators: used cues and new cues.
   - Listing metadata: has_warranty, free_shipping, listing_type_id flags,
     buying mode flags, quantity dynamics, title statistics.
3. Model Configuration:
   - XGBClassifier(n_estimators=2500, learning_rate=0.02, max_depth=9,
                   subsample=0.70, colsample_bytree=0.60, tree_method="hist",
                   n_jobs=16, random_state=8888, eval_metric="logloss")
4. Validation:
   - 5-Fold Stratified K-Fold cross-validation across 70,000 train items.
   - Compute OOF accuracy, ROC AUC, and Log Loss.
   - Generate test set continuous probabilities (mean over 5 folds).
5. Output Artifacts:
   - Save to results/deep_xgboost_depth9_20261110/xgboost_depth9_predictions.npz (keys: oof_xgb, test_xgb).
   - Save summary to results/deep_xgboost_depth9_20261110/summary.json.
   - Also save CSV predictions for downstream ensemble integration.
"""

import argparse
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
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
DEFAULT_TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
DEFAULT_OUT_DIR = ROOT / "results" / "deep_xgboost_depth9_20261110"


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


def extract_intrinsic_features(rows):
    """Extract item intrinsic features invariant to folds (STRICTLY ZERO SELLER_ID)."""
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

    # 5. Shipping Metadata (has_warranty, free_shipping, etc.)
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

    # Common attribute flags
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

    # 9. Warranty Features (has_warranty, length, warranty cues)
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

    # 10. Listing Type Indicators
    listing_types = [str(r.get("listing_type_id") or "") for r in rows]
    feats["is_free"] = np.array([1.0 if lt == "free" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_bronze"] = np.array([1.0 if lt == "bronze" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_silver"] = np.array([1.0 if lt == "silver" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold"] = np.array([1.0 if "gold" in lt else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_special"] = np.array([1.0 if lt == "gold_special" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_pro"] = np.array([1.0 if lt == "gold_pro" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_premium"] = np.array([1.0 if lt == "gold_premium" else 0.0 for lt in listing_types], dtype=np.float32)

    # 11. Buying Mode Indicators
    buying_modes = [str(r.get("buying_mode") or "") for r in rows]
    feats["is_buy_it_now"] = np.array([1.0 if bm == "buy_it_now" else 0.0 for bm in buying_modes], dtype=np.float32)
    feats["is_auction"] = np.array([1.0 if bm == "auction" else 0.0 for bm in buying_modes], dtype=np.float32)

    # 12. Text Features (Title metadata & domain cues)
    titles = [str(r.get("title") or "") for r in rows]
    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_char_count"] = np.array([float(len(t.replace(" ", ""))) for t in titles], dtype=np.float32)
    feats["title_words"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["title_upper_frac"] = np.array([sum(ch.isupper() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_digits"] = np.array([float(sum(ch.isdigit() for ch in t)) for t in titles], dtype=np.float32)
    feats["title_digit_frac"] = np.array([sum(ch.isdigit() for ch in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["title_exclamation_cnt"] = np.array([float(t.count("!")) for t in titles], dtype=np.float32)
    feats["title_punctuation_cnt"] = np.array([float(sum(ch in string.punctuation for ch in t)) for t in titles], dtype=np.float32)

    # 13. Lexical Indicators: Used Cues and New Cues
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

    # Category series
    categories = [str(r.get("category_id") or "MISSING") for r in rows]

    # Full text corpus for 150k TF-IDF
    text_corpus = []
    for r in rows:
        t = str(r.get("title") or "")
        w = str(r.get("warranty") or "")
        attr_vals = " ".join(str(a.get("value_name") or a.get("name") or "") for a in lst(r.get("attributes")) if isinstance(a, dict))
        tags_val = " ".join(str(tg) for tg in lst(r.get("tags")) if tg)
        text_corpus.append(f"{t} WARRANTY {w} ATTR {attr_vals} TAGS {tags_val}".strip())

    df_base = pd.DataFrame(feats)
    log(f"Extracted {df_base.shape[1]} intrinsic item features for {N} records in {time.time()-t0:.2f}s (ZERO seller features)")
    return df_base, pd.Series(categories, name="category_id"), np.asarray(text_corpus, dtype=object)


def compute_leak_free_price_features(df_base, cat_series, train_idx, eval_idx):
    """Compute strictly leak-free category price statistics on train_idx and map to eval_idx."""
    tr_cat = cat_series.iloc[train_idx]
    eval_cat = cat_series.iloc[eval_idx]

    tr_price = df_base["price"].iloc[train_idx]
    eval_price = df_base["price"].iloc[eval_idx].values

    # Category price median and std strictly on train fold
    price_df = pd.DataFrame({"cat": tr_cat.values, "price": tr_price.values})
    grp = price_df.groupby("cat")["price"]
    med_mapping = grp.median().to_dict()
    std_mapping = grp.std().to_dict()
    freq_mapping = grp.count().to_dict()

    global_med = float(tr_price.median())
    global_std = float(tr_price.std()) if float(tr_price.std()) > 0 else 1.0

    eval_med = eval_cat.map(med_mapping).fillna(global_med).values.astype(np.float32)
    eval_std = eval_cat.map(std_mapping).fillna(global_std).values.astype(np.float32)
    eval_std = np.where(eval_std <= 0.0, global_std, eval_std)
    eval_freq = np.log1p(eval_cat.map(freq_mapping).fillna(0.0).values.astype(np.float32))

    # Requested price features:
    # log(price+1), category price median, category price std,
    # relative price to median (price - med) / (med + 1), category price z-score
    rel_price_med = ((eval_price - eval_med) / (eval_med + 1.0)).astype(np.float32)
    cat_price_zscore = np.clip((eval_price - eval_med) / (eval_std + 1e-4), -10.0, 10.0).astype(np.float32)

    fold_price_df = pd.DataFrame({
        "cat_price_median": eval_med,
        "cat_price_std": eval_std,
        "rel_price_to_median": rel_price_med,
        "category_price_zscore": cat_price_zscore,
        "category_freq": eval_freq
    }, index=eval_cat.index)

    return fold_price_df


def main():
    parser = argparse.ArgumentParser(description="Deep XGBoost Depth 9 Specialist (5-Fold Stratified)")
    parser.add_argument("--train_json", default=str(DEFAULT_TRAIN_JSON), help="Path to train_data.jsonlines")
    parser.add_argument("--test_json", default=str(DEFAULT_TEST_JSON), help="Path to test_data.jsonlines")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--seed", type=int, default=8888, help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=2500, help="Number of trees")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--max_depth", type=int, default=9, help="Max tree depth")
    parser.add_argument("--subsample", type=float, default=0.70, help="Subsample ratio")
    parser.add_argument("--colsample_bytree", type=float, default=0.60, help="Colsample ratio")
    parser.add_argument("--early_stopping", type=int, default=150, help="Early stopping rounds")
    parser.add_argument("--n_jobs", type=int, default=16, help="Worker threads")
    parser.add_argument("--svd_dim", type=int, default=200, help="SVD components")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log("=" * 80)
    log("=== DEEP XGBOOST DEPTH 9 SPECIALIST (5-FOLD STRATIFIED) ===")
    log("=" * 80)
    log(f"Config: n_estimators={args.n_estimators}, lr={args.lr}, max_depth={args.max_depth}, "
        f"subsample={args.subsample}, colsample={args.colsample_bytree}, n_jobs={args.n_jobs}, seed={args.seed}")
    log(f"Constraint: STRICTLY ZERO seller_id features (Leak-free item specialist).")

    # 1. Load Data
    log(f"Step 1: Loading raw datasets from {args.train_json} and {args.test_json}...")
    train_rows = []
    with open(args.train_json, "r", encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))
    test_rows = []
    with open(args.test_json, "r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))

    n_train = len(train_rows)
    n_test = len(test_rows)
    assert n_train == 70000, f"Expected 70,000 train rows, found {n_train}"
    assert n_test == 30000, f"Expected 30,000 test rows, found {n_test}"

    y_train = np.array([int(r["condition"] == "used") for r in train_rows], dtype=np.int32)
    train_ids = [str(r["id"]) for r in train_rows]
    test_ids = [str(r["id"]) for r in test_rows]
    global_prior = float(y_train.mean())
    log(f"Loaded Train: {n_train} rows (Used rate: {global_prior:.4f}), Test: {n_test} rows.")

    # 2. Extract Intrinsic Tabular & Lexical Features
    log("Step 2: Extracting intrinsic tabular, lexical, and metadata features...")
    all_rows = train_rows + test_rows
    del train_rows, test_rows
    gc.collect()

    df_base_all, cat_series_all, text_corpus_all = extract_intrinsic_features(all_rows)
    del all_rows
    gc.collect()

    df_base_tr = df_base_all.iloc[:n_train].reset_index(drop=True)
    df_base_te = df_base_all.iloc[n_train:].reset_index(drop=True)
    cat_series_tr = cat_series_all.iloc[:n_train].reset_index(drop=True)
    cat_series_te = cat_series_all.iloc[n_train:].reset_index(drop=True)
    text_corpus_tr = text_corpus_all[:n_train]
    text_corpus_te = text_corpus_all[n_train:]
    del df_base_all, cat_series_all
    gc.collect()

    # 3. 200 TruncatedSVD Components over 150,000 TF-IDF Word & Char N-Grams
    svd_cache_path = out_dir / f"svd_{args.svd_dim}_features.npz"
    svd_cols = [f"svd_{i:03d}" for i in range(args.svd_dim)]

    if svd_cache_path.exists():
        log(f"Step 3: Loading cached {args.svd_dim}-dim SVD components from {svd_cache_path}...")
        svd_data = np.load(svd_cache_path)
        X_tr_svd = svd_data["X_tr_svd"].astype(np.float32)
        X_te_svd = svd_data["X_te_svd"].astype(np.float32)
    else:
        log(f"Step 3: Extracting 150,000 TF-IDF N-Grams and computing TruncatedSVD({args.svd_dim})...")
        t_svd0 = time.time()
        # Word TF-IDF (60,000 features, 1-2 ngrams)
        vec_word = TfidfVectorizer(ngram_range=(1, 2), max_features=60000, min_df=3, sublinear_tf=True, dtype=np.float32)
        log("  Fitting Word TF-IDF (60,000 features)...")
        X_tr_w = vec_word.fit_transform(text_corpus_tr)
        X_te_w = vec_word.transform(text_corpus_te)

        # Char TF-IDF (90,000 features, 3-5 char_wb ngrams)
        vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=90000, min_df=5, sublinear_tf=True, dtype=np.float32)
        log("  Fitting Char TF-IDF (90,000 features)...")
        X_tr_c = vec_char.fit_transform(text_corpus_tr)
        X_te_c = vec_char.transform(text_corpus_te)

        X_tr_tfidf = sparse.hstack([X_tr_w, X_tr_c], format="csr")
        X_te_tfidf = sparse.hstack([X_te_w, X_te_c], format="csr")
        log(f"  Combined TF-IDF Shape: Train {X_tr_tfidf.shape}, Test {X_te_tfidf.shape} ({time.time()-t_svd0:.1f}s)")

        del X_tr_w, X_te_w, X_tr_c, X_te_c, text_corpus_tr, text_corpus_te, text_corpus_all
        gc.collect()

        log(f"  Computing TruncatedSVD({args.svd_dim}) on 150,000 features...")
        svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
        X_tr_svd = svd.fit_transform(X_tr_tfidf).astype(np.float32)
        X_te_svd = svd.transform(X_te_tfidf).astype(np.float32)
        log(f"  SVD Explained Variance Ratio Sum: {svd.explained_variance_ratio_.sum():.4f} in {time.time()-t_svd0:.1f}s")

        del X_tr_tfidf, X_te_tfidf
        gc.collect()

        np.savez_compressed(svd_cache_path, X_tr_svd=X_tr_svd, X_te_svd=X_te_svd)
        log(f"  Saved SVD components to {svd_cache_path}")

    df_svd_tr = pd.DataFrame(X_tr_svd, columns=svd_cols, dtype=np.float32)
    df_svd_te = pd.DataFrame(X_te_svd, columns=svd_cols, dtype=np.float32)
    del X_tr_svd, X_te_svd
    gc.collect()

    # 4. 5-Fold Stratified K-Fold Validation Setup
    log("Step 4: Setting up 5-Fold Stratified K-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    oof_predictions = np.zeros(n_train, dtype=np.float32)
    test_fold_probabilities = np.zeros((5, n_test), dtype=np.float32)
    fold_metrics = []

    xgb_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.lr,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "tree_method": "hist",
        "n_jobs": args.n_jobs,
        "eval_metric": "logloss"
    }
    if args.early_stopping and args.early_stopping > 0:
        xgb_params["early_stopping_rounds"] = args.early_stopping

    log(f"XGBoost Configuration: {xgb_params}")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(df_base_tr, y_train)):
        t_f0 = time.time()
        log(f"\n--- FOLD {fold+1}/5 ---")
        log(f"Train samples: {len(trn_idx)}, Val samples: {len(val_idx)}")

        # Leak-free price statistics computed strictly on trn_idx
        fold_trn_price = compute_leak_free_price_features(df_base_tr, cat_series_tr, trn_idx, trn_idx)
        fold_val_price = compute_leak_free_price_features(df_base_tr, cat_series_tr, trn_idx, val_idx)
        fold_test_price = compute_leak_free_price_features(
            pd.concat([df_base_tr.iloc[trn_idx], df_base_te], ignore_index=True),
            pd.concat([cat_series_tr.iloc[trn_idx], cat_series_te], ignore_index=True),
            np.arange(len(trn_idx)),
            np.arange(len(trn_idx), len(trn_idx) + n_test)
        ).reset_index(drop=True)

        # Assemble Full Feature Matrix for Fold
        X_trn = pd.concat([
            df_base_tr.iloc[trn_idx].reset_index(drop=True),
            df_svd_tr.iloc[trn_idx].reset_index(drop=True),
            fold_trn_price.reset_index(drop=True)
        ], axis=1).fillna(0.0).astype(np.float32)

        X_val = pd.concat([
            df_base_tr.iloc[val_idx].reset_index(drop=True),
            df_svd_tr.iloc[val_idx].reset_index(drop=True),
            fold_val_price.reset_index(drop=True)
        ], axis=1).fillna(0.0).astype(np.float32)

        X_test_fold = pd.concat([
            df_base_te.reset_index(drop=True),
            df_svd_te.reset_index(drop=True),
            fold_test_price.reset_index(drop=True)
        ], axis=1).fillna(0.0).astype(np.float32)

        if fold == 0:
            log(f"Engineered total feature columns: {X_trn.shape[1]}")
            feature_names = list(X_trn.columns)

        # Fit XGBoost Model
        clf = xgb.XGBClassifier(**xgb_params, random_state=args.seed + fold)
        clf.fit(
            X_trn, y_train[trn_idx],
            eval_set=[(X_val, y_train[val_idx])],
            verbose=200
        )

        val_probs = clf.predict_proba(X_val)[:, 1].astype(np.float32)
        oof_predictions[val_idx] = val_probs

        acc_f = float(accuracy_score(y_train[val_idx], (val_probs >= 0.5).astype(int)))
        ll_f = float(log_loss(y_train[val_idx], np.clip(val_probs, 1e-7, 1.0 - 1e-7)))
        auc_f = float(roc_auc_score(y_train[val_idx], val_probs))
        best_iter = int(clf.best_iteration) if hasattr(clf, "best_iteration") and clf.best_iteration is not None else int(args.n_estimators)

        log(f"Fold {fold+1} Finished ({time.time()-t_f0:.1f}s) | Best Iter: {best_iter} | Acc: {acc_f:.5f} | LogLoss: {ll_f:.5f} | AUC: {auc_f:.5f}")
        fold_metrics.append({
            "fold": int(fold + 1),
            "accuracy": acc_f,
            "logloss": ll_f,
            "auc": auc_f,
            "best_iteration": best_iter
        })

        # Predict on Test Set
        test_pred_fold = clf.predict_proba(X_test_fold)[:, 1].astype(np.float32)
        test_fold_probabilities[fold] = test_pred_fold

        del X_trn, X_val, X_test_fold, clf, fold_trn_price, fold_val_price, fold_test_price
        gc.collect()

    # 5. Overall Out-of-Fold Evaluation
    log("\n" + "=" * 80)
    log("=== OVERALL 5-FOLD OUT-OF-FOLD EVALUATION ===")
    log("=" * 80)

    oof_preds_clipped = np.clip(oof_predictions, 1e-7, 1.0 - 1e-7)
    overall_acc_05 = float(accuracy_score(y_train, (oof_predictions >= 0.5).astype(int)))
    overall_ll = float(log_loss(y_train, oof_preds_clipped))
    overall_auc = float(roc_auc_score(y_train, oof_predictions))

    # Optimal threshold search
    grid = np.linspace(0.40, 0.60, 201)
    best_thresh = 0.50
    best_acc = overall_acc_05
    for t in grid:
        acc_t = float(accuracy_score(y_train, (oof_predictions >= t).astype(int)))
        if acc_t > best_acc:
            best_acc = acc_t
            best_thresh = float(t)

    log(f"OOF Accuracy @ 0.50: {overall_acc_05:.5f} ({int(((oof_predictions >= 0.5).astype(int) == y_train).sum())}/{n_train})")
    log(f"OOF Optimal Acc:     {best_acc:.5f} at threshold {best_thresh:.4f}")
    log(f"OOF LogLoss:         {overall_ll:.5f}")
    log(f"OOF ROC-AUC:         {overall_auc:.5f}")

    # 6. Save Predictions (.npz and .csv)
    test_prob_mean = np.mean(test_fold_probabilities, axis=0).astype(np.float32)

    # Required NPZ output: keys 'oof_xgb', 'test_xgb'
    npz_path = out_dir / "xgboost_depth9_predictions.npz"
    np.savez_compressed(
        npz_path,
        oof_xgb=oof_predictions,
        test_xgb=test_prob_mean
    )
    log(f"Saved predictions NPZ to {npz_path} (keys: 'oof_xgb', 'test_xgb')")

    # CSV outputs for ensemble compatibility
    oof_df = pd.DataFrame({
        "id": train_ids,
        "item_id": train_ids,
        "y_true": y_train,
        "prob_used": oof_predictions,
        "pred_condition": np.where(oof_predictions >= best_thresh, "used", "new"),
        "pred_condition_05": np.where(oof_predictions >= 0.5, "used", "new")
    })
    oof_csv_path = out_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    log(f"Saved OOF predictions CSV to {oof_csv_path}")

    test_df = pd.DataFrame({
        "ID": np.arange(1, n_test + 1),
        "id": test_ids,
        "item_id": test_ids,
        "prob_used": test_prob_mean,
        "condition": np.where(test_prob_mean >= best_thresh, "used", "new"),
        "predicted_condition": np.where(test_prob_mean >= best_thresh, "used", "new")
    })
    test_csv_path = out_dir / "test_probabilities.csv"
    test_df.to_csv(test_csv_path, index=False)
    log(f"Saved test predictions CSV to {test_csv_path}")

    n_used_test = int((test_df["condition"] == "used").sum())
    n_new_test = int((test_df["condition"] == "new").sum())
    log(f"Test Distribution: Used = {n_used_test} ({n_used_test/n_test:.2%}), New = {n_new_test} ({n_new_test/n_test:.2%})")

    # 7. Save Summary JSON
    total_duration = time.time() - t_start
    summary = {
        "model": "Deep_XGBoost_Depth9_Specialist",
        "num_features": len(feature_names),
        "n_train": n_train,
        "n_test": n_test,
        "global_prior": global_prior,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.lr,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "tree_method": "hist",
            "n_jobs": args.n_jobs,
            "random_state": args.seed,
            "eval_metric": "logloss",
            "early_stopping_rounds": args.early_stopping
        },
        "svd_components": args.svd_dim,
        "cv_folds": fold_metrics,
        "oof_accuracy": overall_acc_05,
        "oof_accuracy_optimal": best_acc,
        "optimal_threshold": best_thresh,
        "oof_logloss": overall_ll,
        "oof_roc_auc": overall_auc,
        "test_used_count": n_used_test,
        "test_new_count": n_new_test,
        "runtime_seconds": round(total_duration, 2),
        "runtime_minutes": round(total_duration / 60.0, 2),
        "output_artifacts": {
            "predictions_npz": str(npz_path),
            "oof_csv": str(oof_csv_path),
            "test_csv": str(test_csv_path),
            "summary_json": str(out_dir / "summary.json")
        }
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary JSON to {summary_path}")
    log(f"Pipeline finished successfully in {total_duration / 60.0:.2f} minutes.")


if __name__ == "__main__":
    main()
