#!/usr/bin/env python3
"""5-Fold Stratified / Seller-Aware XGBoost Model for Structural Tree Diversity.

Full 70,000 train rows with 5-fold Stratified cross-validation.
Architecture: XGBClassifier(n_estimators=1200, max_depth=7, learning_rate=0.03,
                            colsample_bytree=0.8, subsample=0.8, reg_lambda=3.0,
                            tree_method='hist', random_state=20261109, n_jobs=6)
Early stopping rounds: 80.

Feature pipeline:
- Deep tabular features (prices, relative price to category median, log price, etc.).
- Category frequency and seller frequency encodings.
- Text features: Title length, character count, uppercase ratio, digit ratio, punctuation count.
- Sublinear TF-IDF + TruncatedSVD (64 components) from title + warranty + attributes + tags.
- Nested Naive Bayes log-odds text feature generated strictly out-of-fold using GroupKFold by seller.

Outputs:
- results/xgboost_diversity_5fold/oof_predictions.csv
- results/xgboost_diversity_5fold/test_probabilities.npy
- results/xgboost_diversity_5fold/test_probabilities.csv
- results/xgboost_diversity_5fold/metrics.json
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
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ZIP = ROOT / "data" / "train_data.zip"
DEFAULT_TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
DEFAULT_OUT_DIR = ROOT / "results" / "xgboost_diversity_5fold"


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

    # 2. Structural Flags
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

    # 4. Payment methods
    for code in ["MLATB", "MLAWC", "MLAMO", "MLAOT", "MLACD", "MLABC", "MLAVE", "MLAMC", "MLAAM", "MLADC", "MLAVS", "MLAMS", "MLAWT", "MLAMP", "MLAAB"]:
        feats[f"pm_{code}"] = np.array([1.0 if code in [m.get("id") for m in lst(r.get("non_mercado_pago_payment_methods"))] else 0.0 for r in rows], dtype=np.float32)
    feats["n_pm"] = np.array([float(len(lst(r.get("non_mercado_pago_payment_methods")))) for r in rows], dtype=np.float32)

    # 5. Shipping
    shipping = [r.get("shipping") or {} for r in rows]
    feats["ship_local_pick_up"] = np.array([1.0 if sh.get("local_pick_up") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_free"] = np.array([1.0 if sh.get("free_shipping") else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_dims"] = np.array([1.0 if sh.get("dimensions") is not None else 0.0 for sh in shipping], dtype=np.float32)
    feats["ship_n_methods"] = np.array([float(len(lst(sh.get("methods")))) for sh in shipping], dtype=np.float32)
    feats["ship_n_tags"] = np.array([float(len(lst(sh.get("tags")))) for sh in shipping], dtype=np.float32)

    # Shipping modes
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

    # 7. Variations & Attributes
    feats["n_variations"] = np.array([float(len(lst(r.get("variations")))) for r in rows], dtype=np.float32)
    feats["n_attributes"] = np.array([float(len(lst(r.get("attributes")))) for r in rows], dtype=np.float32)
    feats["var_sold"] = np.array([float(sum(v.get("sold_quantity", 0) or 0 for v in lst(r.get("variations")))) for r in rows], dtype=np.float32)

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

    # 9. Warranty cues
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

    # 10. Listing type indicators
    listing_types = [str(r.get("listing_type_id") or "") for r in rows]
    feats["is_free"] = np.array([1.0 if lt == "free" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_bronze"] = np.array([1.0 if lt == "bronze" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_silver"] = np.array([1.0 if lt == "silver" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold"] = np.array([1.0 if "gold" in lt else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_special"] = np.array([1.0 if lt == "gold_special" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_pro"] = np.array([1.0 if lt == "gold_pro" else 0.0 for lt in listing_types], dtype=np.float32)
    feats["is_gold_premium"] = np.array([1.0 if lt == "gold_premium" else 0.0 for lt in listing_types], dtype=np.float32)

    # 11. Buying mode indicators
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

    # 13. Categorical metadata frame
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

    # 14. Full rich text corpus (title + warranty + attributes + tags)
    text_corpus = []
    for r in rows:
        t = str(r.get("title") or "")
        w = str(r.get("warranty") or "")
        attr_vals = " ".join(str(a.get("value_name") or a.get("name") or "") for a in lst(r.get("attributes")) if isinstance(a, dict))
        tags_val = " ".join(str(tg) for tg in lst(r.get("tags")) if tg)
        text_corpus.append(f"{t} WARRANTY {w} ATTR {attr_vals} TAGS {tags_val}".strip())

    df_base = pd.DataFrame(feats)
    log(f"Extracted {df_base.shape[1]} base intrinsic features for {N} records in {time.time()-t0:.2f}s")
    return df_base, cat_df, np.asarray(text_corpus, dtype=object)


def text_fit(X, y, tr, va):
    """Supervised NB log-count ratio strictly within the corresponding fit fold."""
    pos = np.asarray(X[tr][y[tr] == 1].sum(axis=0)).ravel() + 1.0
    neg = np.asarray(X[tr][y[tr] == 0].sum(axis=0)).ravel() + 1.0
    ratio = np.log((pos / pos.sum()) / (neg / neg.sum()))
    m = LogisticRegression(C=2.0, solver="liblinear", max_iter=300, random_state=104)
    m.fit(X[tr].multiply(ratio), y[tr])
    p = m.predict_proba(X[va].multiply(ratio))[:, 1]
    return p


def compute_nested_nb_features(text_all, y_train, sellers_train, n_train, n_test, out_dir=None, seed=20261109):
    """Compute out-of-fold Naive Bayes text features using GroupKFold by seller."""
    t0 = time.time()
    log("Computing nested Naive Bayes text features with GroupKFold by seller...")

    if out_dir:
        nb_cache_path = Path(out_dir) / "nested_nb_probabilities.npz"
        if nb_cache_path.exists():
            log(f"Loading cached nested NB probabilities from {nb_cache_path}...")
            data = np.load(nb_cache_path)
            return data["nb_prob_tr"], data["nb_logit_tr"], data["nb_prob_te"], data["nb_logit_te"]

    # Check if pre-computed cached artifact exists in legacy results
    cached_path = ROOT / "results" / "full_native_nb_70k_20261105" / "nested_text_probabilities.npy"
    if cached_path.exists():
        log(f"Found existing pre-computed nested text probabilities at {cached_path}!")
        ptext = np.load(cached_path)
        if len(ptext) == (n_train + n_test) and not np.isnan(ptext).any():
            log(f"Loaded cached nested NB probabilities successfully (shape {ptext.shape}).")
            p_tr = ptext[:n_train].astype(np.float32)
            p_te = ptext[n_train:].astype(np.float32)
            p_tr_c = np.clip(p_tr, 1e-6, 1.0 - 1e-6)
            p_te_c = np.clip(p_te, 1e-6, 1.0 - 1e-6)
            logit_tr = np.log(p_tr_c / (1.0 - p_tr_c))
            logit_te = np.log(p_te_c / (1.0 - p_te_c))
            if out_dir:
                np.savez_compressed(
                    Path(out_dir) / "nested_nb_probabilities.npz",
                    nb_prob_tr=p_tr, nb_logit_tr=logit_tr,
                    nb_prob_te=p_te, nb_logit_te=logit_te
                )
            return p_tr, logit_tr, p_te, logit_te

    log("Building HashingVectorizer text representations...")
    word = HashingVectorizer(n_features=2**16, alternate_sign=False, ngram_range=(1, 2), norm="l2", dtype=np.float32)
    char = HashingVectorizer(n_features=2**16, alternate_sign=False, analyzer="char_wb", ngram_range=(3, 5), norm="l2", dtype=np.float32)
    X = sparse.hstack([word.transform(text_all), char.transform(text_all)], format="csr")

    ptext = np.full(n_train + n_test, np.nan, dtype=np.float32)
    fit_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_train + n_test)

    gkf = GroupKFold(n_splits=5)
    for fold, (a, b) in enumerate(gkf.split(fit_idx, y_train, sellers_train)):
        assert not set(sellers_train[a]) & set(sellers_train[b])
        ptext[b] = text_fit(X, y_train, a, b)
        log(f"  Nested NB Fold {fold+1}/5 completed in {time.time()-t0:.1f}s")

    log("  Computing NB predictions on test data from full train fit...")
    ptext[test_idx] = text_fit(X, y_train, fit_idx, test_idx)
    assert np.isfinite(ptext).all()

    p_tr = ptext[:n_train]
    p_te = ptext[n_train:]
    p_tr_c = np.clip(p_tr, 1e-6, 1.0 - 1e-6)
    p_te_c = np.clip(p_te, 1e-6, 1.0 - 1e-6)
    logit_tr = np.log(p_tr_c / (1.0 - p_tr_c))
    logit_te = np.log(p_te_c / (1.0 - p_te_c))

    if out_dir:
        np.savez_compressed(
            Path(out_dir) / "nested_nb_probabilities.npz",
            nb_prob_tr=p_tr, nb_logit_tr=logit_tr,
            nb_prob_te=p_te, nb_logit_te=logit_te
        )

    log(f"Nested Naive Bayes generation finished in {time.time()-t0:.2f}s")
    del X
    gc.collect()
    return p_tr, logit_tr, p_te, logit_te


def compute_fold_aggregations(df_base, cat_df, y_tr, trn_idx, eval_idx, global_prior, is_train=False):
    """Compute out-of-fold target encodings, category priors, and relative prices."""
    cat_trn = cat_df.iloc[trn_idx]
    cat_eval = cat_df.iloc[eval_idx]
    base_trn = df_base.iloc[trn_idx]
    base_eval = df_base.iloc[eval_idx]

    # --- Category Condition Priors (Mean & Variance) ---
    c_df = pd.DataFrame({"cat": cat_trn["category_id"].values, "y": y_tr})
    c_grp = c_df.groupby("cat")["y"]
    c_counts = c_grp.count()
    c_sums = c_grp.sum()
    alpha_c = 10.0

    if is_train:
        raw_cnt = cat_eval["category_id"].map(c_counts).fillna(0.0).values
        raw_sum = cat_eval["category_id"].map(c_sums).fillna(0.0).values
        loo_cnt = np.maximum(0.0, raw_cnt - 1.0)
        loo_sum = np.maximum(0.0, raw_sum - y_tr)
        cat_te_mean = ((loo_sum + alpha_c * global_prior) / (loo_cnt + alpha_c)).astype(np.float32)
        cat_cnt = np.log1p(loo_cnt).astype(np.float32)
    else:
        c_mean_smoothed = (c_sums + alpha_c * global_prior) / (c_counts + alpha_c)
        cat_te_mean = cat_eval["category_id"].map(c_mean_smoothed).fillna(global_prior).astype(np.float32).values
        cat_cnt = np.log1p(cat_eval["category_id"].map(c_counts).fillna(0.0).astype(np.float32).values)

    cat_te_var = (cat_te_mean * (1.0 - cat_te_mean)).astype(np.float32)

    # --- Category Price Median & Relative Price ---
    c_price_grp = pd.DataFrame({
        "cat": cat_trn["category_id"].values,
        "price": base_trn["price"].values,
        "log_price": base_trn["log_price"].values
    })
    c_price_med = c_price_grp.groupby("cat")["log_price"].median()
    c_price_raw_med = c_price_grp.groupby("cat")["price"].median()
    c_price_mean = c_price_grp.groupby("cat")["price"].mean()
    c_price_std = c_price_grp.groupby("cat")["price"].std().fillna(1.0)

    global_lp_med = float(base_trn["log_price"].median())
    global_p_med = float(base_trn["price"].median())
    global_p_mean = float(base_trn["price"].mean())
    global_p_std = float(base_trn["price"].std()) or 1.0

    eval_cat_lp_med = cat_eval["category_id"].map(c_price_med).fillna(global_lp_med).values
    eval_cat_p_med = cat_eval["category_id"].map(c_price_raw_med).fillna(global_p_med).values
    eval_cat_p_mean = cat_eval["category_id"].map(c_price_mean).fillna(global_p_mean).values
    eval_cat_p_std = cat_eval["category_id"].map(c_price_std).fillna(global_p_std).values

    rel_price_cat_med = (base_eval["log_price"].values - eval_cat_lp_med).astype(np.float32)
    price_ratio_cat_med = np.clip(base_eval["price"].values / (eval_cat_p_med + 1.0), 0.0, 100.0).astype(np.float32)
    rel_price_cat_zscore = np.clip((base_eval["price"].values - eval_cat_p_mean) / (eval_cat_p_std + 1e-4), -10.0, 10.0).astype(np.float32)

    # --- Seller Condition History (Laplace-smoothed empirical Bayesian shrinkage target encoding) ---
    s_df = pd.DataFrame({"seller": cat_trn["seller_id"].values, "y": y_tr})
    s_grp = s_df.groupby("seller")["y"]
    s_counts = s_grp.count()
    s_sums = s_grp.sum()
    alpha_s = 5.0

    if is_train:
        s_raw_cnt = cat_eval["seller_id"].map(s_counts).fillna(0.0).values
        s_raw_sum = cat_eval["seller_id"].map(s_sums).fillna(0.0).values
        s_loo_cnt = np.maximum(0.0, s_raw_cnt - 1.0)
        s_loo_sum = np.maximum(0.0, s_raw_sum - y_tr)
        seller_te = ((s_loo_sum + alpha_s * global_prior) / (s_loo_cnt + alpha_s)).astype(np.float32)
        seller_cnt = np.log1p(s_loo_cnt).astype(np.float32)
        seller_is_seen = (s_loo_cnt > 0).astype(np.float32)
    else:
        s_te_smoothed = (s_sums + alpha_s * global_prior) / (s_counts + alpha_s)
        seller_te = cat_eval["seller_id"].map(s_te_smoothed).fillna(global_prior).astype(np.float32).values
        seller_cnt_raw = cat_eval["seller_id"].map(s_counts).fillna(0.0).values
        seller_cnt = np.log1p(seller_cnt_raw).astype(np.float32)
        seller_is_seen = (seller_cnt_raw > 0).astype(np.float32)

    seller_purity = np.abs(seller_te - 0.5).astype(np.float32)

    # Frequency of categories, sellers, cities, states in train partition
    cat_freq = np.log1p(cat_eval["category_id"].map(cat_trn["category_id"].value_counts()).fillna(0.0).values).astype(np.float32)
    seller_freq = np.log1p(cat_eval["seller_id"].map(cat_trn["seller_id"].value_counts()).fillna(0.0).values).astype(np.float32)
    city_freq = np.log1p(cat_eval["city"].map(cat_trn["city"].value_counts()).fillna(0.0).values).astype(np.float32)
    state_freq = np.log1p(cat_eval["state"].map(cat_trn["state"].value_counts()).fillna(0.0).values).astype(np.float32)

    fold_df = pd.DataFrame({
        "cat_te_mean": cat_te_mean,
        "cat_te_var": cat_te_var,
        "cat_cnt": cat_cnt,
        "cat_freq": cat_freq,
        "rel_price_cat_med": rel_price_cat_med,
        "price_ratio_cat_med": price_ratio_cat_med,
        "rel_price_cat_zscore": rel_price_cat_zscore,
        "seller_te": seller_te,
        "seller_cnt": seller_cnt,
        "seller_freq": seller_freq,
        "seller_is_seen": seller_is_seen,
        "seller_purity": seller_purity,
        "city_freq": city_freq,
        "state_freq": state_freq,
    }, index=cat_eval.index)

    return fold_df


def main():
    parser = argparse.ArgumentParser(description="Train 5-Fold Stratified / Seller-Aware XGBoost Diversity Model")
    parser.add_argument("--train_zip", default=str(DEFAULT_TRAIN_ZIP), help="Path to train_data.zip")
    parser.add_argument("--test_data", default=str(DEFAULT_TEST_JSON), help="Path to test_data.jsonlines")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--seed", type=int, default=20261109, help="Random seed")
    parser.add_argument("--n_estimators", type=int, default=1200, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=7, help="Tree max depth")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Feature subsample ratio")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsample ratio")
    parser.add_argument("--reg_lambda", type=float, default=3.0, help="L2 regularization")
    parser.add_argument("--early_stopping", type=int, default=80, help="Early stopping rounds")
    parser.add_argument("--n_jobs", type=int, default=6, help="Worker threads")
    parser.add_argument("--svd_dim", type=int, default=64, help="SVD components for text TF-IDF")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log("=" * 80)
    log("=== 5-FOLD STRATIFIED / SELLER-AWARE XGBOOST DIVERSITY ENGINE ===")
    log("=" * 80)

    # 1. Load Train and Test Data
    log(f"Step 1: Loading raw datasets from {args.train_zip} and {args.test_data}...")
    with zipfile.ZipFile(args.train_zip) as z:
        train_rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    n_train = len(train_rows)
    n_test = len(test_rows)
    assert n_train == 70000, f"Expected 70,000 train rows, found {n_train}"
    assert n_test == 30000, f"Expected 30,000 test rows, found {n_test}"

    y_train = np.array([int(r["condition"] == "used") for r in train_rows], dtype=np.int32)
    train_ids = [str(r["id"]) for r in train_rows]
    test_ids = [str(r["id"]) for r in test_rows]
    train_sellers = np.array([str(r["seller_id"]) for r in train_rows])
    test_sellers = np.array([str(r["seller_id"]) for r in test_rows])
    global_prior = float(y_train.mean())
    log(f"Loaded Train: {n_train} rows (Used rate: {global_prior:.4f}), Test: {n_test} rows.")

    # 2. Extract Base Intrinsic Tabular Features
    log("Step 2: Extracting intrinsic tabular & regex features...")
    all_rows = train_rows + test_rows
    del train_rows, test_rows
    gc.collect()

    df_base_all, cat_df_all, text_corpus_all = extract_base_features(all_rows)
    del all_rows
    gc.collect()

    df_base_tr = df_base_all.iloc[:n_train].reset_index(drop=True)
    df_base_te = df_base_all.iloc[n_train:].reset_index(drop=True)
    cat_df_tr = cat_df_all.iloc[:n_train].reset_index(drop=True)
    cat_df_te = cat_df_all.iloc[n_train:].reset_index(drop=True)
    text_corpus_tr = text_corpus_all[:n_train]
    text_corpus_te = text_corpus_all[n_train:]
    del df_base_all, cat_df_all
    gc.collect()

    # 3. Sublinear TF-IDF + 64-dim TruncatedSVD
    svd_cache_path = out_dir / f"svd_{args.svd_dim}_features.npz"
    svd_cols = [f"text_svd_{i:02d}" for i in range(args.svd_dim)]
    if svd_cache_path.exists():
        log(f"Step 3: Loading cached SVD components from {svd_cache_path}...")
        svd_data = np.load(svd_cache_path)
        X_tr_svd = svd_data["X_tr_svd"]
        X_te_svd = svd_data["X_te_svd"]
    else:
        log(f"Step 3: Extracting Word/Char sublinear TF-IDF and {args.svd_dim}-dim SVD...")
        vec_word = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=3, sublinear_tf=True, dtype=np.float32)
        vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=40000, min_df=5, sublinear_tf=True, dtype=np.float32)

        log("  Fitting Word TF-IDF on train corpus...")
        X_tr_w = vec_word.fit_transform(text_corpus_tr)
        X_te_w = vec_word.transform(text_corpus_te)

        log("  Fitting Char TF-IDF on train corpus...")
        X_tr_c = vec_char.fit_transform(text_corpus_tr)
        X_te_c = vec_char.transform(text_corpus_te)

        X_tr_tfidf = sparse.hstack([X_tr_w, X_tr_c], format="csr")
        X_te_tfidf = sparse.hstack([X_te_w, X_te_c], format="csr")
        log(f"  Combined TF-IDF Shape: Train {X_tr_tfidf.shape}, Test {X_te_tfidf.shape}")

        del X_tr_w, X_te_w, X_tr_c, X_te_c
        gc.collect()

        log(f"  Computing TruncatedSVD({args.svd_dim})...")
        svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
        X_tr_svd = svd.fit_transform(X_tr_tfidf).astype(np.float32)
        X_te_svd = svd.transform(X_te_tfidf).astype(np.float32)
        log(f"  SVD Explained Variance Ratio Sum: {svd.explained_variance_ratio_.sum():.4f}")

        del X_tr_tfidf, X_te_tfidf
        gc.collect()

        np.savez_compressed(svd_cache_path, X_tr_svd=X_tr_svd, X_te_svd=X_te_svd)
        log(f"  Saved cached SVD components to {svd_cache_path}")

    df_svd_tr = pd.DataFrame(X_tr_svd, columns=svd_cols, dtype=np.float32)
    df_svd_te = pd.DataFrame(X_te_svd, columns=svd_cols, dtype=np.float32)
    del X_tr_svd, X_te_svd
    gc.collect()

    # 4. Out-of-Fold Nested Naive Bayes Text Log-Odds Feature
    log("Step 4: Generating strictly out-of-fold Nested Naive Bayes log-odds...")
    nb_prob_tr, nb_logit_tr, nb_prob_te, nb_logit_te = compute_nested_nb_features(
        text_corpus_all, y_train, train_sellers, n_train, n_test, out_dir=out_dir, seed=args.seed
    )
    del text_corpus_all, text_corpus_tr, text_corpus_te
    gc.collect()

    df_nb_tr = pd.DataFrame({
        "nested_nb_prob": nb_prob_tr,
        "nested_nb_logit": nb_logit_tr
    })
    df_nb_te = pd.DataFrame({
        "nested_nb_prob": nb_prob_te,
        "nested_nb_logit": nb_logit_te
    })

    # 5. Cross-Validation Setup: 5-Fold Stratified
    log("Step 5: Setting up 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    oof_predictions = np.zeros(n_train, dtype=np.float32)
    test_fold_probabilities = np.zeros((5, n_test), dtype=np.float32)
    fold_metrics = []

    xgb_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "colsample_bytree": args.colsample_bytree,
        "subsample": args.subsample,
        "reg_lambda": args.reg_lambda,
        "tree_method": "hist",
        "n_jobs": args.n_jobs,
        "early_stopping_rounds": args.early_stopping,
        "eval_metric": "logloss"
    }

    log(f"XGBoost Parameters: {xgb_params}")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(df_base_tr, y_train)):
        t_f0 = time.time()
        log(f"\n--- FOLD {fold+1}/5 ---")
        log(f"Train samples: {len(trn_idx)}, Val samples: {len(val_idx)}")

        # Out-of-Fold Tabular Aggregations strictly on fold train partition
        fold_trn_agg = compute_fold_aggregations(
            df_base_tr, cat_df_tr, y_train[trn_idx], trn_idx, trn_idx, global_prior, is_train=True
        )
        fold_val_agg = compute_fold_aggregations(
            df_base_tr, cat_df_tr, y_train[trn_idx], trn_idx, val_idx, global_prior, is_train=False
        )
        fold_test_agg = compute_fold_aggregations(
            pd.concat([df_base_tr.iloc[trn_idx], df_base_te], ignore_index=True),
            pd.concat([cat_df_tr.iloc[trn_idx], cat_df_te], ignore_index=True),
            y_train[trn_idx],
            np.arange(len(trn_idx)),
            np.arange(len(trn_idx), len(trn_idx) + n_test),
            global_prior,
            is_train=False
        ).reset_index(drop=True)

        # Assemble Full Feature Matrix
        X_trn = pd.concat([
            df_base_tr.iloc[trn_idx].reset_index(drop=True),
            df_svd_tr.iloc[trn_idx].reset_index(drop=True),
            df_nb_tr.iloc[trn_idx].reset_index(drop=True),
            fold_trn_agg.reset_index(drop=True)
        ], axis=1)

        X_val = pd.concat([
            df_base_tr.iloc[val_idx].reset_index(drop=True),
            df_svd_tr.iloc[val_idx].reset_index(drop=True),
            df_nb_tr.iloc[val_idx].reset_index(drop=True),
            fold_val_agg.reset_index(drop=True)
        ], axis=1)

        X_test_fold = pd.concat([
            df_base_te.reset_index(drop=True),
            df_svd_te.reset_index(drop=True),
            df_nb_te.reset_index(drop=True),
            fold_test_agg.reset_index(drop=True)
        ], axis=1)

        if fold == 0:
            log(f"Engineered feature columns count: {X_trn.shape[1]}")
            feature_names = list(X_trn.columns)

        # Fit XGBoost Model
        clf = xgb.XGBClassifier(**xgb_params, random_state=args.seed + fold)
        clf.fit(
            X_trn, y_train[trn_idx],
            eval_set=[(X_val, y_train[val_idx])],
            verbose=200
        )

        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_probs

        acc_f = float(accuracy_score(y_train[val_idx], (val_probs >= 0.5).astype(int)))
        ll_f = float(log_loss(y_train[val_idx], np.clip(val_probs, 1e-7, 1.0 - 1e-7)))
        auc_f = float(roc_auc_score(y_train[val_idx], val_probs))
        best_iter = int(clf.best_iteration) if hasattr(clf, "best_iteration") else int(args.n_estimators)

        log(f"Fold {fold+1} Finished ({time.time()-t_f0:.1f}s) | Best Iter: {best_iter} | Acc: {acc_f:.5f} | LogLoss: {ll_f:.5f} | AUC: {auc_f:.5f}")
        fold_metrics.append({
            "fold": int(fold + 1),
            "accuracy": acc_f,
            "logloss": ll_f,
            "auc": auc_f,
            "best_iteration": best_iter
        })

        # Predict on Test fold
        test_pred_fold = clf.predict_proba(X_test_fold)[:, 1]
        test_fold_probabilities[fold] = test_pred_fold

        del X_trn, X_val, X_test_fold, clf, fold_trn_agg, fold_val_agg, fold_test_agg
        gc.collect()

    # 6. Overall Out-of-Fold Evaluation
    log("\n" + "=" * 80)
    log("=== OVERALL 5-FOLD OUT-OF-FOLD EVALUATION ===")
    log("=" * 80)

    oof_preds_clipped = np.clip(oof_predictions, 1e-7, 1.0 - 1e-7)
    overall_acc_05 = float(accuracy_score(y_train, (oof_predictions >= 0.5).astype(int)))
    overall_ll = float(log_loss(y_train, oof_preds_clipped))
    overall_auc = float(roc_auc_score(y_train, oof_predictions))

    # Optimal threshold search on OOF
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

    # 7. Save Out-of-Fold Predictions
    oof_df = pd.DataFrame({
        "id": train_ids,
        "item_id": train_ids,
        "seller_id": train_sellers,
        "y_true": y_train,
        "prob_used": oof_predictions,
        "pred_condition": np.where(oof_predictions >= best_thresh, "used", "new"),
        "pred_condition_05": np.where(oof_predictions >= 0.5, "used", "new")
    })
    oof_path = out_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    log(f"Saved OOF predictions to {oof_path}")

    # 8. Test Probabilities Generation and Export
    test_prob_mean = np.mean(test_fold_probabilities, axis=0).astype(np.float32)

    test_npy_path = out_dir / "test_probabilities.npy"
    np.save(test_npy_path, test_prob_mean)
    log(f"Saved test probabilities array to {test_npy_path} (shape: {test_prob_mean.shape})")

    test_df = pd.DataFrame({
        "ID": np.arange(1, n_test + 1),
        "id": test_ids,
        "item_id": test_ids,
        "seller_id": test_sellers,
        "prob_used": test_prob_mean,
        "condition": np.where(test_prob_mean >= best_thresh, "used", "new"),
        "predicted_condition": np.where(test_prob_mean >= best_thresh, "used", "new")
    })
    test_csv_path = out_dir / "test_probabilities.csv"
    test_df.to_csv(test_csv_path, index=False)
    log(f"Saved test probabilities CSV to {test_csv_path}")

    n_used_test = int((test_df["condition"] == "used").sum())
    n_new_test = int((test_df["condition"] == "new").sum())
    log(f"Test Class Distribution: Used = {n_used_test} ({n_used_test/n_test:.2%}), New = {n_new_test} ({n_new_test/n_test:.2%})")

    # 9. Save Summary Metrics JSON
    total_duration = time.time() - t_start
    summary = {
        "architecture": "XGBClassifier",
        "num_features": len(feature_names),
        "n_train": n_train,
        "n_test": n_test,
        "global_prior": global_prior,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.lr,
            "colsample_bytree": args.colsample_bytree,
            "subsample": args.subsample,
            "reg_lambda": args.reg_lambda,
            "tree_method": "hist",
            "random_state": args.seed,
            "n_jobs": args.n_jobs,
            "early_stopping_rounds": args.early_stopping,
            "eval_metric": "logloss"
        },
        "svd_components": args.svd_dim,
        "cv_folds": fold_metrics,
        "oof_accuracy": overall_acc_05,
        "oof_accuracy_optimal": best_acc,
        "optimal_threshold": best_thresh,
        "oof_logloss": overall_ll,
        "oof_auc": overall_auc,
        "test_used_count": n_used_test,
        "test_new_count": n_new_test,
        "elapsed_seconds": round(total_duration, 2),
        "elapsed_minutes": round(total_duration / 60.0, 2),
        "output_files": {
            "oof_csv": str(oof_path),
            "test_npy": str(test_npy_path),
            "test_csv": str(test_csv_path),
            "metrics_json": str(out_dir / "metrics.json")
        }
    }

    report_path = out_dir / "metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved metrics summary to {report_path}")
    log(f"All operations finished successfully in {total_duration / 60.0:.2f} minutes.")


if __name__ == "__main__":
    main()
