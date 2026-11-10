#!/usr/bin/env python3
"""Asymmetric Lossguide XGBoost Specialist.

Author: Asymmetric XGBoost Specialist
Date: 2026-11-10

Key Protocols:
1. Strict outer StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped by seller_id.
   - Exactly ZERO seller overlap between train and validation folds.
   - ZERO seller identity features, ZERO seller target encoding (100% item-intrinsic features).
2. SVD Text Embeddings (150 components):
   - TfidfVectorizer + TruncatedSVD(n_components=150, random_state=42) strictly fit ONLY
     on outer training fold text, then transform outer train, validation, and test text.
3. Category Statistics:
   - Out-of-fold smoothed category target encoding and category price medians/z-scores
     computed strictly inside outer training fold.
4. Asymmetric Lossguide XGBoost Hyperparameters:
   - tree_method='hist'
   - grow_policy='lossguide'
   - max_leaves=120
   - max_depth=11
   - learning_rate=0.035
   - n_estimators=600
   - subsample=0.85
   - colsample_bytree=0.75
   - reg_alpha=1.0
   - reg_lambda=5.0
   - eval_metric='logloss'
   - random_state=42 + fold
   - n_jobs=-1
5. Sequential fold processing with gc.collect() and RAM profiling to ensure peak RAM < 3.0 GB.
6. Outputs:
   - results/agent_asymmetric_xgb_20261110/oof_asymmetric_xgb.csv (cols: id, p_used, target_condition_used, seller_id)
   - results/agent_asymmetric_xgb_20261110/test_probabilities_asymmetric_xgb.csv (cols: id, p_used)
   - results/agent_asymmetric_xgb_20261110/summary_report.json
"""

import os
import sys
import gc
import json
import time
import string
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
OUT_DIR = ROOT / "results" / "agent_asymmetric_xgb_20261110"

proc = psutil.Process()


def get_ram_mb() -> float:
    return proc.memory_info().rss / (1024.0 * 1024.0)


def log(msg: str):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    ram = get_ram_mb()
    print(f"[{t_str} UTC] [RAM: {ram:.1f} MB] {msg}", flush=True)


# Decisive lexical cues (compiled from verified high-impact token distributions)
LEXICAL_PATTERNS = [
    # Used indicators (is_used = 1)
    ("cue_usado", re.compile(r"\busad[oa]s?\b", re.I), 1),
    ("cue_segunda_mano", re.compile(r"\bsegunda\s+mano\b", re.I), 1),
    ("cue_reparar_repuesto", re.compile(r"\b(reparar|repuestos?|para\s+repuesto|desarme|despiece|falla|rot[oa]s?|con\s+detalles?)\b", re.I), 1),
    ("cue_detalles_uso", re.compile(r"\b(marcas?\s+de\s+uso|rayones?|desgaste|est[eé]tico)\b", re.I), 1),
    ("cue_vintage_retro", re.compile(r"\b(vintage|retro|antigu[oa]s?|antiguedades|colecci[oó]n|lote)\b", re.I), 1),
    ("cue_buen_estado", re.compile(r"\b(buen\s+estado|muy\s+buen\s+estado|impecable\s+estado|impecable|como\s+nuev[oa])\b", re.I), 1),
    ("cue_un_solo_uso", re.compile(r"\b(un|1)\s+solo\s+uso\b|\bpoco\s+uso\b|\bpoqu[ií]simo\s+uso\b", re.I), 1),
    ("cue_sin_caja", re.compile(r"\bsin\s+caja\b|\bsin\s+cargador\b", re.I), 1),
    ("cue_reacondicionado", re.compile(r"\breacondicionad[oa]\b|\brefurbished\b", re.I), 1),
    # New indicators (is_used = 0)
    ("cue_nuevo", re.compile(r"\bnuev[oa]s?\b", re.I), 0),
    ("cue_sellado", re.compile(r"\b(sellad[oa]s?|en\s+blister|cerrad[oa]\s+en\s+caja|caja\s+cerrada|caja\s+sellada|precintad[oa])\b", re.I), 0),
    ("cue_0km", re.compile(r"\b(0\s*km|cero\s*km|zero\s*km|sin\s+rodar|0km)\b", re.I), 0),
    ("cue_a_estrenar", re.compile(r"\b(a\s+estrenar|sin\s+uso|impecable\s+sin\s+uso|nuevo\s+sin\s+uso|nuevo\s+sellado)\b", re.I), 0),
    ("cue_oficial_fabrica", re.compile(r"\b(oficial|f[aá]brica|original|leg[ií]tim[oa])\b", re.I), 0),
    ("cue_artesanal", re.compile(r"\b(artesanal|hech[oa]\s+a\s+mano|souvenir)\b", re.I), 0),
    ("cue_digital", re.compile(r"\b(libro\s+digital|ebook|pdf|descarga\s+digital)\b", re.I), 0),
    ("cue_garantia", re.compile(r"\bcon\s+garant[ií]a\b|\bgarant[ií]a\s+oficial\b|\bgarant[ií]a\s+escrita\b", re.I), 0),
    ("cue_factura", re.compile(r"\bfactura\s+[ab]\b|\bhacemos\s+factura\b", re.I), 0),
]


def extract_base_intrinsic_features(rows):
    """Extract strictly item-intrinsic tabular features (ZERO seller identity)."""
    t0 = time.time()
    n = len(rows)
    feats = {}

    # 1. Price features
    price = np.array([float(r.get("price") or 0.0) for r in rows], dtype=np.float32)
    base_price = np.array([float(r.get("base_price") or 0.0) for r in rows], dtype=np.float32)
    orig_price = np.array([float(r.get("original_price") or 0.0) for r in rows], dtype=np.float32)
    has_orig = np.array([1.0 if r.get("original_price") is not None else 0.0 for r in rows], dtype=np.float32)

    feats["price"] = price
    feats["log_price"] = np.log1p(np.maximum(0.0, price))
    feats["base_price"] = base_price
    feats["log_base_price"] = np.log1p(np.maximum(0.0, base_price))
    feats["has_orig_price"] = has_orig
    feats["log_orig_price"] = np.log1p(np.maximum(0.0, orig_price))
    feats["price_orig_ratio"] = np.where((has_orig > 0) & (price > 0), np.clip(orig_price / (price + 1e-4), 0.0, 10.0), 1.0).astype(np.float32)
    feats["price_diff"] = (base_price - price).astype(np.float32)
    feats["discount_rate"] = np.clip((base_price - price) / (base_price + 1e-4), -1.0, 1.0).astype(np.float32)
    feats["price_round10"] = ((price % 10) == 0).astype(np.float32)
    feats["price_round100"] = ((price % 100) == 0).astype(np.float32)
    feats["price_end9"] = ((np.round(price) % 10) == 9).astype(np.float32)

    # 2. Inventory / sales quantities
    init_q = np.array([float(r.get("initial_quantity") or 0.0) for r in rows], dtype=np.float32)
    avail_q = np.array([float(r.get("available_quantity") or 0.0) for r in rows], dtype=np.float32)
    sold_q = np.array([float(r.get("sold_quantity") or 0.0) for r in rows], dtype=np.float32)

    feats["initial_quantity"] = init_q
    feats["available_quantity"] = avail_q
    feats["sold_quantity"] = sold_q
    feats["log_iq"] = np.log1p(np.maximum(0.0, init_q))
    feats["log_aq"] = np.log1p(np.maximum(0.0, avail_q))
    feats["log_sq"] = np.log1p(np.maximum(0.0, sold_q))
    feats["sold_ratio"] = (sold_q / (init_q + 1.0)).astype(np.float32)
    feats["avail_ratio"] = (avail_q / (init_q + 1.0)).astype(np.float32)
    feats["init_minus_avail"] = (init_q - avail_q).astype(np.float32)

    # 3. Structural item attributes
    feats["accepts_mercadopago"] = np.array([1.0 if r.get("accepts_mercadopago") else 0.0 for r in rows], dtype=np.float32)
    shipping = [r.get("shipping") or {} for r in rows]
    feats["free_shipping"] = np.array([1.0 if s.get("free_shipping") else 0.0 for s in shipping], dtype=np.float32)
    feats["local_pickup"] = np.array([1.0 if s.get("local_pick_up") else 0.0 for s in shipping], dtype=np.float32)

    feats["n_pictures"] = np.array([float(len(r.get("pictures") or [])) for r in rows], dtype=np.float32)
    feats["n_attributes"] = np.array([float(len(r.get("attributes") or [])) for r in rows], dtype=np.float32)
    feats["n_tags"] = np.array([float(len(r.get("tags") or [])) for r in rows], dtype=np.float32)
    feats["n_variations"] = np.array([float(len(r.get("variations") or [])) for r in rows], dtype=np.float32)
    feats["has_video"] = np.array([1.0 if r.get("video_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_catalog"] = np.array([1.0 if r.get("catalog_product_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["has_parent"] = np.array([1.0 if r.get("parent_item_id") is not None else 0.0 for r in rows], dtype=np.float32)
    feats["automatic_relist"] = np.array([1.0 if r.get("automatic_relist") else 0.0 for r in rows], dtype=np.float32)

    # Warranty
    warranties = [str(r.get("warranty") or "").strip() for r in rows]
    feats["has_warranty"] = np.array([1.0 if len(w) > 0 else 0.0 for w in warranties], dtype=np.float32)
    feats["warranty_len"] = np.array([float(len(w)) for w in warranties], dtype=np.float32)

    # 4. Title intrinsic textual statistics
    titles = [str(r.get("title") or "").strip() for r in rows]
    feats["title_len"] = np.array([float(len(t)) for t in titles], dtype=np.float32)
    feats["title_words"] = np.array([float(len(t.split())) for t in titles], dtype=np.float32)
    feats["digit_ratio"] = np.array([sum(c.isdigit() for c in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["uppercase_ratio"] = np.array([sum(c.isupper() for c in t) / max(len(t), 1) for t in titles], dtype=np.float32)
    feats["question_mark_count"] = np.array([float(t.count("?") + t.count("¿")) for t in titles], dtype=np.float32)
    feats["exclamation_mark_count"] = np.array([float(t.count("!") + t.count("¡")) for t in titles], dtype=np.float32)
    feats["title_char_count"] = np.array([float(len(t.replace(" ", ""))) for t in titles], dtype=np.float32)
    feats["punctuation_count"] = np.array([float(sum(c in string.punctuation for c in t)) for t in titles], dtype=np.float32)

    # 5. Lexical cue indicators
    used_cues = np.zeros(n, dtype=np.float32)
    new_cues = np.zeros(n, dtype=np.float32)
    for name, rx, is_used in LEXICAL_PATTERNS:
        match_arr = np.array([1.0 if rx.search(t) else 0.0 for t in titles], dtype=np.float32)
        feats[name] = match_arr
        if is_used == 1:
            used_cues += match_arr
        else:
            new_cues += match_arr

    feats["used_cues_total"] = used_cues
    feats["new_cues_total"] = new_cues
    feats["net_cue_score"] = new_cues - used_cues

    df_base = pd.DataFrame(feats)
    log(f"Extracted {df_base.shape[1]} intrinsic item features for {n} items in {time.time()-t0:.2f}s")
    return df_base


def compute_fold_category_stats(cat_tr, y_tr, log_price_tr, cat_eval, log_price_eval, is_train=False, m=20.0):
    """Compute smoothed target encoding and price stats for category_id strictly on outer training fold."""
    # Smoothed Category Target Encoding
    global_mean = float(y_tr.mean())
    df_cat = pd.DataFrame({"cat": cat_tr, "y": y_tr, "price": log_price_tr})

    cat_stats = df_cat.groupby("cat").agg(
        count=("y", "count"),
        sum_y=("y", "sum"),
        med_price=("price", "median"),
        std_price=("price", "std"),
    )

    # Smoothed target rate: (sum_y + m * global_mean) / (count + m)
    cat_target_map = ((cat_stats["sum_y"] + m * global_mean) / (cat_stats["count"] + m)).to_dict()
    cat_freq_map = np.log1p(cat_stats["count"]).to_dict()

    global_med_price = float(np.median(log_price_tr))
    global_std_price = float(np.std(log_price_tr)) if float(np.std(log_price_tr)) > 0 else 1.0
    cat_med_price_map = cat_stats["med_price"].to_dict()
    cat_std_price_map = cat_stats["std_price"].fillna(global_std_price).to_dict()

    eval_cat_series = pd.Series(cat_eval)

    if is_train:
        # Leave-One-Out target encoding to prevent self-target leakage on train
        c_sums = eval_cat_series.map(cat_stats["sum_y"]).fillna(0.0).values
        c_cnts = eval_cat_series.map(cat_stats["count"]).fillna(0.0).values
        eval_target_te = ((c_sums - y_tr + m * global_mean) / np.maximum(c_cnts - 1.0 + m, 1.0)).astype(np.float32)
    else:
        eval_target_te = eval_cat_series.map(cat_target_map).fillna(global_mean).values.astype(np.float32)

    eval_freq = eval_cat_series.map(cat_freq_map).fillna(0.0).values.astype(np.float32)
    eval_med_p = eval_cat_series.map(cat_med_price_map).fillna(global_med_price).values.astype(np.float32)
    eval_std_p = eval_cat_series.map(cat_std_price_map).fillna(global_std_price).values.astype(np.float32)
    eval_std_p = np.where(eval_std_p <= 0.0, global_std_price, eval_std_p)

    rel_price_med = (log_price_eval - eval_med_p).astype(np.float32)
    cat_price_zscore = np.clip((log_price_eval - eval_med_p) / (eval_std_p + 1e-4), -10.0, 10.0).astype(np.float32)

    return np.column_stack([
        eval_target_te,
        eval_freq,
        eval_med_p,
        rel_price_med,
        cat_price_zscore,
    ]).astype(np.float32)


def compute_frequency_encoding(meta_tr, meta_eval):
    """Compute frequency encoding log1p(count) strictly on outer train fold."""
    counts = Counter(meta_tr)
    return np.log1p(pd.Series(meta_eval).map(counts).fillna(0.0).values.astype(np.float32))


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("=== ASYMMETRIC LOSSGUIDE XGBOOST SPECIALIST TRAINING ===")
    log("=== OUTER STRATIFIED GROUP K-FOLD (5 SPLITS) GROUPED BY SELLER_ID ===")
    log("=" * 80)

    # 1. Load Data
    log(f"Loading train data from {TRAIN_JSON}...")
    train_rows = []
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))
    n_train = len(train_rows)

    log(f"Loading test data from {TEST_JSON}...")
    test_rows = []
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))
    n_test = len(test_rows)

    assert n_train == 70000, f"Expected 70,000 train rows, got {n_train}"
    assert n_test == 30000, f"Expected 30,000 test rows, got {n_test}"

    # Target: 1 for used, 0 for new
    y_train = np.array([1 if r["condition"] == "used" else 0 for r in train_rows], dtype=np.int32)
    sellers_train = np.array([str(r.get("seller_id") or "UNKNOWN") for r in train_rows])
    train_ids = [str(r.get("id") or "") for r in train_rows]
    test_ids = [str(r.get("id") or "") for r in test_rows]

    titles_train = [str(r.get("title") or "").strip() for r in train_rows]
    titles_test = [str(r.get("title") or "").strip() for r in test_rows]

    cats_train = np.array([str(r.get("category_id") or "MISSING") for r in train_rows])
    cats_test = np.array([str(r.get("category_id") or "MISSING") for r in test_rows])

    listing_type_train = np.array([str(r.get("listing_type_id") or "MISSING") for r in train_rows])
    listing_type_test = np.array([str(r.get("listing_type_id") or "MISSING") for r in test_rows])

    buying_mode_train = np.array([str(r.get("buying_mode") or "MISSING") for r in train_rows])
    buying_mode_test = np.array([str(r.get("buying_mode") or "MISSING") for r in test_rows])

    ship_mode_train = np.array([str((r.get("shipping") or {}).get("mode") or "MISSING") for r in train_rows])
    ship_mode_test = np.array([str((r.get("shipping") or {}).get("mode") or "MISSING") for r in test_rows])

    log(f"Train items: {n_train} | Used items: {int(y_train.sum())} ({y_train.mean()*100:.2f}%)")
    log(f"Test items: {n_test}")
    log(f"Unique sellers in train: {len(np.unique(sellers_train))}")

    # 2. Extract Base Intrinsic Features (ZERO seller identity features)
    log("Extracting item-intrinsic base tabular features (ZERO seller identity features)...")
    all_rows = train_rows + test_rows
    del train_rows, test_rows
    gc.collect()

    df_base_all = extract_base_intrinsic_features(all_rows)
    del all_rows
    gc.collect()

    df_base_tr = df_base_all.iloc[:n_train].reset_index(drop=True)
    df_base_te = df_base_all.iloc[n_train:].reset_index(drop=True)
    del df_base_all
    gc.collect()

    log_price_tr = df_base_tr["log_price"].values
    log_price_te = df_base_te["log_price"].values

    # 3. Setup StratifiedGroupKFold(5)
    log("Initializing StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(df_base_tr.values, y_train, groups=sellers_train))

    # Audit zero seller leakage
    log("Auditing seller overlap across outer folds...")
    for f_idx, (trn_idx, val_idx) in enumerate(splits):
        trn_sellers = set(sellers_train[trn_idx])
        val_sellers = set(sellers_train[val_idx])
        overlap = len(trn_sellers.intersection(val_sellers))
        assert overlap == 0, f"FATAL: Fold {f_idx + 1} has seller overlap of {overlap}!"
        log(f"Fold {f_idx + 1}: Train={len(trn_idx)} ({len(trn_sellers)} sellers), Val={len(val_idx)} ({len(val_sellers)} sellers), Overlap=0")
    log("AUDIT PASSED: Exactly ZERO seller overlap across all 5 folds.")

    oof_p = np.zeros(n_train, dtype=np.float32)
    test_p_folds = np.zeros((5, n_test), dtype=np.float32)
    fold_metrics = []
    peak_ram_mb = get_ram_mb()

    # 4. Fold Processing Loop
    for fold, (trn_idx, val_idx) in enumerate(splits):
        f_num = fold + 1
        t_f0 = time.time()
        log("-" * 70)
        log(f"--- STARTING FOLD {f_num}/5 (Train: {len(trn_idx)}, Val: {len(val_idx)}) ---")

        # Step A: Category statistics computed strictly on trn_idx
        log(f"[Fold {f_num}] Computing leak-free category statistics and frequencies...")
        cat_tr_stats = compute_fold_category_stats(
            cats_train[trn_idx], y_train[trn_idx], log_price_tr[trn_idx],
            cats_train[trn_idx], log_price_tr[trn_idx], is_train=True
        )
        cat_val_stats = compute_fold_category_stats(
            cats_train[trn_idx], y_train[trn_idx], log_price_tr[trn_idx],
            cats_train[val_idx], log_price_tr[val_idx], is_train=False
        )
        cat_te_stats = compute_fold_category_stats(
            cats_train[trn_idx], y_train[trn_idx], log_price_tr[trn_idx],
            cats_test, log_price_te, is_train=False
        )

        # Listing, buying, ship mode frequencies strictly on outer-train
        freq_lt_tr = compute_frequency_encoding(listing_type_train[trn_idx], listing_type_train[trn_idx])
        freq_lt_val = compute_frequency_encoding(listing_type_train[trn_idx], listing_type_train[val_idx])
        freq_lt_te = compute_frequency_encoding(listing_type_train[trn_idx], listing_type_test)

        freq_bm_tr = compute_frequency_encoding(buying_mode_train[trn_idx], buying_mode_train[trn_idx])
        freq_bm_val = compute_frequency_encoding(buying_mode_train[trn_idx], buying_mode_train[val_idx])
        freq_bm_te = compute_frequency_encoding(buying_mode_train[trn_idx], buying_mode_test)

        freq_sm_tr = compute_frequency_encoding(ship_mode_train[trn_idx], ship_mode_train[trn_idx])
        freq_sm_val = compute_frequency_encoding(ship_mode_train[trn_idx], ship_mode_train[val_idx])
        freq_sm_te = compute_frequency_encoding(ship_mode_train[trn_idx], ship_mode_test)

        tab_tr = np.column_stack([
            df_base_tr.iloc[trn_idx].values,
            cat_tr_stats,
            freq_lt_tr,
            freq_bm_tr,
            freq_sm_tr
        ]).astype(np.float32)

        tab_val = np.column_stack([
            df_base_tr.iloc[val_idx].values,
            cat_val_stats,
            freq_lt_val,
            freq_bm_val,
            freq_sm_val
        ]).astype(np.float32)

        tab_te = np.column_stack([
            df_base_te.values,
            cat_te_stats,
            freq_lt_te,
            freq_bm_te,
            freq_sm_te
        ]).astype(np.float32)

        del cat_tr_stats, cat_val_stats, cat_te_stats
        del freq_lt_tr, freq_lt_val, freq_lt_te
        del freq_bm_tr, freq_bm_val, freq_bm_te
        del freq_sm_tr, freq_sm_val, freq_sm_te
        gc.collect()

        # Step B: Fit TfidfVectorizer + TruncatedSVD(150) strictly on trn_idx
        log(f"[Fold {f_num}] Fitting TfidfVectorizer + TruncatedSVD(n_components=150) strictly on train text...")
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=35000,
            sublinear_tf=True,
            min_df=2
        )
        trn_corpus = [titles_train[i] for i in trn_idx]
        val_corpus = [titles_train[i] for i in val_idx]

        X_tr_tfidf = tfidf.fit_transform(trn_corpus)
        X_val_tfidf = tfidf.transform(val_corpus)
        X_te_tfidf = tfidf.transform(titles_test)
        del trn_corpus, val_corpus
        gc.collect()

        svd = TruncatedSVD(n_components=150, random_state=42)
        X_tr_svd = svd.fit_transform(X_tr_tfidf).astype(np.float32)
        X_val_svd = svd.transform(X_val_tfidf).astype(np.float32)
        X_te_svd = svd.transform(X_te_tfidf).astype(np.float32)

        del tfidf, svd, X_tr_tfidf, X_val_tfidf, X_te_tfidf
        gc.collect()

        # Step C: Combine tabular + SVD features
        X_tr = np.column_stack([tab_tr, X_tr_svd])
        X_val = np.column_stack([tab_val, X_val_svd])
        X_te = np.column_stack([tab_te, X_te_svd])

        del tab_tr, tab_val, tab_te, X_tr_svd, X_val_svd, X_te_svd
        gc.collect()

        log(f"[Fold {f_num}] Feature matrix ready: X_tr={X_tr.shape}, X_val={X_val.shape}, X_te={X_te.shape}")

        # Step D: Train XGBoost Classifier with lossguide
        log(f"[Fold {f_num}] Training XGBClassifier (tree_method='hist', grow_policy='lossguide', max_leaves=120, max_depth=11, n_estimators=600, lr=0.035)...")
        model = xgb.XGBClassifier(
            tree_method="hist",
            grow_policy="lossguide",
            max_leaves=120,
            max_depth=11,
            learning_rate=0.035,
            n_estimators=600,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=5.0,
            eval_metric="logloss",
            random_state=42 + fold,
            n_jobs=-1,
        )

        model.fit(X_tr, y_train[trn_idx])

        # Step E: Predict validation and test probabilities
        p_val = model.predict_proba(X_val)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]

        oof_p[val_idx] = p_val
        test_p_folds[fold] = p_te

        val_acc = accuracy_score(y_train[val_idx], (p_val >= 0.5).astype(int))
        val_auc = roc_auc_score(y_train[val_idx], p_val)
        val_ll = log_loss(y_train[val_idx], np.clip(p_val, 1e-15, 1 - 1e-15))
        elapsed_fold = time.time() - t_f0

        current_ram = get_ram_mb()
        peak_ram_mb = max(peak_ram_mb, current_ram)
        log(f"[Fold {f_num} FINISHED] Time: {elapsed_fold:.1f}s | Acc: {val_acc:.5f} | ROC-AUC: {val_auc:.5f} | LogLoss: {val_ll:.5f}")

        fold_metrics.append({
            "fold": f_num,
            "n_train": len(trn_idx),
            "n_val": len(val_idx),
            "accuracy": float(val_acc),
            "roc_auc": float(val_auc),
            "logloss": float(val_ll),
            "elapsed_seconds": float(elapsed_fold),
            "ram_mb": float(current_ram),
        })

        del X_tr, X_val, X_te, model, p_val, p_te
        gc.collect()

    # 5. Global OOF Evaluation
    log("=" * 80)
    global_acc = accuracy_score(y_train, (oof_p >= 0.5).astype(int))
    global_auc = roc_auc_score(y_train, oof_p)
    global_ll = log_loss(y_train, np.clip(oof_p, 1e-15, 1 - 1e-15))
    total_hits = int(((oof_p >= 0.5).astype(int) == y_train).sum())
    total_duration = time.time() - t_start

    log(f"=== GLOBAL ASYMMETRIC XGBOOST OUT-OF-FOLD PERFORMANCE ===")
    log(f"Accuracy: {global_acc:.5f} ({total_hits}/{n_train})")
    log(f"ROC-AUC:  {global_auc:.5f}")
    log(f"LogLoss:  {global_ll:.5f}")
    log(f"Total Execution Time: {total_duration:.1f}s")
    log(f"Peak RAM: {peak_ram_mb:.1f} MB")
    log("=" * 80)

    # 6. Save Artifacts
    oof_csv_path = OUT_DIR / "oof_asymmetric_xgb.csv"
    test_csv_path = OUT_DIR / "test_probabilities_asymmetric_xgb.csv"
    summary_path = OUT_DIR / "summary_report.json"

    # OOF CSV (cols: id, p_used, target_condition_used, seller_id)
    df_oof = pd.DataFrame({
        "id": np.arange(1, n_train + 1),
        "p_used": oof_p,
        "p_used_asymmetric_xgb": oof_p,
        "target_condition_used": y_train,
        "seller_id": sellers_train,
        "item_id": train_ids,
    })
    df_oof.to_csv(oof_csv_path, index=False)
    log(f"Saved OOF predictions to {oof_csv_path} (shape: {df_oof.shape})")

    # Test CSV (cols: id, p_used)
    test_p_mean = test_p_folds.mean(axis=0)
    df_test = pd.DataFrame({
        "id": np.arange(1, n_test + 1),
        "p_used": test_p_mean,
        "p_used_asymmetric_xgb": test_p_mean,
        "item_id": test_ids,
    })
    df_test.to_csv(test_csv_path, index=False)
    log(f"Saved Test predictions to {test_csv_path} (shape: {df_test.shape})")

    # Summary JSON
    summary = {
        "model_name": "Asymmetric Lossguide XGBoost Specialist (max_leaves=120, max_depth=11, lr=0.035, n_estimators=600, SVD=150)",
        "hyperparameters": {
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "max_leaves": 120,
            "max_depth": 11,
            "learning_rate": 0.035,
            "n_estimators": 600,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "eval_metric": "logloss",
            "n_svd_components": 150,
            "random_state_base": 42
        },
        "cv_protocol": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped strictly by seller_id",
        "leak_free_audit": {
            "seller_leakage": "ZERO seller overlap between train and validation across all 5 folds",
            "seller_features": "ZERO seller identity features, ZERO seller target encoding, ZERO seller frequency in model input",
            "svd_projection": "TfidfVectorizer and TruncatedSVD fit strictly inside each outer training fold text only",
            "category_statistics": "Category smoothed target encoding and price medians computed strictly inside each outer training fold"
        },
        "n_train": n_train,
        "n_test": n_test,
        "global_accuracy": float(global_acc),
        "global_roc_auc": float(global_auc),
        "global_logloss": float(global_ll),
        "hits": total_hits,
        "total_duration_seconds": float(total_duration),
        "peak_ram_mb": float(peak_ram_mb),
        "fold_metrics": fold_metrics,
        "output_files": {
            "oof_csv": str(oof_csv_path),
            "test_csv": str(test_csv_path),
            "summary_json": str(summary_path)
        }
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary report to {summary_path}")
    log("All tasks completed successfully.")


if __name__ == "__main__":
    main()
