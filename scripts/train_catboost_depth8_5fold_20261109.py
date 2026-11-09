#!/usr/bin/env python3
"""5-Fold Stratified / Seller-Aware CatBoost Depth 8 Native Text Model.

Full 70,000 train rows with 5-fold cross-validation.
Native text feature processing on normalized title + warranty + attributes + tags.
Deep tabular features: prices, relative prices, quantities, seller/category frequencies, listing/shipping structure.
Outputs:
  - results/catboost_depth8_5fold/oof_predictions.csv
  - results/catboost_depth8_5fold/test_probabilities.npy
  - results/catboost_depth8_5fold/test_probabilities.csv
  - results/catboost_depth8_5fold/metrics.json
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ZIP = ROOT / "data" / "train_data.zip"
DEFAULT_TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
DEFAULT_OUT_DIR = ROOT / "results" / "catboost_depth8_5fold"


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)


def normalize_text(text: str) -> str:
    """Normalize unicode accents, strip excess punctuation/whitespace, lowercase."""
    if not text or not isinstance(text, str):
        return ""
    # Unicode decomposition to strip accents
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_clean_text(r: dict) -> str:
    """Build rich normalized text combining title + warranty + attributes + tags."""
    parts = []

    # Title
    title = normalize_text(r.get("title"))
    if title:
        parts.append(title)

    # Warranty
    warranty = normalize_text(r.get("warranty"))
    if warranty:
        parts.append(warranty)

    # Attributes (name + value_name / value_id)
    attrs = r.get("attributes") or []
    attr_parts = []
    for a in attrs:
        if isinstance(a, dict):
            nm = normalize_text(a.get("name"))
            val = normalize_text(a.get("value_name") or a.get("value_id"))
            if nm and val:
                attr_parts.append(f"{nm} {val}")
            elif nm:
                attr_parts.append(nm)
            elif val:
                attr_parts.append(val)
    if attr_parts:
        parts.append(" ".join(attr_parts))

    # Tags
    tags = r.get("tags") or []
    tag_parts = [normalize_text(t) for t in tags if t]
    if tag_parts:
        parts.append(" ".join(tag_parts))

    full = " ".join(parts).strip()
    return full if full else "missing"


def dims(p: dict):
    try:
        w, h = p.get("max_size", p.get("size", "0x0")).split("x")
        return int(w), int(h)
    except Exception:
        return 0, 0


def extract_features_df(rows: list) -> pd.DataFrame:
    """Extract deep tabular features and normalized rich text from row dictionaries."""
    clean_texts = [build_clean_text(r) for r in rows]
    prices = np.array([float(r.get("price") or 0.0) for r in rows], dtype=np.float32)
    base_prices = np.array([float(r.get("base_price") or 0.0) for r in rows], dtype=np.float32)
    init_q = np.array([float(r.get("initial_quantity") or 0.0) for r in rows], dtype=np.float32)
    avail_q = np.array([float(r.get("available_quantity") or 0.0) for r in rows], dtype=np.float32)
    sold_q = np.array([float(r.get("sold_quantity") or 0.0) for r in rows], dtype=np.float32)

    orig_prices = np.array([float(r.get("original_price") or 0.0) for r in rows], dtype=np.float32)
    has_orig = np.array([1.0 if r.get("original_price") is not None else 0.0 for r in rows], dtype=np.float32)

    # Picture statistics
    n_pics, max_area, mean_area = [], [], []
    for r in rows:
        pics = r.get("pictures") or []
        n_pics.append(float(len(pics)))
        if pics:
            areas = [dims(p)[0] * dims(p)[1] for p in pics]
            max_area.append(float(max(areas)))
            mean_area.append(float(np.mean(areas)))
        else:
            max_area.append(-1.0)
            mean_area.append(-1.0)

    # Dates and durations
    ref = pd.Timestamp("2013-01-01", tz="UTC")
    dates_created = pd.to_datetime([r.get("date_created") for r in rows], utc=True, errors="coerce")
    dates_updated = pd.to_datetime([r.get("last_updated") for r in rows], utc=True, errors="coerce")
    st = pd.to_datetime([r.get("start_time") for r in rows], unit="ms", utc=True, errors="coerce")
    sp = pd.to_datetime([r.get("stop_time") for r in rows], unit="ms", utc=True, errors="coerce")

    created_days = np.nan_to_num((dates_created - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    dur_days = np.nan_to_num((sp - st).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    upd_minus_created = np.nan_to_num((dates_updated - dates_created).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    created_hour = np.nan_to_num(dates_created.hour, nan=0.0).astype(np.float32)
    created_dow = np.nan_to_num(dates_created.dayofweek, nan=0.0).astype(np.float32)

    # Warranty and title regex cues
    warranties = [str(r.get("warranty") or "").lower() for r in rows]
    titles = [str(r.get("title") or "").lower() for r in rows]

    data = {
        # Rich Normalized Text
        "clean_text": clean_texts,
        # Price features
        "price": prices,
        "log_price": np.log1p(np.maximum(0, prices)),
        "base_price": base_prices,
        "log_base_price": np.log1p(np.maximum(0, base_prices)),
        "price_diff": base_prices - prices,
        "price_eq_base": (prices == base_prices).astype(np.float32),
        "has_original_price": has_orig,
        "log_original_price": np.log1p(np.maximum(0, orig_prices)),
        "price_cents": ((prices * 100) % 100).round(),
        "price_round100": ((prices % 100) == 0).astype(np.float32),
        "price_round10": ((prices % 10) == 0).astype(np.float32),
        "price_end9": ((np.round(prices) % 10) == 9).astype(np.float32),
        "currency_usd": np.array([1.0 if r.get("currency_id") == "USD" else 0.0 for r in rows], dtype=np.float32),
        # Quantities
        "initial_quantity": init_q,
        "available_quantity": avail_q,
        "sold_quantity": sold_q,
        "sold_ratio": sold_q / (init_q + 1.0),
        "avail_ratio": avail_q / (init_q + 1.0),
        "init_minus_avail": init_q - avail_q,
        # Listing structure
        "n_pics": np.array(n_pics, dtype=np.float32),
        "pic_max_area": np.array(max_area, dtype=np.float32),
        "pic_mean_area": np.array(mean_area, dtype=np.float32),
        "n_variations": np.array([float(len(r.get("variations") or [])) for r in rows], dtype=np.float32),
        "n_attributes": np.array([float(len(r.get("attributes") or [])) for r in rows], dtype=np.float32),
        "n_tags": np.array([float(len(r.get("tags") or [])) for r in rows], dtype=np.float32),
        "n_deals": np.array([float(len(r.get("deal_ids") or [])) for r in rows], dtype=np.float32),
        "n_pm": np.array([float(len(r.get("non_mercado_pago_payment_methods") or [])) for r in rows], dtype=np.float32),
        "accepts_mp": np.array([1.0 if r.get("accepts_mercadopago") else 0.0 for r in rows], dtype=np.float32),
        "automatic_relist": np.array([1.0 if r.get("automatic_relist") else 0.0 for r in rows], dtype=np.float32),
        "official_store": np.array([1.0 if r.get("official_store_id") is not None else 0.0 for r in rows], dtype=np.float32),
        "has_video": np.array([1.0 if r.get("video_id") is not None else 0.0 for r in rows], dtype=np.float32),
        "has_catalog": np.array([1.0 if r.get("catalog_product_id") is not None else 0.0 for r in rows], dtype=np.float32),
        "has_parent": np.array([1.0 if r.get("parent_item_id") is not None else 0.0 for r in rows], dtype=np.float32),
        # Shipping details
        "ship_free": np.array([1.0 if (r.get("shipping") or {}).get("free_shipping") else 0.0 for r in rows], dtype=np.float32),
        "ship_local_pickup": np.array([1.0 if (r.get("shipping") or {}).get("local_pick_up") else 0.0 for r in rows], dtype=np.float32),
        "ship_dims": np.array([1.0 if (r.get("shipping") or {}).get("dimensions") is not None else 0.0 for r in rows], dtype=np.float32),
        "ship_n_methods": np.array([float(len((r.get("shipping") or {}).get("methods") or [])) for r in rows], dtype=np.float32),
        "ship_n_tags": np.array([float(len((r.get("shipping") or {}).get("tags") or [])) for r in rows], dtype=np.float32),
        # Temporal
        "created_days": created_days,
        "dur_days": dur_days,
        "upd_minus_created": upd_minus_created,
        "created_hour": created_hour,
        "created_dow": created_dow,
        # Warranty signals
        "has_warranty": np.array([1.0 if bool(r.get("warranty")) else 0.0 for r in rows], dtype=np.float32),
        "warranty_len": np.array([float(len(w)) for w in warranties], dtype=np.float32),
        "w_sin": np.array([1.0 if bool(re.search(r"\bsin\b", w)) else 0.0 for w in warranties], dtype=np.float32),
        "w_no": np.array([1.0 if bool(re.search(r"\bno\b", w)) else 0.0 for w in warranties], dtype=np.float32),
        "w_meses": np.array([1.0 if bool(re.search(r"mes", w)) else 0.0 for w in warranties], dtype=np.float32),
        "w_anio": np.array([1.0 if bool(re.search(r"a[ñn]o", w)) else 0.0 for w in warranties], dtype=np.float32),
        "w_fabrica": np.array([1.0 if bool(re.search(r"f[aá]brica", w)) else 0.0 for w in warranties], dtype=np.float32),
        "w_oficial": np.array([1.0 if bool(re.search(r"oficial", w)) else 0.0 for w in warranties], dtype=np.float32),
        "w_usado": np.array([1.0 if bool(re.search(r"usad", w)) else 0.0 for w in warranties], dtype=np.float32),
        "w_nuevo": np.array([1.0 if bool(re.search(r"nuev", w)) else 0.0 for w in warranties], dtype=np.float32),
        "w_dias": np.array([1.0 if bool(re.search(r"d[ií]as", w)) else 0.0 for w in warranties], dtype=np.float32),
        "w_garantia": np.array([1.0 if bool(re.search(r"garant", w)) else 0.0 for w in warranties], dtype=np.float32),
        # Title regex
        "title_len": np.array([float(len(t)) for t in titles], dtype=np.float32),
        "title_words": np.array([float(len(t.split())) for t in titles], dtype=np.float32),
        "t_usado": np.array([1.0 if bool(re.search(r"usad", t)) else 0.0 for t in titles], dtype=np.float32),
        "t_nuevo": np.array([1.0 if bool(re.search(r"nuev", t)) else 0.0 for t in titles], dtype=np.float32),
        "t_original": np.array([1.0 if bool(re.search(r"original", t)) else 0.0 for t in titles], dtype=np.float32),
        "t_antiguo": np.array([1.0 if bool(re.search(r"antig|vintage|retro|colecci", t)) else 0.0 for t in titles], dtype=np.float32),
        "t_libro": np.array([1.0 if bool(re.search(r"libro", t)) else 0.0 for t in titles], dtype=np.float32),
        "t_lote": np.array([1.0 if bool(re.search(r"lote", t)) else 0.0 for t in titles], dtype=np.float32),
        "t_impecable": np.array([1.0 if bool(re.search(r"impecable|excelente estado|muy buen estado", t)) else 0.0 for t in titles], dtype=np.float32),
        # Metadata / Categoricals
        "listing_type_id": [str(r.get("listing_type_id") or "MISSING") for r in rows],
        "shipping_mode": [str((r.get("shipping") or {}).get("mode") or "MISSING") for r in rows],
        "state": [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "MISSING") for r in rows],
        "category_id": [str(r.get("category_id") or "MISSING") for r in rows],
        "buying_mode": [str(r.get("buying_mode") or "MISSING") for r in rows],
        "currency_id": [str(r.get("currency_id") or "MISSING") for r in rows],
        "status": [str(r.get("status") or "MISSING") for r in rows],
        "seller_id": [str(r.get("seller_id") or "MISSING") for r in rows],
        "id": [str(r.get("id") or "") for r in rows],
    }
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Train 5-Fold Stratified / Seller-aware CatBoost Depth 8 Model")
    parser.add_argument("--train-zip", type=str, default=str(DEFAULT_TRAIN_ZIP), help="Path to train_data.zip")
    parser.add_argument("--test-json", type=str, default=str(DEFAULT_TEST_JSON), help="Path to test_data.jsonlines")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--cv-scheme", type=str, default="stratified", choices=["stratified", "seller_group"], help="CV Scheme")
    parser.add_argument("--seed", type=int, default=20261109, help="Random seed")
    parser.add_argument("--threads", type=int, default=6, help="Thread count for CatBoost")
    parser.add_argument("--iterations", type=int, default=1800, help="Max iterations")
    parser.add_argument("--depth", type=int, default=8, help="Tree depth")
    parser.add_argument("--lr", type=float, default=0.04, help="Learning rate")
    parser.add_argument("--l2", type=float, default=5.0, help="L2 leaf reg")
    parser.add_argument("--early-stopping", type=int, default=100, help="Early stopping rounds")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log("=" * 80)
    log("=== CATBOOST DEPTH 8 5-FOLD TRAINING ENGINE (2026-11-09) ===")
    log("=" * 80)
    log(f"Parameters: cv={args.cv_scheme}, seed={args.seed}, depth={args.depth}, lr={args.lr}, l2={args.l2}, iters={args.iterations}, threads={args.threads}")

    # 1. Load Train and Test Data
    log("[Step 1/5] Loading Train (70k) and Test (30k) Datasets...")
    with zipfile.ZipFile(args.train_zip) as z:
        train_rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(train_rows) == 70000, f"Expected 70,000 train rows, got {len(train_rows)}"

    with open(args.test_json, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]
    assert len(test_rows) == 30000, f"Expected 30,000 test rows, got {len(test_rows)}"

    y_train = np.array([int(r["condition"] == "used") for r in train_rows], dtype=np.int32)
    seller_train = np.array([str(r["seller_id"]) for r in train_rows])
    ids_train = np.array([str(r["id"]) for r in train_rows])
    ids_test = np.array([str(r["id"]) for r in test_rows])

    log(f"Loaded {len(train_rows)} train items (used={y_train.sum()}, new={(y_train == 0).sum()}) and {len(test_rows)} test items.")

    # 2. Extract Features
    log("[Step 2/5] Engineering Enriched Tabular Features & Rich Normalized Text...")
    t_feat = time.time()
    df_train_raw = extract_features_df(train_rows)
    df_test_raw = extract_features_df(test_rows)
    del train_rows, test_rows
    gc.collect()
    log(f"Feature extraction completed in {time.time() - t_feat:.2f}s. Raw shape: train={df_train_raw.shape}, test={df_test_raw.shape}")

    # 3. Setup Cross-Validation Splits
    log(f"[Step 3/5] Setting up 5-Fold {args.cv_scheme} Cross-Validation...")
    if args.cv_scheme == "seller_group":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        splits = list(cv.split(df_train_raw, y_train, seller_train))
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        splits = list(cv.split(df_train_raw, y_train))

    cat_cols = ["listing_type_id", "shipping_mode", "state", "category_id", "buying_mode", "currency_id", "status"]
    text_cols = ["clean_text"]

    oof_preds = np.zeros(len(df_train_raw), dtype=np.float32)
    test_preds_accum = np.zeros(len(df_test_raw), dtype=np.float32)

    fold_metrics = []

    # 4. Train 5 Folds
    log(f"[Step 4/5] Training 5 Folds CatBoostClassifier(depth={args.depth}, lr={args.lr}, l2={args.l2}, threads={args.threads})...")
    for fold, (trn_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        log("-" * 60)
        log(f"--- FOLD {fold + 1}/5 (Train: {len(trn_idx)}, Val: {len(val_idx)}) ---")

        # Fold-safe relative prices and frequencies
        df_tr = df_train_raw.iloc[trn_idx].copy()
        df_va = df_train_raw.iloc[val_idx].copy()
        df_te = df_test_raw.copy()

        # Relative price to category median
        cat_med_log = df_tr.groupby("category_id")["log_price"].median()
        global_med_log = float(df_tr["log_price"].median())
        df_tr["rel_price_cat"] = (df_tr["log_price"] - df_tr["category_id"].map(cat_med_log).fillna(global_med_log)).astype(np.float32)
        df_va["rel_price_cat"] = (df_va["log_price"] - df_va["category_id"].map(cat_med_log).fillna(global_med_log)).astype(np.float32)
        df_te["rel_price_cat"] = (df_te["log_price"] - df_te["category_id"].map(cat_med_log).fillna(global_med_log)).astype(np.float32)

        cat_med_raw = df_tr.groupby("category_id")["price"].median()
        global_med_raw = float(df_tr["price"].median())
        df_tr["price_ratio_cat"] = np.clip(df_tr["price"] / (df_tr["category_id"].map(cat_med_raw).fillna(global_med_raw) + 1.0), 0, 50).astype(np.float32)
        df_va["price_ratio_cat"] = np.clip(df_va["price"] / (df_va["category_id"].map(cat_med_raw).fillna(global_med_raw) + 1.0), 0, 50).astype(np.float32)
        df_te["price_ratio_cat"] = np.clip(df_te["price"] / (df_te["category_id"].map(cat_med_raw).fillna(global_med_raw) + 1.0), 0, 50).astype(np.float32)

        # Frequencies computed on train fold
        seller_counts = df_tr["seller_id"].value_counts()
        df_tr["freq_seller"] = np.log1p(df_tr["seller_id"].map(seller_counts).fillna(0)).astype(np.float32)
        df_va["freq_seller"] = np.log1p(df_va["seller_id"].map(seller_counts).fillna(0)).astype(np.float32)
        df_te["freq_seller"] = np.log1p(df_te["seller_id"].map(seller_counts).fillna(0)).astype(np.float32)

        cat_counts = df_tr["category_id"].value_counts()
        df_tr["freq_cat"] = np.log1p(df_tr["category_id"].map(cat_counts).fillna(0)).astype(np.float32)
        df_va["freq_cat"] = np.log1p(df_va["category_id"].map(cat_counts).fillna(0)).astype(np.float32)
        df_te["freq_cat"] = np.log1p(df_te["category_id"].map(cat_counts).fillna(0)).astype(np.float32)

        # Drop identifier and target columns before feeding to CatBoost
        drop_cols = ["id", "seller_id", "price", "base_price"]
        train_features = [c for c in df_tr.columns if c not in drop_cols]

        X_tr = df_tr[train_features]
        y_tr = y_train[trn_idx]
        X_va = df_va[train_features]
        y_va = y_train[val_idx]
        X_te = df_te[train_features]

        # Model configuration
        model = CatBoostClassifier(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.lr,
            l2_leaf_reg=args.l2,
            random_seed=args.seed + fold,
            thread_count=args.threads,
            loss_function="Logloss",
            eval_metric="Logloss",
            use_best_model=True,
            early_stopping_rounds=args.early_stopping,
            verbose=100,
            allow_writing_files=False,
        )

        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            text_features=text_cols,
            eval_set=(X_va, y_va),
        )

        best_it = int(model.best_iteration_)
        val_probs = model.predict_proba(X_va)[:, 1].astype(np.float32)
        oof_preds[val_idx] = val_probs

        test_probs_fold = model.predict_proba(X_te)[:, 1].astype(np.float32)
        test_preds_accum += test_probs_fold / len(splits)

        fold_acc = float(accuracy_score(y_va, (val_probs >= 0.5).astype(int)))
        fold_ll = float(log_loss(y_va, val_probs))
        fold_auc = float(roc_auc_score(y_va, val_probs))
        fold_time = time.time() - t_fold

        log(f"Fold {fold + 1} Finished in {fold_time:.1f}s | Best Iteration: {best_it}")
        log(f"Fold {fold + 1} Metrics -> Acc@0.5: {fold_acc:.5f} | LogLoss: {fold_ll:.5f} | ROC-AUC: {fold_auc:.5f}")

        fold_metrics.append({
            "fold": fold + 1,
            "best_iteration": best_it,
            "duration_seconds": fold_time,
            "accuracy_05": fold_acc,
            "logloss": fold_ll,
            "roc_auc": fold_auc,
        })

        del df_tr, df_va, df_te, X_tr, y_tr, X_va, y_va, X_te, model
        gc.collect()

    # 5. Full Out-of-Fold Evaluation and Artifact Generation
    log("[Step 5/5] Computing Full OOF Performance & Saving Test Probabilities...")
    oof_acc_05 = float(accuracy_score(y_train, (oof_preds >= 0.5).astype(int)))
    oof_ll = float(log_loss(y_train, oof_preds))
    oof_auc = float(roc_auc_score(y_train, oof_preds))

    # Optimal threshold search
    grid = np.arange(0.35, 0.651, 0.005)
    best_th = float(max(grid, key=lambda t: accuracy_score(y_train, (oof_preds >= t).astype(int))))
    best_acc = float(accuracy_score(y_train, (oof_preds >= best_th).astype(int)))

    log("=" * 80)
    log(f"=== FINAL 5-FOLD OOF RESULTS ===")
    log(f"OOF Accuracy @ 0.50: {oof_acc_05:.5f} ({oof_acc_05 * 100:.3f}%)")
    log(f"OOF Optimal Acc:     {best_acc:.5f} at threshold {best_th:.3f}")
    log(f"OOF LogLoss:         {oof_ll:.5f}")
    log(f"OOF ROC-AUC:         {oof_auc:.5f}")
    log("=" * 80)

    # Save OOF Predictions
    oof_df = pd.DataFrame({
        "id": ids_train,
        "seller_id": seller_train,
        "y_true": y_train,
        "condition": np.where(y_train == 1, "used", "new"),
        "pred_proba": oof_preds,
        "pred_condition_05": np.where(oof_preds >= 0.5, "used", "new"),
        "pred_condition_opt": np.where(oof_preds >= best_th, "used", "new"),
    })
    oof_path = out_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    log(f"Saved OOF predictions to {oof_path}")

    # Save Test Predictions (.npy and .csv)
    test_npy_path = out_dir / "test_probabilities.npy"
    np.save(test_npy_path, test_preds_accum)
    log(f"Saved test probabilities array to {test_npy_path} (shape: {test_preds_accum.shape})")

    test_df = pd.DataFrame({
        "ID": np.arange(1, len(test_preds_accum) + 1),
        "id": ids_test,
        "pred_proba": test_preds_accum,
        "predicted_condition": np.where(test_preds_accum >= 0.5, "used", "new"),
    })
    test_csv_path = out_dir / "test_probabilities.csv"
    test_df.to_csv(test_csv_path, index=False)
    log(f"Saved test probabilities CSV to {test_csv_path}")

    # Save Summary Report JSON
    total_duration = time.time() - t_start
    summary = {
        "architecture": "CatBoostClassifier",
        "depth": args.depth,
        "learning_rate": args.lr,
        "l2_leaf_reg": args.l2,
        "max_iterations": args.iterations,
        "early_stopping_rounds": args.early_stopping,
        "threads": args.threads,
        "cv_scheme": args.cv_scheme,
        "seed": args.seed,
        "total_duration_minutes": total_duration / 60.0,
        "oof_accuracy_05": oof_acc_05,
        "oof_accuracy_optimal": best_acc,
        "optimal_threshold": best_th,
        "oof_logloss": oof_ll,
        "oof_roc_auc": oof_auc,
        "folds": fold_metrics,
        "output_files": {
            "oof_csv": str(oof_path),
            "test_npy": str(test_npy_path),
            "test_csv": str(test_csv_path),
        },
    }
    report_path = out_dir / "metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved metrics summary to {report_path}")
    log(f"All operations completed in {total_duration / 60.0:.2f} minutes.")


if __name__ == "__main__":
    main()
