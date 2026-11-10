#!/usr/bin/env python3
"""High-Capacity LightGBM 127-Leaves Specialist.

Author: Antigravity
Date: 2026-11-10

Specifications:
1. Data:
   - Train: data/ap_2026_q_2/raw/train_data.jsonlines (70,000 items)
   - Test: data/ap_2026_q_2/raw/test_data.jsonlines (30,000 items)
2. Features (STRICTLY LEAK-FREE, ZERO seller_id features):
   - 200 TruncatedSVD components over 150,000 TF-IDF word and character n-grams.
   - Price features: log(price+1), category price median, category price std, relative price to median (price - med) / (med + 1), category price z-score.
   - Lexical indicators: used cues ("usad", "reacond", "buen estado", "impecable estado", "repuesto", "despiece", "partes", "detalles", "falla", "antigu", "retro", "vintage") and new cues ("nuevo", "nueva", "nuevas", "nuevos", "blister", "sellad", "cerrad", "estrenar", "0km", "0 km", "original", "garantia", "caja sellada").
   - Listing metadata: has_warranty, free_shipping, listing_type_id flags (gold, bronze, free), title length, word count.
   - Native categorical: category_id, listing_type_id.
3. Model Configuration:
   - LGBMClassifier(n_estimators=2500, learning_rate=0.02, num_leaves=127, min_child_samples=30, subsample=0.75, colsample_bytree=0.65, random_state=4242, n_jobs=16, verbose=-1).
4. Validation:
   - 5-Fold Stratified K-Fold cross-validation across 70,000 train items.
   - Compute OOF accuracy, ROC AUC, and Log Loss.
   - Generate test set continuous probabilities (mean over 5 folds).
5. Output Artifacts:
   - results/deep_lgbm_leaves127_20261110/lgbm_leaves127_predictions.npz (keys: oof_lgb, test_lgb)
   - results/deep_lgbm_leaves127_20261110/summary.json
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)


def lst(x):
    return x if isinstance(x, list) else []


USED_CUES = [
    "usad", "reacond", "buen estado", "impecable estado", "repuesto",
    "despiece", "partes", "detalles", "falla", "antigu", "retro", "vintage"
]

NEW_CUES = [
    "nuevo", "nueva", "nuevas", "nuevos", "blister", "sellad", "cerrad",
    "estrenar", "0km", "0 km", "original", "garantia", "caja sellada"
]


def extract_item_metadata_and_corpus(rows):
    """Extract corpus, prices, lexical cues, and metadata without any seller_id features."""
    t0 = time.time()
    corpus = []
    
    price = np.array([float(r.get("price") or 0.0) for r in rows], dtype=np.float32)
    log_price = np.log1p(np.maximum(0.0, price)).astype(np.float32)

    titles = [str(r.get("title") or "") for r in rows]
    title_lens = np.array([float(len(t)) for t in titles], dtype=np.float32)
    word_counts = np.array([float(len(t.split())) for t in titles], dtype=np.float32)

    has_warranty = np.array([1.0 if bool(r.get("warranty")) else 0.0 for r in rows], dtype=np.float32)
    free_shipping = np.array([1.0 if bool((r.get("shipping") or {}).get("free_shipping")) else 0.0 for r in rows], dtype=np.float32)

    listing_types = [str(r.get("listing_type_id") or "").lower() for r in rows]
    is_gold = np.array([1.0 if "gold" in lt else 0.0 for lt in listing_types], dtype=np.float32)
    is_bronze = np.array([1.0 if "bronze" in lt else 0.0 for lt in listing_types], dtype=np.float32)
    is_free = np.array([1.0 if "free" in lt else 0.0 for lt in listing_types], dtype=np.float32)

    # Lexical cues on lowercased titles
    lex_feats = {}
    used_total = np.zeros(len(rows), dtype=np.float32)
    new_total = np.zeros(len(rows), dtype=np.float32)

    for cue in USED_CUES:
        cue_col = f"cue_u_{cue.replace(' ', '_')}"
        arr = np.array([1.0 if cue in t.lower() else 0.0 for t in titles], dtype=np.float32)
        lex_feats[cue_col] = arr
        used_total += arr

    for cue in NEW_CUES:
        cue_col = f"cue_n_{cue.replace(' ', '_')}"
        arr = np.array([1.0 if cue in t.lower() else 0.0 for t in titles], dtype=np.float32)
        lex_feats[cue_col] = arr
        new_total += arr

    lex_feats["used_cue_count"] = used_total
    lex_feats["new_cue_count"] = new_total
    lex_feats["cue_net_diff"] = used_total - new_total

    # Composite text corpus for 150k TF-IDF
    for r in rows:
        t = str(r.get("title") or "")
        w = str(r.get("warranty") or "")
        attrs = " ".join(str(a.get("value_name") or a.get("name") or "") for a in lst(r.get("attributes")) if isinstance(a, dict))
        tags = " ".join(str(tg) for tg in lst(r.get("tags")) if tg)
        full_text = f"{t} WARRANTY {w} ATTR {attrs} TAGS {tags}".strip()
        corpus.append(full_text)

    df_meta = pd.DataFrame({
        "price": price,
        "log_price": log_price,
        "title_length": title_lens,
        "word_count": word_counts,
        "has_warranty": has_warranty,
        "free_shipping": free_shipping,
        "is_gold": is_gold,
        "is_bronze": is_bronze,
        "is_free": is_free,
        "category_id": [str(r.get("category_id") or "MISSING") for r in rows],
        "listing_type_id": [str(r.get("listing_type_id") or "MISSING") for r in rows],
        **lex_feats
    })

    log(f"Extracted metadata & corpus for {len(rows)} records in {time.time()-t0:.2f}s")
    return corpus, df_meta


def compute_fold_price_features(df_train_fold, df_target, global_p_med, global_p_std, global_p_mean):
    """Compute strictly fold-safe category price aggregates and relative price features."""
    cat_grp = df_train_fold.groupby("category_id")["price"]
    c_med = cat_grp.median()
    c_mean = cat_grp.mean()
    c_std = cat_grp.std().fillna(global_p_std)

    med_vals = df_target["category_id"].map(c_med).fillna(global_p_med).values
    std_vals = df_target["category_id"].map(c_std).fillna(global_p_std).values
    mean_vals = df_target["category_id"].map(c_mean).fillna(global_p_mean).values

    prices = df_target["price"].values
    rel_price_med = (prices - med_vals) / (med_vals + 1.0)
    cat_price_z = (prices - mean_vals) / (std_vals + 1e-4)

    return pd.DataFrame({
        "cat_price_median": med_vals.astype(np.float32),
        "cat_price_std": std_vals.astype(np.float32),
        "rel_price_to_median": rel_price_med.astype(np.float32),
        "cat_price_zscore": cat_price_z.astype(np.float32)
    }, index=df_target.index)


def main():
    parser = argparse.ArgumentParser(description="Train High-Capacity LightGBM 127-Leaves Specialist")
    parser.add_argument("--train_path", default="data/ap_2026_q_2/raw/train_data.jsonlines", help="Train jsonlines")
    parser.add_argument("--test_path", default="data/ap_2026_q_2/raw/test_data.jsonlines", help="Test jsonlines")
    parser.add_argument("--out_dir", default="results/deep_lgbm_leaves127_20261110", help="Output directory")
    parser.add_argument("--seed", type=int, default=4242, help="Random seed")
    parser.add_argument("--svd_dim", type=int, default=200, help="SVD components")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log("=" * 80)
    log("=== HIGH-CAPACITY LIGHTGBM 127-LEAVES SPECIALIST TRAINING ENGINE ===")
    log("=" * 80)

    # 1. Load Data
    log(f"Step 1: Loading raw datasets: {args.train_path} and {args.test_path}...")
    train_rows = []
    with open(args.train_path, "r", encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))
    test_rows = []
    with open(args.test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))

    n_train = len(train_rows)
    n_test = len(test_rows)
    assert n_train == 70000, f"Expected 70,000 train items, got {n_train}"
    assert n_test == 30000, f"Expected 30,000 test items, got {n_test}"

    y_train = np.array([int(r["condition"] == "used") for r in train_rows], dtype=np.int32)
    log(f"Loaded Train: {n_train} rows (Used rate: {y_train.mean():.4f}), Test: {n_test} rows.")

    # 2. Extract Metadata & Text Corpus
    log("Step 2: Extracting metadata, price features, and lexical cues (ZERO seller_id)...")
    train_corpus, df_meta_tr = extract_item_metadata_and_corpus(train_rows)
    test_corpus, df_meta_te = extract_item_metadata_and_corpus(test_rows)
    del train_rows, test_rows
    gc.collect()

    # 3. 150k TF-IDF & 200 TruncatedSVD Components
    log("Step 3: Extracting 150,000 TF-IDF n-grams (60k word + 90k char) & 200 SVD components...")
    t_svd_start = time.time()
    vec_word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=60000,
        sublinear_tf=True,
        dtype=np.float32
    )
    vec_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_features=90000,
        sublinear_tf=True,
        dtype=np.float32
    )

    log("  Fitting Word TF-IDF on 70k train...")
    X_tr_w = vec_word.fit_transform(train_corpus)
    X_te_w = vec_word.transform(test_corpus)

    log("  Fitting Char-WB TF-IDF on 70k train...")
    X_tr_c = vec_char.fit_transform(train_corpus)
    X_te_c = vec_char.transform(test_corpus)

    X_tr_tfidf = sparse.hstack([X_tr_w, X_tr_c], format="csr")
    X_te_tfidf = sparse.hstack([X_te_w, X_te_c], format="csr")
    log(f"  Combined TF-IDF Shape: Train {X_tr_tfidf.shape}, Test {X_te_tfidf.shape}")

    del X_tr_w, X_te_w, X_tr_c, X_te_c, train_corpus, test_corpus
    gc.collect()

    log(f"  Fitting TruncatedSVD({args.svd_dim}) on 70k train TF-IDF...")
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)
    X_te_svd = svd.transform(X_te_tfidf)
    log(f"  SVD({args.svd_dim}) completed in {time.time()-t_svd_start:.1f}s | Explained Var: {svd.explained_variance_ratio_.sum():.4f}")

    del X_tr_tfidf, X_te_tfidf
    gc.collect()

    svd_cols = [f"svd_{i:03d}" for i in range(args.svd_dim)]
    df_svd_tr = pd.DataFrame(X_tr_svd, columns=svd_cols, dtype=np.float32)
    df_svd_te = pd.DataFrame(X_te_svd, columns=svd_cols, dtype=np.float32)
    del X_tr_svd, X_te_svd
    gc.collect()

    # 4. Cross-Validation Setup: 5-Fold Stratified K-Fold
    log("Step 4: Setting up 5-Fold Stratified K-Fold CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    oof_lgb = np.zeros(n_train, dtype=np.float32)
    test_fold_probs = np.zeros((5, n_test), dtype=np.float32)

    cat_cols = ["category_id", "listing_type_id"]
    for c in cat_cols:
        df_meta_tr[c] = df_meta_tr[c].astype("category")
        df_meta_te[c] = df_meta_te[c].astype("category")

    lgb_params = {
        "n_estimators": 2500,
        "learning_rate": 0.02,
        "num_leaves": 127,
        "min_child_samples": 30,
        "subsample": 0.75,
        "colsample_bytree": 0.65,
        "random_state": args.seed,
        "n_jobs": 16,
        "verbose": -1,
        "objective": "binary",
        "metric": "binary_logloss"
    }

    fold_scores = []
    global_p_med = float(df_meta_tr["price"].median())
    global_p_mean = float(df_meta_tr["price"].mean())
    global_p_std = float(df_meta_tr["price"].std()) or 1.0

    # Base feature columns excluding raw price and categoricals (which are handled separately)
    base_numeric_cols = [c for c in df_meta_tr.columns if c not in ["price", "category_id", "listing_type_id"]]

    for fold, (trn_idx, val_idx) in enumerate(skf.split(df_meta_tr, y_train)):
        t_f0 = time.time()
        log(f"\n--- FOLD {fold+1}/5 ---")
        log(f"Train size: {len(trn_idx)}, Val size: {len(val_idx)}")

        # A. Fold-safe price features
        trn_meta = df_meta_tr.iloc[trn_idx]
        val_meta = df_meta_tr.iloc[val_idx]

        price_trn = compute_fold_price_features(trn_meta, trn_meta, global_p_med, global_p_std, global_p_mean)
        price_val = compute_fold_price_features(trn_meta, val_meta, global_p_med, global_p_std, global_p_mean)
        price_te = compute_fold_price_features(trn_meta, df_meta_te, global_p_med, global_p_std, global_p_mean)

        # B. Assemble Full Feature DataFrames
        X_trn = pd.concat([
            trn_meta[base_numeric_cols].reset_index(drop=True),
            price_trn.reset_index(drop=True),
            df_svd_tr.iloc[trn_idx].reset_index(drop=True),
            trn_meta[cat_cols].reset_index(drop=True)
        ], axis=1)

        X_val = pd.concat([
            val_meta[base_numeric_cols].reset_index(drop=True),
            price_val.reset_index(drop=True),
            df_svd_tr.iloc[val_idx].reset_index(drop=True),
            val_meta[cat_cols].reset_index(drop=True)
        ], axis=1)

        X_te = pd.concat([
            df_meta_te[base_numeric_cols].reset_index(drop=True),
            price_te.reset_index(drop=True),
            df_svd_te.reset_index(drop=True),
            df_meta_te[cat_cols].reset_index(drop=True)
        ], axis=1)

        if fold == 0:
            log(f"Total features: {X_trn.shape[1]} (200 SVD + {X_trn.shape[1]-200} Price/Lexical/Metadata)")
            feature_names = list(X_trn.columns)

        # C. Train LGBMClassifier
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            X_trn, y_train[trn_idx],
            eval_set=[(X_val, y_train[val_idx])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=500)
            ]
        )

        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = val_probs

        acc_f = float(accuracy_score(y_train[val_idx], (val_probs >= 0.5).astype(int)))
        ll_f = float(log_loss(y_train[val_idx], np.clip(val_probs, 1e-7, 1.0 - 1e-7)))
        auc_f = float(roc_auc_score(y_train[val_idx], val_probs))
        best_iter = int(clf.best_iteration_ if hasattr(clf, "best_iteration_") else lgb_params["n_estimators"])

        log(f"Fold {fold+1} Finished ({time.time()-t_f0:.1f}s) | Best Iter: {best_iter} | Acc: {acc_f:.5f} | LogLoss: {ll_f:.5f} | AUC: {auc_f:.5f}")
        fold_scores.append({
            "fold": int(fold + 1),
            "accuracy": acc_f,
            "logloss": ll_f,
            "auc": auc_f,
            "best_iteration": best_iter
        })

        # D. Predict on Test
        test_fold_probs[fold] = clf.predict_proba(X_te)[:, 1]

    # 5. Out-of-Fold Evaluation
    log("\n" + "=" * 80)
    log("=== OVERALL 5-FOLD OUT-OF-FOLD EVALUATION ===")
    log("=" * 80)

    oof_clipped = np.clip(oof_lgb, 1e-7, 1.0 - 1e-7)
    overall_acc = float(accuracy_score(y_train, (oof_lgb >= 0.5).astype(int)))
    overall_ll = float(log_loss(y_train, oof_clipped))
    overall_auc = float(roc_auc_score(y_train, oof_lgb))

    log(f"OOF Accuracy (t=0.50): {overall_acc:.5f} ({int(((oof_lgb >= 0.5).astype(int) == y_train).sum())}/{n_train})")
    log(f"OOF LogLoss: {overall_ll:.5f}")
    log(f"OOF ROC-AUC: {overall_auc:.5f}")

    # 6. Test Set Probabilities (Mean over 5 folds)
    test_lgb = np.mean(test_fold_probs, axis=0).astype(np.float32)
    log(f"Test Predictions: Mean={test_lgb.mean():.4f}, Min={test_lgb.min():.4f}, Max={test_lgb.max():.4f}")
    n_used_test = int((test_lgb >= 0.5).sum())
    log(f"Test Predicted Used (t=0.5): {n_used_test}/{n_test} ({n_used_test/n_test:.2%})")

    # 7. Save Output Artifacts
    npz_path = out_dir / "lgbm_leaves127_predictions.npz"
    np.savez_compressed(npz_path, oof_lgb=oof_lgb, test_lgb=test_lgb)
    log(f"Saved predictions npz to {npz_path}")

    runtime_sec = round(time.time() - t_start, 2)
    summary = {
        "model": "High-Capacity LightGBM 127-Leaves Specialist",
        "n_train": n_train,
        "n_test": n_test,
        "features_count": len(feature_names),
        "svd_components": args.svd_dim,
        "lgbm_params": lgb_params,
        "cv_folds": fold_scores,
        "oof_accuracy": overall_acc,
        "oof_roc_auc": overall_auc,
        "oof_logloss": overall_ll,
        "test_used_count": n_used_test,
        "runtime_seconds": runtime_sec,
        "artifacts": {
            "predictions_npz": str(npz_path),
            "summary_json": str(out_dir / "summary.json")
        }
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary JSON to {summary_path}")
    log(f"Pipeline completed in {runtime_sec:.1f}s")


if __name__ == "__main__":
    main()
