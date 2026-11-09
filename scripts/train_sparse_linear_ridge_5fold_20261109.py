#!/usr/bin/env python3
"""Engine A: 5-Fold Stratified Sparse Linear Logistic / Ridge Classifier.

Author: Antigravity
Date: 2026-11-09

Specifications:
- Rich text corpus: title + warranty + attributes (name & value) + tags.
- Word TF-IDF: ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True.
- Char TF-IDF: analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_features=70000, sublinear_tf=True.
- Standardized deep tabular features: log_price, price_diff, rel_price_cat, seller frequency, category frequency, dates.
- Model: LogisticRegression(C=1.5, solver='saga', max_iter=250, random_state=20261109).
- 5-Fold StratifiedKFold on condition.
- Outputs:
  - results/sparse_linear_5fold/oof_predictions.csv
  - results/sparse_linear_5fold/test_probabilities.npy
  - results/sparse_linear_5fold/test_probabilities.csv
  - results/sparse_linear_5fold/metrics.json
"""

import argparse
import gc
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)


def lst(x):
    return x if isinstance(x, list) else []


def extract_metadata(rows):
    """Extract raw text corpus and intrinsic tabular features."""
    t0 = time.time()
    corpus = []
    
    price = np.array([float(r.get("price") or 0.0) for r in rows], dtype=np.float32)
    base_price = np.array([float(r.get("base_price") or 0.0) for r in rows], dtype=np.float32)
    log_price = np.log1p(np.maximum(0.0, price)).astype(np.float32)
    price_diff = (base_price - price).astype(np.float32)
    
    ref = pd.Timestamp("2013-01-01", tz="UTC")
    dates_created = pd.to_datetime([r.get("date_created") for r in rows], utc=True, errors="coerce")
    dates_updated = pd.to_datetime([r.get("last_updated") for r in rows], utc=True, errors="coerce")
    st = pd.to_datetime([r.get("start_time") for r in rows], unit="ms", utc=True, errors="coerce")
    sp = pd.to_datetime([r.get("stop_time") for r in rows], unit="ms", utc=True, errors="coerce")

    created_days = np.nan_to_num((dates_created - ref).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    dur_days = np.nan_to_num((sp - st).total_seconds() / 86400.0, nan=0.0).astype(np.float32)
    upd_minus_created = np.nan_to_num((dates_updated - dates_created).total_seconds() / 86400.0, nan=0.0).astype(np.float32)

    cat_ids = [str(r.get("category_id") or "MISSING") for r in rows]
    seller_ids = [str(r.get("seller_id") or "MISSING") for r in rows]

    for r in rows:
        title = str(r.get("title") or "")
        warranty = str(r.get("warranty") or "")
        attrs = r.get("attributes") or []
        attr_parts = []
        for a in attrs:
            if isinstance(a, dict):
                n = str(a.get("name") or "").strip()
                v = str(a.get("value_name") or "").strip()
                if n or v:
                    attr_parts.append(f"{n} {v}".strip())
        attr_str = " ".join(attr_parts)
        tags_str = " ".join(str(tg) for tg in lst(r.get("tags")) if tg)
        full_text = f"{title} WARRANTY {warranty} ATTR {attr_str} TAGS {tags_str}".strip()
        corpus.append(full_text)

    df_meta = pd.DataFrame({
        "log_price": log_price,
        "price_diff": price_diff,
        "created_days": created_days,
        "dur_days": dur_days,
        "upd_minus_created": upd_minus_created,
        "category_id": cat_ids,
        "seller_id": seller_ids
    })

    log(f"Extracted metadata and corpus for {len(rows)} records in {time.time()-t0:.2f}s")
    return corpus, df_meta


def compute_tabular_features(df_trn, df_target, trn_cat_med, trn_cat_counts, trn_seller_counts, global_lp_med):
    """Compute relative prices and frequencies strictly using training partition mappings."""
    rel_price_cat = df_target["log_price"].values - df_target["category_id"].map(trn_cat_med).fillna(global_lp_med).values
    seller_freq = np.log1p(df_target["seller_id"].map(trn_seller_counts).fillna(0.0).values)
    cat_freq = np.log1p(df_target["category_id"].map(trn_cat_counts).fillna(0.0).values)

    tab_mat = np.column_stack([
        df_target["log_price"].values,
        df_target["price_diff"].values,
        rel_price_cat,
        seller_freq,
        cat_freq,
        df_target["created_days"].values,
        df_target["dur_days"].values,
        df_target["upd_minus_created"].values
    ]).astype(np.float32)

    return tab_mat


def main():
    parser = argparse.ArgumentParser(description="Train Engine A: 5-Fold Sparse Linear Logistic Classifier")
    parser.add_argument("--train_zip", default="data/train_data.zip", help="Path to train_data.zip")
    parser.add_argument("--test_data", default="data/ap_2026_q_2/raw/test_data.jsonlines", help="Path to test_data.jsonlines")
    parser.add_argument("--out_dir", default="results/sparse_linear_5fold", help="Output directory")
    parser.add_argument("--seed", type=int, default=20261109, help="Random seed")
    parser.add_argument("--c", type=float, default=1.5, help="Inverse regularization parameter C")
    parser.add_argument("--solver", default="lbfgs", choices=["saga", "lbfgs"], help="Logistic regression solver")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log("=" * 80)
    log("=== ENGINE A: 5-FOLD STRATIFIED SPARSE LINEAR LOGISTIC CLASSIFIER ===")
    log("=" * 80)

    # 1. Load Data
    log(f"Loading raw datasets: {args.train_zip} and {args.test_data}...")
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
    train_sellers = [str(r["seller_id"]) for r in train_rows]
    test_sellers = [str(r["seller_id"]) for r in test_rows]
    global_prior = float(y_train.mean())
    log(f"Loaded Train: {n_train} rows (Used rate: {global_prior:.4f}), Test: {n_test} rows.")

    # 2. Extract Text Corpus and Tabular Metadata
    log("Extracting text corpus and metadata...")
    train_corpus, df_meta_tr = extract_metadata(train_rows)
    test_corpus, df_meta_te = extract_metadata(test_rows)
    del train_rows, test_rows
    gc.collect()

    # 3. Fit Word and Char TF-IDF Vectorizers on Full Train Corpus
    log("Fitting TF-IDF Feature Extractors on 70k Train Corpus...")
    t_tfidf = time.time()
    vec_word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32
    )
    vec_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_features=70000,
        sublinear_tf=True,
        dtype=np.float32
    )

    log("  Transforming Word n-grams (1, 2)...")
    X_tr_w = vec_word.fit_transform(train_corpus)
    X_te_w = vec_word.transform(test_corpus)

    log("  Transforming Char-WB n-grams (3, 5)...")
    X_tr_c = vec_char.fit_transform(train_corpus)
    X_te_c = vec_char.transform(test_corpus)

    X_tr_tfidf = sparse.hstack([X_tr_w, X_tr_c], format="csr")
    X_te_tfidf = sparse.hstack([X_te_w, X_te_c], format="csr")
    log(f"Combined TF-IDF Shape: Train {X_tr_tfidf.shape}, Test {X_te_tfidf.shape} (Time: {time.time()-t_tfidf:.1f}s)")

    del X_tr_w, X_te_w, X_tr_c, X_te_c, train_corpus, test_corpus
    gc.collect()

    # 4. Cross-Validation: 5-Fold Stratified
    log("Setting up 5-Fold Stratified CV on condition...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    oof_predictions = np.zeros(n_train, dtype=np.float32)
    test_fold_probabilities = np.zeros((5, n_test), dtype=np.float32)
    fold_scores = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr_tfidf, y_train)):
        t_f0 = time.time()
        log(f"\n--- FOLD {fold+1}/5 ---")
        log(f"Train partition: {len(trn_idx)}, Val partition: {len(val_idx)}")

        # A. Compute fold-safe tabular statistics strictly on training fold
        trn_meta = df_meta_tr.iloc[trn_idx]
        val_meta = df_meta_tr.iloc[val_idx]

        trn_cat_med = trn_meta.groupby("category_id")["log_price"].median()
        global_lp_med = float(trn_meta["log_price"].median())
        trn_cat_counts = trn_meta["category_id"].value_counts()
        trn_seller_counts = trn_meta["seller_id"].value_counts()

        tab_trn = compute_tabular_features(trn_meta, trn_meta, trn_cat_med, trn_cat_counts, trn_seller_counts, global_lp_med)
        tab_val = compute_tabular_features(trn_meta, val_meta, trn_cat_med, trn_cat_counts, trn_seller_counts, global_lp_med)
        tab_te = compute_tabular_features(trn_meta, df_meta_te, trn_cat_med, trn_cat_counts, trn_seller_counts, global_lp_med)

        # B. Standardize tabular features using StandardScaler fit on trn only
        scaler = StandardScaler()
        tab_trn_scaled = sparse.csr_matrix(scaler.fit_transform(tab_trn))
        tab_val_scaled = sparse.csr_matrix(scaler.transform(tab_val))
        tab_te_scaled = sparse.csr_matrix(scaler.transform(tab_te))

        # C. Stack sparse TF-IDF with standardized tabular features
        X_trn = sparse.hstack([X_tr_tfidf[trn_idx], tab_trn_scaled], format="csr")
        X_val = sparse.hstack([X_tr_tfidf[val_idx], tab_val_scaled], format="csr")
        X_te = sparse.hstack([X_te_tfidf, tab_te_scaled], format="csr")

        if fold == 0:
            log(f"Full feature dimensionality (sparse): {X_trn.shape[1]} (120,000 TF-IDF + 8 Tabular)")

        # D. Train Linear Logistic Regression
        log(f"  Fitting LogisticRegression(C={args.c}, solver='{args.solver}', max_iter=250)...")
        clf = LogisticRegression(
            C=args.c,
            solver=args.solver,
            max_iter=250,
            random_state=args.seed + fold,
            tol=1e-4
        )
        clf.fit(X_trn, y_train[trn_idx])

        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_probs

        acc_f = float(accuracy_score(y_train[val_idx], (val_probs >= 0.5).astype(int)))
        ll_f = float(log_loss(y_train[val_idx], np.clip(val_probs, 1e-7, 1.0 - 1e-7)))
        auc_f = float(roc_auc_score(y_train[val_idx], val_probs))
        n_iter_f = int(clf.n_iter_[0])

        log(f"Fold {fold+1} Finished ({time.time()-t_f0:.1f}s) | Iterations: {n_iter_f} | Acc: {acc_f:.5f} | LogLoss: {ll_f:.5f} | AUC: {auc_f:.5f}")
        fold_scores.append({
            "fold": int(fold + 1),
            "accuracy": acc_f,
            "logloss": ll_f,
            "auc": auc_f,
            "iterations": n_iter_f
        })

        # E. Predict on Test Set
        test_pred_fold = clf.predict_proba(X_te)[:, 1]
        test_fold_probabilities[fold] = test_pred_fold

    # 5. Out-of-Fold Evaluation
    log("\n" + "=" * 80)
    log("=== OVERALL 5-FOLD OUT-OF-FOLD LINEAR MODEL BENCHMARK RESULTS ===")
    log("=" * 80)

    oof_preds_clipped = np.clip(oof_predictions, 1e-7, 1.0 - 1e-7)
    overall_acc = float(accuracy_score(y_train, (oof_predictions >= 0.5).astype(int)))
    overall_ll = float(log_loss(y_train, oof_preds_clipped))
    overall_auc = float(roc_auc_score(y_train, oof_predictions))

    # Optimal threshold search
    grid = np.linspace(0.40, 0.60, 201)
    best_thresh = 0.50
    best_acc = overall_acc
    for t in grid:
        acc_t = float(accuracy_score(y_train, (oof_predictions >= t).astype(int)))
        if acc_t > best_acc:
            best_acc = acc_t
            best_thresh = float(t)

    log(f"OOF Accuracy (t=0.50): {overall_acc:.5f} ({int(((oof_predictions >= 0.5).astype(int) == y_train).sum())}/{n_train})")
    log(f"OOF Accuracy (optimal t={best_thresh:.4f}): {best_acc:.5f} ({int(((oof_predictions >= best_thresh).astype(int) == y_train).sum())}/{n_train})")
    log(f"OOF LogLoss: {overall_ll:.5f}")
    log(f"OOF ROC-AUC: {overall_auc:.5f}")

    # 6. Save Out-of-Fold Predictions
    oof_df = pd.DataFrame({
        "item_id": train_ids,
        "seller_id": train_sellers,
        "y_true": y_train,
        "prob_used": oof_predictions,
        "pred_condition": np.where(oof_predictions >= best_thresh, "used", "new")
    })
    oof_path = out_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    log(f"Saved OOF predictions to {oof_path}")

    # 7. Test Predictions Ensemble and Export
    log("Generating 5-fold ensemble test probabilities...")
    test_prob_mean = np.mean(test_fold_probabilities, axis=0)

    test_npy_path = out_dir / "test_probabilities.npy"
    np.save(test_npy_path, test_prob_mean)
    log(f"Saved test probabilities numpy array to {test_npy_path}")

    test_df = pd.DataFrame({
        "ID": np.arange(1, n_test + 1),
        "item_id": test_ids,
        "seller_id": test_sellers,
        "prob_used": test_prob_mean,
        "condition": np.where(test_prob_mean >= best_thresh, "used", "new")
    })
    test_csv_path = out_dir / "test_probabilities.csv"
    test_df.to_csv(test_csv_path, index=False)
    log(f"Saved test probabilities CSV to {test_csv_path}")

    n_used_test = int((test_df["condition"] == "used").sum())
    n_new_test = int((test_df["condition"] == "new").sum())
    log(f"Test Class Distribution: Used = {n_used_test} ({n_used_test/n_test:.2%}), New = {n_new_test} ({n_new_test/n_test:.2%})")

    # 8. Save Metrics JSON
    metrics = {
        "model": "Engine A: 5-Fold Stratified Sparse Linear Logistic Classifier",
        "solver": args.solver,
        "C": args.c,
        "n_train": n_train,
        "n_test": n_test,
        "total_sparse_features": int(X_trn.shape[1]),
        "tf_idf_word_features": 50000,
        "tf_idf_char_features": 70000,
        "tabular_features": 8,
        "cv_folds": fold_scores,
        "oof_accuracy_t050": overall_acc,
        "oof_optimal_threshold": best_thresh,
        "oof_accuracy_optimal": best_acc,
        "oof_logloss": overall_ll,
        "oof_roc_auc": overall_auc,
        "test_used_count": n_used_test,
        "test_new_count": n_new_test,
        "elapsed_seconds": round(time.time() - t_start, 2)
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved metrics to {metrics_path}")
    log(f"Pipeline finished in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
