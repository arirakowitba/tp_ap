#!/usr/bin/env python3
"""Deep CatBoost Depth 8 Specialist (5-Fold Stratified, Leak-Free).

Author: Antigravity
Date: 2026-11-10

Uses the exact same leak-free features as Deep XGBoost Depth 9:
- 200 TruncatedSVD components over 150,000 TF-IDF n-grams (cached).
- Intrinsic listing metadata, price features, pictures, attributes, warranty.
- Lexical indicators.
- Out-of-fold category price statistics (fold-safe).
- ZERO seller_id features.
"""

import os, sys, time, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_deep_xgboost_depth9_20261110 import (
    extract_intrinsic_features,
    compute_leak_free_price_features,
    log
)

TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
OUT_DIR = ROOT / "results" / "deep_catboost_depth8_20261110"
SVD_CACHE = ROOT / "results" / "deep_xgboost_depth9_20261110" / "svd_200_features.npz"


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("=== DEEP CATBOOST DEPTH 8 SPECIALIST (5-FOLD STRATIFIED) ===")
    log("=" * 80)

    log("Loading raw JSONLines data...")
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
    # Target: 1 for new, 0 for used
    y_train = np.array([1 if r["condition"] == "new" else 0 for r in train_rows], dtype=np.int32)

    # 1. Base Intrinsic Features
    log("Extracting intrinsic metadata & lexical features...")
    all_rows = train_rows + test_rows
    df_base_all, cat_series_all, _ = extract_intrinsic_features(all_rows)

    df_base_tr = df_base_all.iloc[:n_train].reset_index(drop=True)
    df_base_te = df_base_all.iloc[n_train:].reset_index(drop=True)
    cat_series_tr = cat_series_all.iloc[:n_train].reset_index(drop=True)
    cat_series_te = cat_series_all.iloc[n_train:].reset_index(drop=True)

    del all_rows, df_base_all, cat_series_all
    gc.collect()

    # 2. SVD Features
    log(f"Loading cached SVD components from {SVD_CACHE}...")
    svd_data = np.load(SVD_CACHE)
    X_tr_svd = svd_data["X_tr_svd"]
    X_te_svd = svd_data["X_te_svd"]

    svd_cols = [f"svd_{i}" for i in range(X_tr_svd.shape[1])]
    df_svd_tr = pd.DataFrame(X_tr_svd, columns=svd_cols, dtype=np.float32)
    df_svd_te = pd.DataFrame(X_te_svd, columns=svd_cols, dtype=np.float32)
    del X_tr_svd, X_te_svd
    gc.collect()

    # 3. 5-Fold Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4242)

    oof_cat = np.zeros(n_train, dtype=np.float32)
    test_cat = np.zeros(n_test, dtype=np.float32)
    fold_metrics = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(df_base_tr, y_train)):
        t_fold0 = time.time()
        log(f"\n--- FOLD {fold+1}/5 ---")

        # Leak-free price statistics
        fold_trn_price = compute_leak_free_price_features(df_base_tr, cat_series_tr, trn_idx, trn_idx)
        fold_val_price = compute_leak_free_price_features(df_base_tr, cat_series_tr, trn_idx, val_idx)
        fold_test_price = compute_leak_free_price_features(
            pd.concat([df_base_tr.iloc[trn_idx], df_base_te], ignore_index=True),
            pd.concat([cat_series_tr.iloc[trn_idx], cat_series_te], ignore_index=True),
            np.arange(len(trn_idx)),
            np.arange(len(trn_idx), len(trn_idx) + n_test)
        ).reset_index(drop=True)

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

        clf = CatBoostClassifier(
            iterations=1800,
            learning_rate=0.04,
            depth=8,
            l2_leaf_reg=5.0,
            thread_count=14,
            random_seed=4242 + fold,
            eval_metric="Logloss",
            early_stopping_rounds=150,
            verbose=300
        )

        clf.fit(X_trn, y_train[trn_idx], eval_set=(X_val, y_train[val_idx]), use_best_model=True)

        val_probs = clf.predict_proba(X_val)[:, 1].astype(np.float32)
        oof_cat[val_idx] = val_probs
        test_cat += clf.predict_proba(X_test_fold)[:, 1].astype(np.float32) / 5.0

        val_preds = (val_probs >= 0.5).astype(int)
        acc = float(accuracy_score(y_train[val_idx], val_preds))
        auc = float(roc_auc_score(y_train[val_idx], val_probs))
        ll = float(log_loss(y_train[val_idx], np.clip(val_probs, 1e-7, 1.0 - 1e-7)))
        best_iter = int(clf.get_best_iteration())
        elapsed = time.time() - t_fold0

        log(f"Fold {fold+1} Finished ({elapsed:.1f}s) | Best Iter: {best_iter} | Acc: {acc:.5f} | AUC: {auc:.5f} | LogLoss: {ll:.5f}")
        fold_metrics.append({
            "fold": fold + 1,
            "accuracy": acc,
            "auc": auc,
            "logloss": ll,
            "best_iteration": best_iter,
            "elapsed_seconds": elapsed
        })

        del X_trn, X_val, X_test_fold, clf, fold_trn_price, fold_val_price, fold_test_price
        gc.collect()

    overall_acc = float(accuracy_score(y_train, (oof_cat >= 0.5).astype(int)))
    overall_auc = float(roc_auc_score(y_train, oof_cat))
    overall_ll = float(log_loss(y_train, np.clip(oof_cat, 1e-7, 1.0 - 1e-7)))
    total_time = time.time() - t_start

    log("\n" + "=" * 80)
    log(f"OVERALL CATBOOST DEPTH 8 METRICS:")
    log(f"  OOF Accuracy: {overall_acc:.5f} ({int(overall_acc * n_train)} / {n_train})")
    log(f"  OOF ROC-AUC:  {overall_auc:.5f}")
    log(f"  OOF LogLoss:  {overall_ll:.5f}")
    log(f"  Total Time:   {total_time:.1f}s ({total_time / 60.0:.2f} min)")
    log("=" * 80)

    # Save artifacts
    np.savez_compressed(
        OUT_DIR / "catboost_depth8_predictions.npz",
        oof_cat=oof_cat,
        test_cat=test_cat
    )

    pd.DataFrame({
        "ID": range(1, n_train + 1),
        "prob_new": oof_cat,
        "prob_used": 1.0 - oof_cat,
        "true_condition": np.where(y_train == 1, "new", "used"),
        "predicted_condition": np.where(oof_cat >= 0.5, "new", "used")
    }).to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    pd.DataFrame({
        "ID": range(1, n_test + 1),
        "prob_new": test_cat,
        "prob_used": 1.0 - test_cat,
        "predicted_condition": np.where(test_cat >= 0.5, "new", "used")
    }).to_csv(OUT_DIR / "test_probabilities.csv", index=False)

    summary = {
        "model": "Deep_CatBoost_Depth8_Specialist",
        "num_features": 348,
        "n_train": n_train,
        "n_test": n_test,
        "overall_metrics": {
            "accuracy": overall_acc,
            "hits": int(overall_acc * n_train),
            "auc": overall_auc,
            "logloss": overall_ll,
            "total_time_seconds": total_time
        },
        "fold_metrics": fold_metrics
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"Saved all artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
