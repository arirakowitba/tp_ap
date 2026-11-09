#!/usr/bin/env python3
"""Engine C: 5-Fold Stratified Multi-Layer Perceptron (MLP) Neural Diversity Model.

Full 70,000 train rows with 5-Fold Stratified cross-validation.
Architecture: MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu',
                            solver='adam', alpha=1e-4, batch_size=256,
                            learning_rate_init=1e-3, early_stopping=True,
                            max_iter=80, random_state=20261109)

Features:
- Exact 207 features engineered for XGBoost (from results/xgboost_diversity_5fold/svd_64_features.npz
  and nested_nb_probabilities.npz, plus base tabular features and fold aggregations).
- StandardScaler strictly fit on train fold, and transformed on val and test.

Outputs:
- results/mlp_neural_5fold/oof_predictions.csv
- results/mlp_neural_5fold/test_probabilities.npy
- results/mlp_neural_5fold/test_probabilities.csv
- results/mlp_neural_5fold/metrics.json
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
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_xgboost_diversity_5fold_20261109 import (
    extract_base_features,
    compute_fold_aggregations,
)

DEFAULT_TRAIN_ZIP = ROOT / "data" / "train_data.zip"
DEFAULT_TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
XGB_DIR = ROOT / "results" / "xgboost_diversity_5fold"
DEFAULT_OUT_DIR = ROOT / "results" / "mlp_neural_5fold"


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{t_str} UTC] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train 5-Fold Stratified MLP Neural Diversity Model")
    parser.add_argument("--train-zip", default=str(DEFAULT_TRAIN_ZIP), help="Path to train_data.zip")
    parser.add_argument("--test-data", default=str(DEFAULT_TEST_JSON), help="Path to test_data.jsonlines")
    parser.add_argument("--xgb-dir", default=str(XGB_DIR), help="Directory containing cached SVD and NB features")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--seed", type=int, default=20261109, help="Random seed")
    parser.add_argument("--max-iter", type=int, default=80, help="Max epochs for MLP")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 penalty alpha")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xgb_dir = Path(args.xgb_dir)

    t_start = time.time()
    log("=" * 80)
    log("=== ENGINE C: 5-FOLD STRATIFIED MLP NEURAL DIVERSITY MODEL ===")
    log("=" * 80)
    log(f"Parameters: seed={args.seed}, max_iter={args.max_iter}, batch_size={args.batch_size}, lr={args.lr}, alpha={args.alpha}")

    # 1. Load Raw Train and Test Datasets
    log("[Step 1/5] Loading Train (70k) and Test (30k) raw datasets...")
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
    log("[Step 2/5] Extracting intrinsic tabular & regex features (127 features)...")
    all_rows = train_rows + test_rows
    del train_rows, test_rows
    gc.collect()

    df_base_all, cat_df_all, _ = extract_base_features(all_rows)
    del all_rows
    gc.collect()

    df_base_tr = df_base_all.iloc[:n_train].reset_index(drop=True)
    df_base_te = df_base_all.iloc[n_train:].reset_index(drop=True)
    cat_df_tr = cat_df_all.iloc[:n_train].reset_index(drop=True)
    cat_df_te = cat_df_all.iloc[n_train:].reset_index(drop=True)
    del df_base_all, cat_df_all
    gc.collect()

    # 3. Load Cached SVD 64 and Nested NB features
    log("[Step 3/5] Loading cached 64-dim SVD and Nested NB features from XGBoost artifacts...")
    svd_cache_path = xgb_dir / "svd_64_features.npz"
    assert svd_cache_path.exists(), f"Missing SVD cache at {svd_cache_path}"
    svd_data = np.load(svd_cache_path)
    X_tr_svd = svd_data["X_tr_svd"]
    X_te_svd = svd_data["X_te_svd"]
    svd_cols = [f"text_svd_{i:02d}" for i in range(X_tr_svd.shape[1])]
    df_svd_tr = pd.DataFrame(X_tr_svd, columns=svd_cols, dtype=np.float32)
    df_svd_te = pd.DataFrame(X_te_svd, columns=svd_cols, dtype=np.float32)
    del X_tr_svd, X_te_svd
    gc.collect()

    nb_cache_path = xgb_dir / "nested_nb_probabilities.npz"
    assert nb_cache_path.exists(), f"Missing Nested NB cache at {nb_cache_path}"
    nb_data = np.load(nb_cache_path)
    df_nb_tr = pd.DataFrame({
        "nested_nb_prob": nb_data["nb_prob_tr"],
        "nested_nb_logit": nb_data["nb_logit_tr"],
    })
    df_nb_te = pd.DataFrame({
        "nested_nb_prob": nb_data["nb_prob_te"],
        "nested_nb_logit": nb_data["nb_logit_te"],
    })

    log("Features loaded: 127 base + 64 SVD + 2 NB = 193 static features.")

    # 4. 5-Fold Stratified Cross-Validation Setup
    log("[Step 4/5] Executing 5-Fold Stratified Training with MLPClassifier and StandardScaler...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    oof_predictions = np.zeros(n_train, dtype=np.float32)
    test_fold_probabilities = np.zeros((5, n_test), dtype=np.float32)
    fold_metrics = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(df_base_tr, y_train)):
        t_f0 = time.time()
        log(f"\n--- FOLD {fold+1}/5 ---")
        log(f"Train samples: {len(trn_idx)}, Val samples: {len(val_idx)}")

        # Out-of-Fold Tabular Aggregations strictly on fold train partition (14 features)
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

        # Assemble Full 207-Feature Matrix
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

        assert X_trn.shape[1] == 207, f"Expected 207 features, got {X_trn.shape[1]}"
        if fold == 0:
            log(f"Verified feature count: {X_trn.shape[1]} features.")

        # Clean any invalid values before scaling
        X_trn_clean = np.nan_to_num(X_trn.values, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
        X_val_clean = np.nan_to_num(X_val.values, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
        X_test_clean = np.nan_to_num(X_test_fold.values, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)

        # StandardScaler strictly fit on train fold
        scaler = StandardScaler()
        X_trn_scaled = scaler.fit_transform(X_trn_clean)
        X_val_scaled = scaler.transform(X_val_clean)
        X_test_scaled = scaler.transform(X_test_clean)

        # Clip extreme scaled values to stabilize neural activations
        X_trn_scaled = np.clip(X_trn_scaled, -10.0, 10.0)
        X_val_scaled = np.clip(X_val_scaled, -10.0, 10.0)
        X_test_scaled = np.clip(X_test_scaled, -10.0, 10.0)

        # Multi-Layer Perceptron Classifier
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=args.alpha,
            batch_size=args.batch_size,
            learning_rate_init=args.lr,
            early_stopping=True,
            max_iter=args.max_iter,
            random_state=args.seed + fold,
            verbose=False,
        )

        mlp.fit(X_trn_scaled, y_train[trn_idx])

        # Predict out-of-fold validation probabilities
        val_probs = mlp.predict_proba(X_val_scaled)[:, 1].astype(np.float32)
        oof_predictions[val_idx] = val_probs

        acc_f = float(accuracy_score(y_train[val_idx], (val_probs >= 0.5).astype(int)))
        ll_f = float(log_loss(y_train[val_idx], np.clip(val_probs, 1e-7, 1.0 - 1e-7)))
        auc_f = float(roc_auc_score(y_train[val_idx], val_probs))
        epochs = int(mlp.n_iter_)

        log(f"Fold {fold+1} Finished ({time.time()-t_f0:.1f}s) | Epochs: {epochs} | Acc: {acc_f:.5f} | LogLoss: {ll_f:.5f} | AUC: {auc_f:.5f}")
        fold_metrics.append({
            "fold": int(fold + 1),
            "accuracy": acc_f,
            "logloss": ll_f,
            "auc": auc_f,
            "epochs": epochs,
        })

        # Predict on Test fold
        test_pred_fold = mlp.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
        test_fold_probabilities[fold] = test_pred_fold

        del X_trn, X_val, X_test_fold, X_trn_clean, X_val_clean, X_test_clean
        del X_trn_scaled, X_val_scaled, X_test_scaled, scaler, mlp
        del fold_trn_agg, fold_val_agg, fold_test_agg
        gc.collect()

    # 5. Full OOF Evaluation and Output Generation
    log("\n[Step 5/5] Computing Full 5-Fold OOF Metrics and Generating Output Artifacts...")
    oof_accuracy_05 = float(accuracy_score(y_train, (oof_predictions >= 0.5).astype(int)))
    oof_ll = float(log_loss(y_train, np.clip(oof_predictions, 1e-7, 1.0 - 1e-7)))
    oof_auc = float(roc_auc_score(y_train, oof_predictions))

    # Optimal threshold search
    th_grid = np.arange(0.35, 0.651, 0.005)
    best_th = float(max(th_grid, key=lambda t: accuracy_score(y_train, (oof_predictions >= t).astype(int))))
    best_acc = float(accuracy_score(y_train, (oof_predictions >= best_th).astype(int)))

    # Average test probabilities across all 5 folds
    test_probs_mean = test_fold_probabilities.mean(axis=0)

    log("=" * 80)
    log("=== FINAL 5-FOLD MLP NEURAL MODEL OOF RESULTS ===")
    log(f"OOF Accuracy @ 0.50: {oof_accuracy_05:.5f} ({oof_accuracy_05*100:.3f}%)")
    log(f"OOF Optimal Acc:     {best_acc:.5f} at threshold {best_th:.3f}")
    log(f"OOF LogLoss:         {oof_ll:.5f}")
    log(f"OOF ROC-AUC:         {oof_auc:.5f}")
    log(f"Test Mean Probability: {test_probs_mean.mean():.4f}")
    log("=" * 80)

    # 1. Save OOF Predictions CSV
    oof_df = pd.DataFrame({
        "id": train_ids,
        "seller_id": train_sellers,
        "y_true": y_train,
        "condition": np.where(y_train == 1, "used", "new"),
        "pred_proba": oof_predictions,
        "pred_condition_05": np.where(oof_predictions >= 0.5, "used", "new"),
        "pred_condition_opt": np.where(oof_predictions >= best_th, "used", "new"),
    })
    oof_path = out_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    log(f"Saved OOF predictions to {oof_path}")

    # 2. Save Test Probabilities NPY
    test_npy_path = out_dir / "test_probabilities.npy"
    np.save(test_npy_path, test_probs_mean)
    log(f"Saved test probabilities NPY to {test_npy_path} (shape: {test_probs_mean.shape})")

    # 3. Save Test Probabilities CSV
    test_df = pd.DataFrame({
        "ID": np.arange(1, len(test_probs_mean) + 1),
        "id": test_ids,
        "pred_proba": test_probs_mean,
        "predicted_condition": np.where(test_probs_mean >= 0.5, "used", "new"),
    })
    test_csv_path = out_dir / "test_probabilities.csv"
    test_df.to_csv(test_csv_path, index=False)
    log(f"Saved test probabilities CSV to {test_csv_path}")

    # 4. Save Metrics Summary JSON
    total_elapsed = time.time() - t_start
    summary = {
        "architecture": "MLPClassifier",
        "num_features": 207,
        "n_train": n_train,
        "n_test": n_test,
        "hyperparameters": {
            "hidden_layer_sizes": [256, 128, 64],
            "activation": "relu",
            "solver": "adam",
            "alpha": args.alpha,
            "batch_size": args.batch_size,
            "learning_rate_init": args.lr,
            "early_stopping": True,
            "max_iter": args.max_iter,
            "random_state": args.seed,
        },
        "scaler": "StandardScaler",
        "cv_folds": fold_metrics,
        "oof_accuracy_05": oof_accuracy_05,
        "oof_accuracy_optimal": best_acc,
        "optimal_threshold": best_th,
        "oof_logloss": oof_ll,
        "oof_auc": oof_auc,
        "test_used_count": int((test_probs_mean >= 0.5).sum()),
        "test_new_count": int((test_probs_mean < 0.5).sum()),
        "elapsed_seconds": round(total_elapsed, 2),
        "elapsed_minutes": round(total_elapsed / 60.0, 2),
        "output_files": {
            "oof_csv": str(oof_path),
            "test_npy": str(test_npy_path),
            "test_csv": str(test_csv_path),
            "metrics_json": str(out_dir / "metrics.json"),
        },
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved metrics summary to {metrics_path}")
    log(f"All operations completed in {total_elapsed / 60.0:.2f} minutes.")


if __name__ == "__main__":
    main()
