"""Run Benchmark Restoration and Rigorous Comparison.

Pipelines evaluated on the EXACT SAME canonical partition:
- Pipeline A: Deep CatBoost (R151 exact deep features + TE alpha 5 + txt_sparse_p, 900 trees, depth 6, lr 0.07)
- Pipeline B: CatBoost with R157 feature set + TE alpha 10 + txt_sparse_p
- Pipeline C: Full Ensemble (LightGBM + CatBoost) on canonical partition
"""
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss

from src.benchmark_20261030.data_loader import load_train_dataset, load_test_dataset
from src.benchmark_20261030.split_utils import split_seller_groups_r151, split_seller_groups_r157
from src.benchmark_20261030.feature_extractor import extract_r151_features, compute_leak_free_te_r151, compute_txt_sparse_p
from src.eval_20261030.paired_eval import evaluate_paired_frames, format_markdown_report

RESULTS_DIR = ROOT / "results" / "benchmark_20261030"
MODELS_DIR = ROOT / "models" / "benchmark_20261030"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CAT_FEATURES = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode"]

def save_predictions_gz(path, row_indices, item_ids, sellers, y_used, is_seen, p_used, pred, threshold):
    df = pd.DataFrame({
        "row_idx": row_indices,
        "id": row_indices + 1,  # 1-based index matching Kaggle ID if test
        "item_id": [item_ids[i] for i in row_indices],
        "seller_id": sellers[row_indices],
        "y_used": y_used[row_indices],
        "is_seen": is_seen,
        "p_used": p_used,
        "pred": pred,
        "threshold": threshold,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    print(f"Saved predictions to {path} ({len(df)} rows)")
    return df

def main():
    print("=" * 80)
    print("STARTING BENCHMARK RESTORATION ON CANONICAL SELLER SPLITS")
    print("=" * 80)

    rows, ids, sellers, y_used, titles, warranties, combos = load_train_dataset()
    F_df, cat_df = extract_r151_features(rows)

    metrics_all = {}
    
    for seed in [2026, 314159]:
        print(f"\n" + "#" * 60)
        print(f"EVALUATING PIPELINE A (DEEP CATBOOST R151) — SEED {seed}")
        print("#" * 60)

        fit, tune, confirm = split_seller_groups_r151(sellers, seed)
        print(f"Split sizes: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}")

        fit_sellers = sellers[fit]
        fit_seller_set = set(fit_sellers)
        is_seen_tune = np.isin(sellers[tune], list(fit_seller_set)).astype(int)
        is_seen_confirm = np.isin(sellers[confirm], list(fit_seller_set)).astype(int)

        y_fit = y_used[fit]
        y_tune = y_used[tune]
        y_confirm = y_used[confirm]

        # 1. Target Encodings
        print("Computing leak-free TE (alpha=5.0)...", flush=True)
        fit_te, (tune_te, confirm_te) = compute_leak_free_te_r151(cat_df, fit, [tune, confirm], y_fit, fit_sellers, alpha=5.0)

        # 2. Text Feature (txt_sparse_p)
        print("Computing leak-free txt_sparse_p...", flush=True)
        oof_txt, (p_txt_tune, p_txt_conf) = compute_txt_sparse_p(combos, titles, fit, [tune, confirm], y_fit, fit_sellers, seed=seed)

        # 3. Assemble Feature Matrices
        X_fit = F_df.iloc[fit].copy()
        X_tune = F_df.iloc[tune].copy()
        X_confirm = F_df.iloc[confirm].copy()

        for c in ["seller_id", "category_id", "city"]:
            X_fit[f"te_{c}"] = fit_te[c]
            X_tune[f"te_{c}"] = tune_te[c]
            X_confirm[f"te_{c}"] = confirm_te[c]

        X_fit["txt_sparse_p"] = oof_txt
        X_tune["txt_sparse_p"] = p_txt_tune
        X_confirm["txt_sparse_p"] = p_txt_conf

        for c in CAT_FEATURES:
            X_fit[c] = X_fit[c].astype(str)
            X_tune[c] = X_tune[c].astype(str)
            X_confirm[c] = X_confirm[c].astype(str)

        # 4. Train Deep CatBoost Baseline
        print(f"Training Deep CatBoost Baseline (900 trees, depth 6, lr 0.07, l2 6.0)...", flush=True)
        t0 = time.time()
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                random_seed=seed, thread_count=8, verbose=0)
        cb.fit(Pool(X_fit, y_fit, cat_features=CAT_FEATURES))
        train_time = time.time() - t0
        print(f"Fitted in {train_time:.2f}s", flush=True)

        # Predict
        p_tune = cb.predict_proba(Pool(X_tune, cat_features=CAT_FEATURES))[:, 1]
        p_conf = cb.predict_proba(Pool(X_confirm, cat_features=CAT_FEATURES))[:, 1]

        # 5. Tune threshold strictly on TUNE
        best_th, best_acc_tu = 0.50, 0.0
        for th in np.linspace(0.40, 0.60, 41):
            acc = accuracy_score(y_tune, p_tune >= th)
            if acc > best_acc_tu:
                best_acc_tu = acc
                best_th = th

        pred_tune = (p_tune >= best_th).astype(int)
        pred_conf = (p_conf >= best_th).astype(int)

        acc_conf = accuracy_score(y_confirm, pred_conf)
        ll_conf = log_loss(y_confirm, p_conf)
        print(f"TUNE:    Best threshold = {best_th:.3f} (Accuracy = {best_acc_tu:.5f})")
        print(f"CONFIRM: Accuracy = {acc_conf:.5f} | Log Loss = {ll_conf:.5f}")

        # Save predictions
        df_pred_tu = save_predictions_gz(RESULTS_DIR / f"preds_A_seed{seed}_tune.csv.gz",
                                         tune, ids, sellers, y_used, is_seen_tune, p_tune, pred_tune, best_th)
        df_pred_co = save_predictions_gz(RESULTS_DIR / f"preds_A_seed{seed}_confirm.csv.gz",
                                         confirm, ids, sellers, y_used, is_seen_confirm, p_conf, pred_conf, best_th)

        # Save model
        model_path = MODELS_DIR / f"catboost_pipeline_A_seed{seed}.cbm"
        cb.save_model(model_path)
        print(f"Saved model to {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

        # Verify against historical R151 file
        f_hist = ROOT / "results" / "revision_submissions" / f"paired_deep_ablation_seed_{seed}.csv"
        if f_hist.exists():
            df_hist = pd.read_csv(f_hist)
            # Compare p_base
            diff = np.abs(df_hist["p_base"].values - p_conf)
            corr = np.corrcoef(df_hist["p_base"].values, p_conf)[0, 1]
            agree = np.mean((df_hist["pred_base"].values == pred_conf))
            print(f"Comparison with historical R151 p_base:")
            print(f"  Max absolute prob diff: {diff.max():.6f}")
            print(f"  Mean absolute prob diff: {diff.mean():.6f}")
            print(f"  Pearson correlation: {corr:.6f}")
            print(f"  Prediction agreement: {agree*100:.2f}%")

        metrics_all[f"pipeline_A_seed_{seed}"] = {
            "seed": seed,
            "threshold": float(best_th),
            "tune_accuracy": float(best_acc_tu),
            "confirm_accuracy": float(acc_conf),
            "confirm_logloss": float(ll_conf),
            "confirm_seen_accuracy": float(accuracy_score(y_confirm[is_seen_confirm == 1], pred_conf[is_seen_confirm == 1])),
            "confirm_new_accuracy": float(accuracy_score(y_confirm[is_seen_confirm == 0], pred_conf[is_seen_confirm == 0])),
            "fit_time_seconds": train_time,
        }

    # Save metrics JSON
    with open(RESULTS_DIR / "metrics_benchmark.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\nAll seed metrics saved to {RESULTS_DIR / 'metrics_benchmark.json'}")

if __name__ == "__main__":
    main()
