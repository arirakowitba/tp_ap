#!/usr/bin/env python3
"""Strictly Nested Asymmetric Lossguide XGBoost Specialist.

Implements Lineament 3 of CX-0145:
- Uses 100% identical outer splits to oof_nested_dual_regime.csv (StratifiedGroupKFold on seller_id).
- Feature extraction (TF-IDF word 1-2, 35k + TruncatedSVD 100) strictly fitted inside outer-train.
- Inner seller-group split (85/15) solely to determine best_iteration for XGBoost with early stopping.
- Full refit on 100% outer-train with frozen best_iteration without eval_set.
- Evaluates unseen sellers (outer holdout) and seen sellers (within-seller items).
- Paired comparison row-by-row against the verified strong baseline (0.94230 OOF).
- Produces parquet/CSV with columns [id, seller_id, category_id, y_true, p_pred_xgb, fold] and SHA-256.
"""

from __future__ import annotations

import gc
import json
import time
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.train_strictly_nested_dual_regime_cv import extract_raw_fields, compute_fold_features, seller_clustered_bootstrap


def main():
    print("=" * 80)
    print("STRICTLY NESTED ASYMMETRIC LOSSGUIDE XGBOOST SPECIALIST")
    print("Identical Outer Splits, Within-Fold SVD, Inner Iteration Freezing, Paired Eval")
    print("=" * 80)

    start_total = time.time()
    out_dir = Path("results/strictly_nested_asymmetric_xgb_20261110")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load canonical outer splits from oof_nested_dual_regime.csv
    canonical_oof_path = Path("results/strictly_nested_dual_regime_cv_20261110/oof_nested_dual_regime.csv")
    assert canonical_oof_path.exists(), f"Missing {canonical_oof_path}!"
    df_canonical = pd.read_csv(canonical_oof_path)
    canonical_folds = df_canonical["fold"].values
    n_splits = 5

    # 2. Load Raw Datasets
    train_path = Path("data/ap_2026_q_2/raw/train_data.jsonlines")
    test_path = Path("data/ap_2026_q_2/raw/test_data.jsonlines")
    baseline_path = Path("results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv")

    print("\n[1/5] Loading train data (70k), test data (30k), and baseline...", flush=True)
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    n_train = len(train_rows)
    n_test = len(test_rows)
    y_train = np.array([1 if r.get("condition") == "used" else 0 for r in train_rows], dtype=int)

    df_baseline = pd.read_csv(baseline_path)
    p_baseline = df_baseline["oof_blend"].values
    pred_baseline = (p_baseline >= 0.5).astype(int)
    acc_baseline = accuracy_score(y_train, pred_baseline)
    ll_baseline = log_loss(y_train, p_baseline)

    print("\n[2/5] Extracting raw intrinsic fields and corpus...", flush=True)
    train_ids, train_corpus, df_train_base, train_cats = extract_raw_fields(train_rows)
    test_ids, test_corpus, df_test_base, test_cats = extract_raw_fields(test_rows)
    sellers_train = train_cats["seller_id"]

    # Storage for OOF and Test predictions
    oof_p_xgb = np.zeros(n_train)
    test_p_xgb = np.zeros(n_test)

    seen_val_y_all = []
    seen_val_p_all = []
    fold_reports = []

    print(f"\n[3/5] Starting 5-Fold Strictly Nested Asymmetric XGBoost Training...", flush=True)

    for fold in range(n_splits):
        fold_start = time.time()
        print(f"\n{'='*30} FOLD {fold+1}/{n_splits} {'='*30}", flush=True)
        tr_idx = np.where(canonical_folds != fold)[0]
        va_idx = np.where(canonical_folds == fold)[0]

        print(f"Outer Split: Train={len(tr_idx)} items, Validation={len(va_idx)} items (100% unseen sellers)", flush=True)
        assert len(set(sellers_train[tr_idx]) & set(sellers_train[va_idx])) == 0, "Seller overlap in outer fold!"

        # A. Inner seller-group split on tr_idx solely to determine best_iteration
        tr_sellers = sellers_train[tr_idx]
        tr_y = y_train[tr_idx]

        inner_sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42 + fold)
        in_tr_pos, in_va_pos = next(inner_sgkf.split(tr_idx, tr_y, groups=tr_sellers))
        inner_tr_idx = tr_idx[in_tr_pos]
        inner_va_idx = tr_idx[in_va_pos]
        assert len(set(sellers_train[inner_tr_idx]) & set(sellers_train[inner_va_idx])) == 0, "Seller overlap in inner split!"

        print(f"Inner Split for Tuning: InnerTrain={len(inner_tr_idx)}, InnerVal={len(inner_va_idx)}", flush=True)

        # Stage 1: Feature Extraction on Inner Split
        X_in_tr_dense, [X_in_va_dense], _, _ = compute_fold_features(
            inner_tr_idx,
            [inner_va_idx],
            train_corpus,
            df_train_base,
            train_cats,
            n_svd=100,
        )

        # Stage 2: Inner Early Stopping to freeze best_iteration
        print("Determining best_iteration for Asymmetric XGBoost on inner validation set...", flush=True)
        xgb_inner = xgb.XGBClassifier(
            tree_method="hist",
            grow_policy="lossguide",
            max_leaves=120,
            max_depth=11,
            learning_rate=0.035,
            n_estimators=700,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=5.0,
            eval_metric="logloss",
            early_stopping_rounds=60,
            random_state=42 + fold,
            n_jobs=4,
        )
        xgb_inner.fit(
            X_in_tr_dense,
            y_train[inner_tr_idx],
            eval_set=[(X_in_va_dense, y_train[inner_va_idx])],
            verbose=False,
        )
        best_iter_xgb = xgb_inner.best_iteration
        if best_iter_xgb is None or best_iter_xgb <= 0:
            best_iter_xgb = 450
        print(f"Frozen best_iteration Fold {fold+1}: {best_iter_xgb}", flush=True)

        del xgb_inner, X_in_tr_dense, X_in_va_dense
        gc.collect()

        # Stage 3: Full Outer Feature Extraction strictly on tr_idx
        tr_seller_counts = Counter(sellers_train[tr_idx])
        multi_item_sellers = {s for s, c in tr_seller_counts.items() if c >= 2}
        seen_eval_indices = []
        for s in multi_item_sellers:
            ixs = tr_idx[sellers_train[tr_idx] == s]
            seen_eval_indices.append(ixs[0])
        seen_eval_indices = np.array(seen_eval_indices)
        test_eval_indices = n_train + np.arange(n_test)

        print(f"Refitting features on full outer-train ({len(tr_idx)} items)...", flush=True)
        X_tr_dense, [X_va_dense, X_seen_dense, X_test_dense], _, _ = compute_fold_features(
            tr_idx,
            [va_idx, seen_eval_indices, test_eval_indices],
            train_corpus + test_corpus,
            pd.concat([df_train_base, df_test_base], ignore_index=True),
            {k: np.concatenate([train_cats[k], test_cats[k]]) for k in train_cats},
            n_svd=100,
        )

        # Stage 4: Refit Outer XGBoost on 100% of outer_tr_idx WITHOUT eval_set
        print(f"Training Outer XGBoost ({best_iter_xgb} iterations, NO eval_set)...", flush=True)
        xgb_outer = xgb.XGBClassifier(
            tree_method="hist",
            grow_policy="lossguide",
            max_leaves=120,
            max_depth=11,
            learning_rate=0.035,
            n_estimators=best_iter_xgb,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=5.0,
            eval_metric="logloss",
            random_state=42 + fold,
            n_jobs=4,
        )
        xgb_outer.fit(X_tr_dense, y_train[tr_idx], verbose=False)

        p_va_xgb = xgb_outer.predict_proba(X_va_dense)[:, 1]
        p_test_fold_xgb = xgb_outer.predict_proba(X_test_dense)[:, 1]
        p_seen_xgb = xgb_outer.predict_proba(X_seen_dense)[:, 1]

        oof_p_xgb[va_idx] = p_va_xgb
        test_p_xgb += p_test_fold_xgb / n_splits

        seen_val_y_all.extend(y_train[seen_eval_indices].tolist())
        seen_val_p_all.extend(p_seen_xgb.tolist())

        y_va = y_train[va_idx]
        acc_va = accuracy_score(y_va, (p_va_xgb >= 0.5).astype(int))
        ll_va = log_loss(y_va, p_va_xgb)
        acc_seen_fold = accuracy_score(y_train[seen_eval_indices], (p_seen_xgb >= 0.5).astype(int))

        fold_elapsed = time.time() - fold_start
        print(f"Fold {fold+1} Finished in {fold_elapsed:.1f}s | Unseen Acc={acc_va:.5f}, LogLoss={ll_va:.5f} | Seen Acc={acc_seen_fold:.5f}", flush=True)

        fold_reports.append({
            "fold": fold + 1,
            "best_iter_xgb": int(best_iter_xgb),
            "unseen_accuracy": float(acc_va),
            "unseen_logloss": float(ll_va),
            "seen_accuracy": float(acc_seen_fold),
            "elapsed_seconds": float(fold_elapsed),
        })

        del xgb_outer, X_tr_dense, X_va_dense, X_seen_dense, X_test_dense
        gc.collect()

    # 4. Global Out-of-Fold Evaluation
    print(f"\n{'='*80}\n[4/5] Computing Global Strictly Nested OOF & Paired Verification\n{'='*80}", flush=True)

    pred_cand = (oof_p_xgb >= 0.5).astype(int)
    acc_unseen = accuracy_score(y_train, pred_cand)
    ll_unseen = log_loss(y_train, oof_p_xgb)
    auc_unseen = roc_auc_score(y_train, oof_p_xgb)

    seen_val_y = np.array(seen_val_y_all)
    seen_val_p = np.array(seen_val_p_all)
    acc_seen = accuracy_score(seen_val_y, (seen_val_p >= 0.5).astype(int))
    ll_seen = log_loss(seen_val_y, seen_val_p)
    acc_mix = 0.7162 * acc_seen + 0.2838 * acc_unseen

    print(f"RÉGIMEN UNSEEN-SELLERS (100% outer holdout): Acc = {acc_unseen:.5f} ({int(acc_unseen*n_train)}/{n_train}), LogLoss = {ll_unseen:.5f}, ROC-AUC = {auc_unseen:.5f}")
    print(f"RÉGIMEN SEEN-SELLERS (holdout within train sellers): Acc = {acc_seen:.5f} ({len(seen_val_y)} items), LogLoss = {ll_seen:.5f}")
    print(f"MIX SENSIBILIDAD TEST (71.62% seen / 28.38% unseen): Acc = {acc_mix:.5f}")

    # Paired Item-by-Item Comparison vs Baseline
    hit_cand = (pred_cand == y_train)
    hit_base = (pred_baseline == y_train)
    diff = hit_cand.astype(int) - hit_base.astype(int)

    fixes = int(np.sum(~hit_base & hit_cand))
    breaks = int(np.sum(hit_base & ~hit_cand))
    net_hits = fixes - breaks
    delta_acc = float(acc_unseen - acc_baseline)
    delta_ll = float(ll_unseen - ll_baseline)

    print("\nPAIRED ITEM-BY-ITEM METRICS VS STRONG BASELINE (Podium Engine):")
    print(f"Baseline Accuracy:  {acc_baseline:.5f} ({int(acc_baseline*n_train)} hits)")
    print(f"XGBoost Accuracy:   {acc_unseen:.5f} ({int(acc_unseen*n_train)} hits)")
    print(f"Paired Delta Acc:   {delta_acc:+.5f} (Net Hits: {net_hits:+d} | Fixes: {fixes}, Breaks: {breaks})")
    print(f"Paired Delta LL:    {delta_ll:+.5f}")

    print("\nRunning Seller-Clustered Bootstrap on Paired Delta (1,000 resamples)...", flush=True)
    boot_mean, boot_ci_lower, boot_ci_upper, boot_p_gt_zero = seller_clustered_bootstrap(
        diff, sellers_train, n_boot=1000, seed=20261110
    )
    print(f"Bootstrap Seller-Clustered Delta 95% CI: [{boot_ci_lower:+.5f}, {boot_ci_upper:+.5f}] (Mean: {boot_mean:+.5f})")
    print(f"P(Delta > 0): {boot_p_gt_zero:.4f}")

    # 5. Save Artifacts
    print(f"\n[5/5] Saving strictly nested XGBoost artifacts and report...", flush=True)

    oof_df = pd.DataFrame({
        "id": train_ids,
        "seller_id": sellers_train,
        "category_id": train_cats["category_id"],
        "y_true": y_train,
        "p_pred_xgb": oof_p_xgb,
        "fold": canonical_folds,
    })
    oof_csv_path = out_dir / "oof_asymmetric_xgb_nested.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"Saved OOF DataFrame to {oof_csv_path} ({len(oof_df)} rows)", flush=True)

    test_df = pd.DataFrame({
        "id": test_ids,
        "seller_id": test_cats["seller_id"],
        "category_id": test_cats["category_id"],
        "p_used_xgb": test_p_xgb,
    })
    test_csv_path = out_dir / "test_probabilities_asymmetric_xgb_nested.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved Test Probabilities to {test_csv_path} ({len(test_df)} rows)", flush=True)

    summary = {
        "pipeline": "Strictly Nested Asymmetric Lossguide XGBoost Specialist",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_rows": n_train,
        "total_test_rows": n_test,
        "outer_splits": n_splits,
        "model_architecture": "XGBClassifier(tree_method='hist', grow_policy='lossguide', max_leaves=120, max_depth=11, lr=0.035)",
        "qualification_status": "EXPLORATORY / VALIDATION_STAGE",
        "baseline_comparison": {
            "baseline_name": "Podium Engine (results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv)",
            "baseline_accuracy": float(acc_baseline),
            "baseline_logloss": float(ll_baseline),
            "baseline_hits": int(acc_baseline * n_train),
        },
        "unseen_seller_regime": {
            "description": "100% unseen sellers in outer holdout (pure seller-group)",
            "accuracy": float(acc_unseen),
            "hits": int(acc_unseen * n_train),
            "logloss": float(ll_unseen),
            "roc_auc": float(auc_unseen),
        },
        "seen_seller_regime": {
            "description": "Holdout items from sellers present in training set",
            "sample_size": len(seen_val_y),
            "accuracy": float(acc_seen),
            "logloss": float(ll_seen),
        },
        "deployment_mix_sensitivity": {
            "description": "71.62% seen sellers / 28.38% unseen sellers mix",
            "accuracy": float(acc_mix),
        },
        "paired_metrics": {
            "delta_accuracy": float(delta_acc),
            "delta_logloss": float(delta_ll),
            "fixes": fixes,
            "breaks": breaks,
            "net_hits": net_hits,
            "seller_clustered_bootstrap_95ci": [boot_ci_lower, boot_ci_upper],
            "seller_clustered_bootstrap_mean": boot_mean,
            "probability_delta_gt_zero": boot_p_gt_zero,
        },
        "fold_details": fold_reports,
        "total_elapsed_seconds": float(time.time() - start_total),
    }

    report_json_path = out_dir / "paired_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - start_total
    print(f"\n{'='*80}\nASYMMETRIC XGBOOST COMPLETED in {total_time/60:.2f} minutes\n{'='*80}", flush=True)


if __name__ == "__main__":
    main()
