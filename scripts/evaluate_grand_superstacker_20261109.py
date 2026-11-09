"""Grand Super-Stacker Optimization across All Diverse & Feedback-Integrated Models.

Combines:
1. Expanded Feedback LightGBM (Acc: 0.93914, LogLoss: 0.15826)
2. Expanded Feedback XGBoost
3. High Capacity LightGBM Feedback (Acc: 0.93859, LogLoss: 0.15893)
4. High Capacity LightGBM Base (Acc: 0.93824, LogLoss: 0.15937)
5. XGBoost Diversity Base (Acc: 0.93666, LogLoss: 0.15992)
6. CatBoost Depth 8 (Acc: 0.92799, LogLoss: 0.18028)
7. ExtraTrees Bagging (Acc: 0.92646, LogLoss: 0.20682)
8. MLP Neural Net (Acc: 0.92370, LogLoss: 0.19426)
9. Sparse Linear SAGA (Acc: 0.84704, LogLoss: 0.35466)

Uses bounded non-negative least squares and L2 logistic meta-regression in logit space.
Computes OOF Accuracy, LogLoss, ROC-AUC, and generates updated test probabilities.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit, logit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(".")

def load_model(name, oof_path, test_path, y_true, col=None):
    oof_p = ROOT / oof_path
    test_p = ROOT / test_path
    if not oof_p.exists() or not test_p.exists():
        print(f"Skipping {name}: file not found ({oof_p} or {test_p})")
        return None
    
    df_oof = pd.read_csv(oof_p)
    if col and col in df_oof.columns:
        p_oof = df_oof[col].values
    elif "prob_used" in df_oof.columns:
        p_oof = df_oof["prob_used"].values
    elif "pred_proba" in df_oof.columns:
        p_oof = df_oof["pred_proba"].values
    else:
        p_oof = df_oof.iloc[:, 1].values
    
    if str(test_p).endswith(".npy"):
        p_test = np.load(test_p)
    else:
        df_test = pd.read_csv(test_p)
        if col and col in df_test.columns:
            p_test = df_test[col].values
        elif "prob_used" in df_test.columns:
            p_test = df_test["prob_used"].values
        elif "pred_proba" in df_test.columns:
            p_test = df_test["pred_proba"].values
        else:
            p_test = df_test.iloc[:, 1].values
        
    p_oof = np.clip(p_oof, 1e-6, 1.0 - 1e-6)
    p_test = np.clip(p_test, 1e-6, 1.0 - 1e-6)
    return {
        "name": name,
        "p_oof": p_oof,
        "logit_oof": logit(p_oof),
        "p_test": p_test,
        "logit_test": logit(p_test),
        "acc": accuracy_score(y_true, (p_oof >= 0.5).astype(int)),
        "ll": log_loss(y_true, p_oof)
    }

def main():
    print("=== GRAND SUPER-STACKER OPTIMIZATION (ALL ARCHITECTURES) ===", flush=True)
    t0 = time.time()
    
    # Ground truth
    train_df = pd.read_json(ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines", lines=True)
    y_true = (train_df["condition"] == "used").astype(int).values
    N_tr = len(y_true)
    
    models = []
    
    # Candidate models
    specs = [
        ("LGBM_Expanded_Feedback", "results/expanded_feedback_lgbm_5fold/oof_predictions.csv", "results/expanded_feedback_lgbm_5fold/test_probabilities.npy"),
        ("XGB_Expanded_Feedback", "results/expanded_feedback_xgb_5fold/oof_predictions.csv", "results/expanded_feedback_xgb_5fold/test_probabilities.npy"),
        ("LGBM_Feedback_Base", "results/high_capacity_lgbm_feedback_5fold/oof_predictions.csv", "results/high_capacity_lgbm_feedback_5fold/test_probabilities.npy"),
        ("LGBM_HighCapacity_Base", "results/high_capacity_lgbm_5fold/oof_predictions.csv", "results/high_capacity_lgbm_5fold/test_probabilities.npy"),
        ("XGB_Diversity_Base", "results/xgboost_diversity_5fold/oof_predictions.csv", "results/xgboost_diversity_5fold/test_probabilities.npy"),
        ("CatBoost_Depth8", "results/catboost_depth8_5fold/oof_predictions.csv", "results/catboost_depth8_5fold/test_probabilities.npy"),
        ("ExtraTrees_Bagging", "results/extratrees_5fold/oof_predictions.csv", "results/extratrees_5fold/test_probabilities.npy"),
        ("MLP_Neural", "results/mlp_neural_5fold/oof_predictions.csv", "results/mlp_neural_5fold/test_probabilities.npy"),
        ("Sparse_Linear_SAGA", "results/sparse_linear_5fold/oof_predictions.csv", "results/sparse_linear_5fold/test_probabilities.npy"),
    ]
    
    for name, oof_p, test_p in specs:
        m = load_model(name, oof_p, test_p, y_true)
        if m is not None:
            models.append(m)
            acc = accuracy_score(y_true, (m["p_oof"] >= 0.5).astype(int))
            ll = log_loss(y_true, m["p_oof"])
            auc = roc_auc_score(y_true, m["p_oof"])
            hits = int(round(acc * N_tr))
            print(f"Loaded {name:<26} | OOF Acc: {acc:.5f} ({hits} hits) | LogLoss: {ll:.5f} | AUC: {auc:.5f}")

    K = len(models)
    print(f"\nTotal active models in ensemble pool: {K}")
    
    # Design matrix in logit space
    X_logit_oof = np.column_stack([m["logit_oof"] for m in models])
    X_logit_test = np.column_stack([m["logit_test"] for m in models])
    
    # 1. Non-negative Least Squares in Logit Space
    # Target in logit space: smooth binary target
    y_target_logit = np.where(y_true == 1, 4.0, -4.0)
    beta_nnls, _ = nnls(X_logit_oof, y_target_logit)
    sum_beta = beta_nnls.sum()
    if sum_beta > 0:
        weights_norm = beta_nnls / sum_beta
    else:
        weights_norm = np.ones(K) / K
        
    print("\n--- NON-NEGATIVE LOGIT WEIGHTS ---")
    for m, w in zip(models, weights_norm):
        print(f"  {m['name']:<26}: {w:.4f}")
        
    ensemble_logit_oof = X_logit_oof @ weights_norm
    ensemble_p_oof = expit(ensemble_logit_oof)
    
    acc_nnls = accuracy_score(y_true, (ensemble_p_oof >= 0.5).astype(int))
    ll_nnls = log_loss(y_true, ensemble_p_oof)
    auc_nnls = roc_auc_score(y_true, ensemble_p_oof)
    hits_nnls = int(round(acc_nnls * N_tr))
    
    print("\n--- ENSEMBLE OOF PERFORMANCE (Weighted Logit Blend) ---")
    print(f"OOF Accuracy : {acc_nnls:.5f} ({hits_nnls} / {N_tr} hits, +{hits_nnls - max(int(round(m['acc']*N_tr)) for m in models)} hits net)")
    print(f"OOF LogLoss  : {ll_nnls:.5f}")
    print(f"OOF ROC-AUC  : {auc_nnls:.5f}")
    
    # 2. Logistic Meta-Regression Stacker (L2 Regularized)
    meta = LogisticRegression(C=0.05, max_iter=500, random_state=42)
    meta.fit(X_logit_oof, y_true)
    meta_p_oof = meta.predict_proba(X_logit_oof)[:, 1]
    
    acc_meta = accuracy_score(y_true, (meta_p_oof >= 0.5).astype(int))
    ll_meta = log_loss(y_true, meta_p_oof)
    auc_meta = roc_auc_score(y_true, meta_p_oof)
    hits_meta = int(round(acc_meta * N_tr))
    
    print("\n--- ENSEMBLE OOF PERFORMANCE (L2 Logistic Stacker C=0.05) ---")
    print(f"OOF Accuracy : {acc_meta:.5f} ({hits_meta} / {N_tr} hits, +{hits_meta - max(int(round(m['acc']*N_tr)) for m in models)} hits net)")
    print(f"OOF LogLoss  : {ll_meta:.5f}")
    print(f"OOF ROC-AUC  : {auc_meta:.5f}")
    
    # Predict on test
    final_p_test = meta.predict_proba(X_logit_test)[:, 1]
    
    out_dir = ROOT / "results" / "meta_stacking_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "test_probabilities.npy", final_p_test)
    pd.DataFrame({"ID": range(len(final_p_test)), "prob_used": final_p_test}).to_csv(out_dir / "test_probabilities.csv", index=False)
    
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "n_models": K,
        "models": [m["name"] for m in models],
        "weights_nnls": {m["name"]: float(w) for m, w in zip(models, weights_norm)},
        "nnls_metrics": {"accuracy": acc_nnls, "hits": hits_nnls, "logloss": ll_nnls, "auc": auc_nnls},
        "meta_metrics": {"accuracy": acc_meta, "hits": hits_meta, "logloss": ll_meta, "auc": auc_meta},
    }
    with open(out_dir / "stacker_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSaved updated Super-Stacker probabilities to {out_dir}/ in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
