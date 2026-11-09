#!/usr/bin/env python3
"""Evaluate all trained model architectures and optimize the Super-Blender."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

models = {
    'LightGBM': ('results/high_capacity_lgbm_5fold/oof_predictions.csv', 'prob_used', 'results/high_capacity_lgbm_5fold/test_probabilities.npy'),
    'XGBoost': ('results/xgboost_diversity_5fold/oof_predictions.csv', 'prob_used', 'results/xgboost_diversity_5fold/test_probabilities.npy'),
    'CatBoost': ('results/catboost_depth8_5fold/oof_predictions.csv', 'pred_proba', 'results/catboost_depth8_5fold/test_probabilities.npy'),
    'ExtraTrees': ('results/extratrees_5fold/oof_predictions.csv', 'prob_used', 'results/extratrees_5fold/test_probabilities.npy'),
    'MLP_Neural': ('results/mlp_neural_5fold/oof_predictions.csv', 'pred_proba', 'results/mlp_neural_5fold/test_probabilities.npy'),
    'Sparse_Linear': ('results/sparse_linear_5fold/oof_predictions.csv', 'prob_used', 'results/sparse_linear_5fold/test_probabilities.npy'),
}

df_base = pd.read_csv('results/high_capacity_lgbm_5fold/oof_predictions.csv')
y_true = df_base['y_true'].to_numpy()

oof_dict = {}
te_dict = {}

print("=== 6-MODEL LEAK-FREE 5-FOLD BENCHMARK (70k TRAIN ROWS) ===")
print(f"{'Model':<15} | {'Acc':<8} | {'Hits':<8} | {'LogLoss':<8} | {'ROC-AUC':<8}")
print("-" * 55)
for name, (oof_path, col, te_path) in models.items():
    df = pd.read_csv(oof_path)
    p_oof = df[col].to_numpy()
    p_te = np.load(te_path)
    oof_dict[name] = p_oof
    te_dict[name] = p_te
    acc = accuracy_score(y_true, p_oof >= 0.5)
    hits = int(((p_oof >= 0.5) == y_true).sum())
    ll = log_loss(y_true, np.clip(p_oof, 1e-7, 1 - 1e-7))
    auc = roc_auc_score(y_true, p_oof)
    print(f"{name:<15} | {acc:<8.5f} | {hits:<8d} | {ll:<8.5f} | {auc:<8.5f}")

df_corr = pd.DataFrame(oof_dict).corr()
print("\n=== PAIRWISE PREDICTION CORRELATION MATRIX ===")
print(df_corr.to_string())

# Select high-performing architectures (acc > 0.90) for ensemble optimization
selected = ['LightGBM', 'XGBoost', 'CatBoost', 'ExtraTrees', 'MLP_Neural']
print(f"\n=== OPTIMIZING 5-ARCHITECTURE SUPER-BLENDER ({', '.join(selected)}) ===")

oof_mat = np.column_stack([oof_dict[m] for m in selected])
te_mat = np.column_stack([te_dict[m] for m in selected])
K = len(selected)

def loss_func(w):
    w = np.maximum(0, w)
    if w.sum() == 0:
        return 1e6
    w = w / w.sum()
    p = oof_mat @ w
    return log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7))

res = minimize(loss_func, [1.0 / K] * K, method='SLSQP', bounds=[(0.0, 1.0)] * K)
opt_w = np.maximum(0, res.x)
opt_w = opt_w / opt_w.sum()

print("Optimal SLSQP Weights:")
for m, w in zip(selected, opt_w):
    print(f"  {m:<15}: {w:.4f} ({w*100:.1f}%)")

p_blend_oof = oof_mat @ opt_w
acc_b = accuracy_score(y_true, p_blend_oof >= 0.5)
hits_b = int(((p_blend_oof >= 0.5) == y_true).sum())
ll_b = log_loss(y_true, np.clip(p_blend_oof, 1e-7, 1 - 1e-7))
auc_b = roc_auc_score(y_true, p_blend_oof)

best_single_hits = max(int(((oof_dict[m] >= 0.5) == y_true).sum()) for m in selected)
net_oof_gain = hits_b - best_single_hits

print(f"\nSuper-Blender OOF Accuracy: {acc_b:.5f} ({hits_b}/70,000 hits)")
print(f"Super-Blender OOF LogLoss:  {ll_b:.5f}")
print(f"Super-Blender OOF ROC-AUC:  {auc_b:.5f}")
print(f"Net Gain vs Single Best:    +{net_oof_gain} hits net!")
