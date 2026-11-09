#!/usr/bin/env python3
"""Test Meta-Stacking and non-linear blend combinations across all 5 architectures."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

models = {
    'LightGBM': ('results/high_capacity_lgbm_5fold/oof_predictions.csv', 'prob_used', 'results/high_capacity_lgbm_5fold/test_probabilities.npy'),
    'XGBoost': ('results/xgboost_diversity_5fold/oof_predictions.csv', 'prob_used', 'results/xgboost_diversity_5fold/test_probabilities.npy'),
    'CatBoost': ('results/catboost_depth8_5fold/oof_predictions.csv', 'pred_proba', 'results/catboost_depth8_5fold/test_probabilities.npy'),
    'ExtraTrees': ('results/extratrees_5fold/oof_predictions.csv', 'prob_used', 'results/extratrees_5fold/test_probabilities.npy'),
    'MLP_Neural': ('results/mlp_neural_5fold/oof_predictions.csv', 'pred_proba', 'results/mlp_neural_5fold/test_probabilities.npy'),
}

df_base = pd.read_csv('results/high_capacity_lgbm_5fold/oof_predictions.csv')
y_true = df_base['y_true'].to_numpy()

oof_list = []
te_list = []
names = list(models.keys())
for name in names:
    oof_path, col, te_path = models[name]
    df = pd.read_csv(oof_path)
    oof_list.append(df[col].to_numpy())
    te_list.append(np.load(te_path))

OOF = np.column_stack(oof_list)  # (70000, 5)
TE = np.column_stack(te_list)    # (30000, 5)

print("=== METHOD 1: PROBABILITY LINEAR BLEND (SLSQP) ===")
def loss_func(w):
    w = np.maximum(0, w)
    if w.sum() == 0:
        return 1e6
    w = w / w.sum()
    p = OOF @ w
    return log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7))

res = minimize(loss_func, [1.0 / len(names)] * len(names), method='SLSQP', bounds=[(0.0, 1.0)] * len(names))
opt_w = np.maximum(0, res.x) / np.maximum(0, res.x).sum()
p_linear_oof = OOF @ opt_w
acc_lin = accuracy_score(y_true, p_linear_oof >= 0.5)
ll_lin = log_loss(y_true, np.clip(p_linear_oof, 1e-7, 1 - 1e-7))
auc_lin = roc_auc_score(y_true, p_linear_oof)
print(f"Weights: {dict(zip(names, [round(x, 4) for x in opt_w]))}")
print(f"Acc: {acc_lin:.5f} | Hits: {int(((p_linear_oof >= 0.5) == y_true).sum())} | LogLoss: {ll_lin:.5f} | AUC: {auc_lin:.5f}")

print("\n=== METHOD 2: LOGIT SPACE LINEAR BLEND (Nelder-Mead / SLSQP) ===")
eps = 1e-5
LOGIT_OOF = logit(np.clip(OOF, eps, 1 - eps))
LOGIT_TE = logit(np.clip(TE, eps, 1 - eps))

def loss_logit(w):
    w = np.maximum(0, w)
    if w.sum() == 0:
        return 1e6
    w = w / w.sum()
    p = expit(LOGIT_OOF @ w)
    return log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7))

res_logit = minimize(loss_logit, [1.0 / len(names)] * len(names), method='SLSQP', bounds=[(0.0, 1.0)] * len(names))
opt_w_logit = np.maximum(0, res_logit.x) / np.maximum(0, res_logit.x).sum()
p_logit_oof = expit(LOGIT_OOF @ opt_w_logit)
acc_log = accuracy_score(y_true, p_logit_oof >= 0.5)
ll_log = log_loss(y_true, np.clip(p_logit_oof, 1e-7, 1 - 1e-7))
auc_log = roc_auc_score(y_true, p_logit_oof)
print(f"Logit Weights: {dict(zip(names, [round(x, 4) for x in opt_w_logit]))}")
print(f"Acc: {acc_log:.5f} | Hits: {int(((p_logit_oof >= 0.5) == y_true).sum())} | LogLoss: {ll_log:.5f} | AUC: {auc_log:.5f}")

print("\n=== METHOD 3: 5-FOLD META-STACKER LOGISTIC REGRESSION (CROSS-VALIDATED) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261109)
meta_oof = np.zeros(len(y_true), dtype=np.float32)

for c_val in [0.01, 0.1, 1.0, 5.0, 10.0]:
    for tr_idx, val_idx in cv.split(LOGIT_OOF, y_true):
        clf = LogisticRegression(C=c_val, random_state=20261109, max_iter=200)
        clf.fit(LOGIT_OOF[tr_idx], y_true[tr_idx])
        meta_oof[val_idx] = clf.predict_proba(LOGIT_OOF[val_idx])[:, 1]
    
    acc_meta = accuracy_score(y_true, meta_oof >= 0.5)
    hits_meta = int(((meta_oof >= 0.5) == y_true).sum())
    ll_meta = log_loss(y_true, np.clip(meta_oof, 1e-7, 1 - 1e-7))
    auc_meta = roc_auc_score(y_true, meta_oof)
    print(f"C={c_val:<5} | Acc: {acc_meta:.5f} | Hits: {hits_meta} | LogLoss: {ll_meta:.5f} | AUC: {auc_meta:.5f}")

print("\n=== METHOD 4: ACCURACY OPTIMIZED BLEND (Nelder-Mead on 0/1 Loss) ===")
# Sometimes probability calibration optimization differs slightly from 0-1 accuracy
from scipy.optimize import differential_evolution

def acc_loss(w):
    w = np.maximum(0, w)
    if w.sum() == 0:
        return 1.0
    w = w / w.sum()
    pred = (OOF @ w) >= 0.5
    return 1.0 - accuracy_score(y_true, pred)

res_de = differential_evolution(acc_loss, bounds=[(0.0, 1.0)] * len(names), maxiter=50, seed=20261109)
opt_w_acc = np.maximum(0, res_de.x) / np.maximum(0, res_de.x).sum()
p_acc_oof = OOF @ opt_w_acc
acc_de = accuracy_score(y_true, p_acc_oof >= 0.5)
hits_de = int(((p_acc_oof >= 0.5) == y_true).sum())
ll_de = log_loss(y_true, np.clip(p_acc_oof, 1e-7, 1 - 1e-7))
print(f"Acc-optimal Weights: {dict(zip(names, [round(x, 4) for x in opt_w_acc]))}")
print(f"Acc: {acc_de:.5f} | Hits: {hits_de} | LogLoss: {ll_de:.5f}")
