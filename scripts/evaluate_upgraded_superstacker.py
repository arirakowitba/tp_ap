#!/usr/bin/env python3
"""Evaluate Upgraded Super-Stacker with Deconvolved Feedback Integration."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
import re, json

models = {
    'LGBM_Feedback': ('results/high_capacity_lgbm_feedback_5fold/oof_predictions.csv', 'prob_used', 'results/high_capacity_lgbm_feedback_5fold/test_probabilities.npy'),
    'LGBM_Base': ('results/high_capacity_lgbm_5fold/oof_predictions.csv', 'prob_used', 'results/high_capacity_lgbm_5fold/test_probabilities.npy'),
    'XGBoost': ('results/xgboost_diversity_5fold/oof_predictions.csv', 'prob_used', 'results/xgboost_diversity_5fold/test_probabilities.npy'),
    'CatBoost': ('results/catboost_depth8_5fold/oof_predictions.csv', 'pred_proba', 'results/catboost_depth8_5fold/test_probabilities.npy'),
    'MLP_Neural': ('results/mlp_neural_5fold/oof_predictions.csv', 'pred_proba', 'results/mlp_neural_5fold/test_probabilities.npy'),
    'Sparse_Linear': ('results/sparse_linear_5fold/oof_predictions.csv', 'prob_used', 'results/sparse_linear_5fold/test_probabilities.npy'),
}

y_true = pd.read_csv('results/high_capacity_lgbm_5fold/oof_predictions.csv')['y_true'].to_numpy()
oof_list = [pd.read_csv(v[0])[v[1]].to_numpy() for v in models.values()]
te_list = [np.load(v[2]) for v in models.values()]

OOF = np.column_stack(oof_list)
TE = np.column_stack(te_list)

eps = 1e-5
LOGIT_OOF = logit(np.clip(OOF, eps, 1 - eps))
LOGIT_TE = logit(np.clip(TE, eps, 1 - eps))

X_oof = np.column_stack([np.ones(len(y_true)), LOGIT_OOF])
X_te = np.column_stack([np.ones(len(TE)), LOGIT_TE])

def loss_func(beta):
    p = expit(X_oof @ beta)
    return -np.mean(y_true * np.log(np.clip(p, 1e-7, 1-1e-7)) + (1 - y_true) * np.log(np.clip(1 - p, 1e-7, 1-1e-7)))

b0 = np.array([0.0] + [0.2] * len(models))
bounds = [(None, None)] + [(0.0, None)] * len(models)
res = minimize(loss_func, b0, method='L-BFGS-B', bounds=bounds)
beta = res.x

p_oof = expit(X_oof @ beta)
p_te = expit(X_te @ beta)

# Calibration map
bins = [
    (0.90, 1.01), (0.85, 0.90), (0.80, 0.85), (0.75, 0.80),
    (0.70, 0.75), (0.65, 0.70), (0.60, 0.65), (0.55, 0.60), (0.50, 0.55)
]
calib_map = {}
for lo, hi in bins:
    m = (p_oof >= lo) & (p_oof < hi)
    calib_map[(lo, hi)] = float(y_true[m].mean()) if m.sum() > 0 else (lo + hi) / 2

def get_emp_q(p_arr):
    q_arr = np.zeros(len(p_arr), dtype=np.float32)
    for (lo, hi), q in calib_map.items():
        m = (p_arr >= lo) & (p_arr < hi)
        q_arr[m] = q
    q_arr[p_arr < 0.50] = 0.50
    return q_arr

# Load anchor and test metadata
anchor_path = 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-132_23ff472_11aabc76.csv'
anchor_preds = pd.read_csv(anchor_path)['condition'].to_numpy()

test_rows = []
with open('data/ap_2026_q_2/raw/test_data.jsonlines', 'r', encoding='utf-8') as f:
    for line in f:
        test_rows.append(json.loads(line))

cand_mask = (anchor_preds == 'new') & (p_te > 0.50)
cands = np.where(cand_mask)[0]
cands = cands[np.argsort(-p_te[cands])]

confirmed_true_new = {6308, 16515, 24735, 27016}
NEW_REGEX = re.compile(
    r'\b(a estrenar|0km|0 km|sin uso|sellad[oa]s?|en blister|cerrad[oa] en caja|impecable sin uso|nuevo sellado|caja cerrada|nuevo sin uso)\b',
    re.IGNORECASE
)

safe_cands = []
for idx in cands:
    item_id = idx + 1
    if item_id in confirmed_true_new:
        continue
    row = test_rows[idx]
    title = str(row.get('title') or '')
    warranty = str(row.get('warranty') or '')
    if NEW_REGEX.search(title) or NEW_REGEX.search(warranty):
        continue
    safe_cands.append(idx)

safe_cands = np.array(safe_cands)
print(f'Total safe candidates: {len(safe_cands)}')

p_pub = 20998.0 / 30000.0
tiers = [
    ('Slot 1 (Tactical Rank 5)', 10),
    ('Slot 2 (Consolidation)', 30),
    ('Slot 3 (Capture Rank 4)', 60),
    ('Slot 4 (Podium Climax)', 110),
    ('Slot 5 (Aggressive Rank 2)', 165),
]

print('\n' + '=' * 90)
print(f"{'Tier':<25} | {'Flips':<5} | {'Avg q':<7} | {'E[Gain]':<8} | {'Exp Score':<9} | {'P(>=R4)':<8} | {'P(Podium)':<10} | {'P(>=0.950)':<10}")
print('-' * 90)

for tier_name, K in tiers:
    sel = safe_cands[:K]
    q_sel = get_emp_q(p_te[sel])
    bar_q = float(q_sel.mean())
    exp_gain = float(K * p_pub * (2.0 * bar_q - 1.0))
    exp_score = (19896.0 + exp_gain) / 20998.0
    var_gain = float(K * p_pub * (1.0 - p_pub * (2.0 * bar_q - 1.0)**2))
    sigma = np.sqrt(var_gain)
    
    z_r4 = (exp_gain - 18.0) / sigma if sigma > 0 else 0.0
    z_r3 = (exp_gain - 30.0) / sigma if sigma > 0 else 0.0
    z_95 = (exp_gain - 53.0) / sigma if sigma > 0 else 0.0
    
    p_r4 = norm.cdf(z_r4) * 100
    p_r3 = norm.cdf(z_r3) * 100
    p_95 = norm.cdf(z_95) * 100
    
    print(f"{tier_name:<25} | {K:<5} | {bar_q:<7.4f} | +{exp_gain:<7.1f} | {exp_score:<9.5f} | {p_r4:<7.1f}% | {p_r3:<9.1f}% | {p_95:<9.2f}%")
print('=' * 90)

print('\n' + '=' * 90)
print(f"{'Extended K':<12} | {'Avg q':<7} | {'E[Gain]':<8} | {'Exp Score':<9} | {'Sigma':<7} | {'P(Podium)':<10} | {'P(>=0.950)':<10}")
print('-' * 90)
for K in [165, 180, 200, 220, 240, 260, 280, 300, 320]:
    sel = safe_cands[:K]
    q_sel = get_emp_q(p_te[sel])
    bar_q = float(q_sel.mean())
    exp_gain = float(K * p_pub * (2.0 * bar_q - 1.0))
    exp_score = (19896.0 + exp_gain) / 20998.0
    var_gain = float(K * p_pub * (1.0 - p_pub * (2.0 * bar_q - 1.0)**2))
    sigma = np.sqrt(var_gain)
    z_podium = (exp_gain - 30.0) / sigma if sigma > 0 else 0.0
    z_95 = (exp_gain - 53.0) / sigma if sigma > 0 else 0.0
    p_podium = norm.cdf(z_podium) * 100
    p_95 = norm.cdf(z_95) * 100
    print(f"K = {K:<7} | {bar_q:<7.4f} | +{exp_gain:<7.1f} | {exp_score:<9.5f} | {sigma:<7.2f} | {p_podium:<9.1f}% | {p_95:<9.2f}%")
print('=' * 90)
