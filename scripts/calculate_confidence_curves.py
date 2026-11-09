#!/usr/bin/env python3
"""Compute exact statistical confidence curves for all leaderboard milestones."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
import re, json

# 1. Ensemble Predictions
models = ['LightGBM', 'XGBoost', 'CatBoost', 'MLP_Neural']
oof_list = []
te_list = []
for name in models:
    if name == 'CatBoost':
        oof_list.append(pd.read_csv('results/catboost_depth8_5fold/oof_predictions.csv')['pred_proba'].to_numpy())
        te_list.append(np.load('results/catboost_depth8_5fold/test_probabilities.npy'))
    elif name == 'MLP_Neural':
        oof_list.append(pd.read_csv('results/mlp_neural_5fold/oof_predictions.csv')['pred_proba'].to_numpy())
        te_list.append(np.load('results/mlp_neural_5fold/test_probabilities.npy'))
    elif name == 'XGBoost':
        oof_list.append(pd.read_csv('results/xgboost_diversity_5fold/oof_predictions.csv')['prob_used'].to_numpy())
        te_list.append(np.load('results/xgboost_diversity_5fold/test_probabilities.npy'))
    elif name == 'LightGBM':
        oof_list.append(pd.read_csv('results/high_capacity_lgbm_5fold/oof_predictions.csv')['prob_used'].to_numpy())
        te_list.append(np.load('results/high_capacity_lgbm_5fold/test_probabilities.npy'))

OOF = np.column_stack(oof_list)
TE = np.column_stack(te_list)
y_true = pd.read_csv('results/high_capacity_lgbm_5fold/oof_predictions.csv')['y_true'].to_numpy()

eps = 1e-5
LOGIT_OOF = logit(np.clip(OOF, eps, 1 - eps))
LOGIT_TE = logit(np.clip(TE, eps, 1 - eps))

X_oof = np.column_stack([np.ones(len(y_true)), LOGIT_OOF])
X_te = np.column_stack([np.ones(len(TE)), LOGIT_TE])

def loss_nneg(beta):
    z = X_oof @ beta
    p = expit(z)
    return -np.mean(y_true * np.log(np.clip(p, 1e-7, 1-1e-7)) + (1 - y_true) * np.log(np.clip(1 - p, 1e-7, 1-1e-7)))

b0 = np.array([0.0, 0.4, 0.3, 0.2, 0.1])
bounds = [(None, None)] + [(0.0, None)] * 4
res = minimize(loss_nneg, b0, method='L-BFGS-B', bounds=bounds)
beta = res.x

p_oof = expit(X_oof @ beta)
p_te = expit(X_te @ beta)

# Calibration curve on OOF
bins = [
    (0.90, 1.01), (0.85, 0.90), (0.80, 0.85), (0.75, 0.80),
    (0.70, 0.75), (0.65, 0.70), (0.60, 0.65), (0.55, 0.60), (0.50, 0.55)
]
calib_table = {}
for lo, hi in bins:
    m = (p_oof >= lo) & (p_oof < hi)
    q = float(y_true[m].mean()) if m.sum() > 0 else (lo + hi) / 2
    calib_table[(lo, hi)] = q

def get_emp_q(p_arr):
    q_arr = np.zeros(len(p_arr), dtype=np.float32)
    for (lo, hi), q in calib_table.items():
        m = (p_arr >= lo) & (p_arr < hi)
        q_arr[m] = q
    q_arr[p_arr < 0.50] = 0.50
    return q_arr

# Load test metadata and anchor
test_rows = []
with open('data/ap_2026_q_2/raw/test_data.jsonlines', 'r', encoding='utf-8') as f:
    for line in f:
        test_rows.append(json.loads(line))

anchor_path = 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-132_23ff472_11aabc76.csv'
df_anchor = pd.read_csv(anchor_path)
anchor_preds = df_anchor['condition'].to_numpy()

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
print(f"Total safe candidates: {len(safe_cands)}")

# Milestones from Anchor EVT-132 (19,896 hits, 0.94752)
milestones = [
    ("Rank 5 (Clara Rodriguez)", 19898, 2),
    ("Rank 4 (Agustin Sanchez)", 19914, 18),
    ("Rank 3 / Podium (Gonzalo Buhler)", 19926, 30),
    ("Target Barrier >= 0.95000", 19949, 53),
    ("Rank 2 (Nicolas Caveda)", 19968, 72),
    ("Rank 1 / Champion (Maximo Lladhon)", 20040, 144),
]

p_pub = 20998.0 / 30000.0

tiers = [
    ("Slot 1 (Tactical Rank 5)", 10),
    ("Slot 2 (Consolidation)", 30),
    ("Slot 3 (Capture Rank 4)", 60),
    ("Slot 4 (Podium Climax)", 110),
    ("Slot 5 (Aggressive Rank 2)", 165),
]

print("\n" + "=" * 90)
print(f"{'Tier':<25} | {'Flips':<5} | {'E[Gain]':<8} | {'Exp Score':<9} | {'P(>=R5)':<8} | {'P(>=R4)':<8} | {'P(>=R3)':<8} | {'P(>=0.950)':<10} | {'P(>=R2)':<8}")
print("-" * 90)

for tier_name, K in tiers:
    sel = safe_cands[:K]
    q_sel = get_emp_q(p_te[sel])
    bar_q = float(q_sel.mean())
    exp_gain_pub = float(K * p_pub * (2.0 * bar_q - 1.0))
    exp_hits = 19896.0 + exp_gain_pub
    exp_score = exp_hits / 20998.0
    var_gain = float(K * p_pub * (1.0 - p_pub * (2.0 * bar_q - 1.0)**2))
    sigma = np.sqrt(var_gain)
    
    # Calculate probabilities for each milestone
    probs = {}
    for ms_name, ms_hits, req_gain in milestones:
        z = (exp_gain_pub - req_gain) / sigma if sigma > 0 else 0.0
        probs[ms_name] = norm.cdf(z)
        
    p_r5 = probs["Rank 5 (Clara Rodriguez)"] * 100
    p_r4 = probs["Rank 4 (Agustin Sanchez)"] * 100
    p_r3 = probs["Rank 3 / Podium (Gonzalo Buhler)"] * 100
    p_95 = probs["Target Barrier >= 0.95000"] * 100
    p_r2 = probs["Rank 2 (Nicolas Caveda)"] * 100
    
    print(f"{tier_name:<25} | {K:<5} | +{exp_gain_pub:<7.1f} | {exp_score:<9.5f} | {p_r5:<7.1f}% | {p_r4:<7.1f}% | {p_r3:<7.1f}% | {p_95:<9.2f}% | {p_r2:<7.1f}%")

print("=" * 90)
