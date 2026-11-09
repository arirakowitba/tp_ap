#!/usr/bin/env python3
"""Audit Candidate Flips with Semantic and Domain Filters."""

import json
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

# 1. Compute 4-Architecture Logit Stack
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

p_te = expit(X_te @ beta)

# Load test metadata
test_rows = []
with open('data/ap_2026_q_2/raw/test_data.jsonlines', 'r', encoding='utf-8') as f:
    for line in f:
        test_rows.append(json.loads(line))

# Load EVT-132 Anchor
anchor_path = 'results/canonical_ledger_rebuild_20261108/evaluated_vectors_git_pinned/EVT-132_23ff472_11aabc76.csv'
df_anchor = pd.read_csv(anchor_path)
anchor_preds = df_anchor['condition'].to_numpy()

cand_mask = (anchor_preds == 'new') & (p_te > 0.50)
cands = np.where(cand_mask)[0]
cands = cands[np.argsort(-p_te[cands])]

confirmed_true_new = {6308, 16515, 24735, 27016}

# Domain new signals that should protect an item from being flipped to used
NEW_REGEX = re.compile(
    r'\b(a estrenar|0km|0 km|sin uso|sellad[oa]s?|en blister|cerrad[oa] en caja|impecable sin uso|nuevo sellado|caja cerrada|nuevo sin uso)\b',
    re.IGNORECASE
)

# Domain used signals that strongly confirm used
USED_REGEX = re.compile(
    r'\b(usad[oa]s?|antigu[oa]s?|lote de|segunda mano|para repuesto|detalles de uso|leves detalles|buen estado|muy buen estado|antiguedad|vintage)\b',
    re.IGNORECASE
)

print(f"Total raw candidates (anchor=new, p_te > 0.50): {len(cands)}")

safe_candidates = []
flagged_false_positives = []

for idx in cands:
    item_id = idx + 1
    if item_id in confirmed_true_new:
        continue
    row = test_rows[idx]
    title = str(row.get('title') or '')
    warranty = str(row.get('warranty') or '')
    
    # Check if text contains strong NEW indicators
    if NEW_REGEX.search(title) or NEW_REGEX.search(warranty):
        flagged_false_positives.append((item_id, p_te[idx], title, "Regex matched strong NEW marker"))
        continue
        
    safe_candidates.append(idx)

print(f"Flagged as likely False Positives (protected from flip): {len(flagged_false_positives)}")
for item_id, p, title, reason in flagged_false_positives[:10]:
    print(f"  BLOCKED ID {item_id}: P={p:.4f} | Title: \"{title}\" ({reason})")

print(f"\nRemaining high-confidence safe candidates: {len(safe_candidates)}")
print(f"Top 20 Safe Candidates:")
for rank, idx in enumerate(safe_candidates[:20], 1):
    row = test_rows[idx]
    title = str(row.get('title') or '')
    has_used_cue = bool(USED_REGEX.search(title))
    cue_str = " [CONFIRMED USED CUE]" if has_used_cue else ""
    print(f"  {rank:<2}: ID={idx+1:<5} | P={p_te[idx]:.4f} | \"{title[:50]}\"{cue_str}")
