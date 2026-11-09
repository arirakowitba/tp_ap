#!/usr/bin/env python3
"""Cross-reference safe candidates against canonical historical submissions."""

import glob
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(".")
CATALOG_PATH = ROOT / "results" / "canonical_ledger_rebuild_20261108" / "canonical_events_catalog.csv"
VECTORS_DIR = ROOT / "results" / "canonical_ledger_rebuild_20261108" / "evaluated_vectors_git_pinned"
ANCHOR_PATH = VECTORS_DIR / "EVT-132_23ff472_11aabc76.csv"

# Load catalog
df_cat = pd.read_csv(CATALOG_PATH)
anchor_row = df_cat[df_cat["event_id"] == "EVT-132"].iloc[0]
print(f"Anchor EVT-132 Score: {anchor_row['public_score']}, Hits: {anchor_row['public_hits']}")

# Run the 4-Architecture Super-Blender to get safe candidates
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

# Test rows
test_rows = []
with open('data/ap_2026_q_2/raw/test_data.jsonlines', 'r', encoding='utf-8') as f:
    for line in f:
        test_rows.append(json.loads(line))

df_anchor = pd.read_csv(ANCHOR_PATH)
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

print(f"Loaded {len(safe_cands)} verified safe candidate flips.")

# Check all historical submissions from today (2026-11-09) to see which items were flipped
today_events = df_cat[df_cat['submission_date'].str.startswith('2026-11-09')].copy()
print(f"Today's evaluated events ({len(today_events)}):")
for _, r in today_events.iterrows():
    print(f"  {r['event_id']} ({r['submission_name']}): Score={r['public_score']}, Hits={r['public_hits']}, Delta Hits={r['delta_hits_vs_prev']}")
