import json, pandas as pd
from pathlib import Path

ROOT = Path('.')
with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f]

ids = [7610, 13637, 14589, 16674, 29896]
df_cb = pd.read_csv('results/full_native_nb_70k_20261105/test_probabilities.csv')
df_lgb = pd.read_csv('results/full_lightgbm_enriched_70k_20261105/lgb_enriched_test_probabilities.csv')

for i in ids:
    r = rows[i - 1]
    p_cb = df_cb.loc[i - 1, 'p_blend']
    p_lgb = df_lgb.loc[i - 1, 'p_lgb_enriched']
    print(f"ID {i}: p_cb={p_cb:.4f}, p_lgb={p_lgb:.4f} | title='{r.get('title')}' | warranty='{r.get('warranty')}'")
