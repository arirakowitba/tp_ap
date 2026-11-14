import pickle
import pandas as pd
import numpy as np

# Load test in original file order
train_jl = pd.read_json('data/ap_2026_q_2/raw/train_data.jsonlines', lines=True)
test_jl = pd.read_json('data/ap_2026_q_2/raw/test_data.jsonlines', lines=True)

df_split = pd.read_csv('data/splits/canonical_3way_splits.csv')
dev_ids = set(df_split[df_split['role'] != 'confirm']['id'])
dev_jl = train_jl[train_jl['id'].isin(dev_ids)].copy()

def claude_regimes(f, e):
    s = f.groupby('seller_id').date_created.agg(['min', 'max'])
    d = pd.to_datetime(e.date_created).dt.floor('D')
    mn = pd.to_datetime(e.seller_id.map(s['min'])).dt.floor('D')
    mx = pd.to_datetime(e.seller_id.map(s['max'])).dt.floor('D')
    return np.where(mn.isna(), 'cold', np.where(d < mn, 'before', np.where(d > mx, 'after', np.where((d == mn) | (d == mx), 'ties', 'bracketed'))))

reg_full70k = claude_regimes(train_jl, test_jl)
reg_dev = claude_regimes(dev_jl, test_jl)

np.save('scratch/reg_full70k.npy', reg_full70k)
np.save('scratch/reg_dev.npy', reg_dev)
print("Saved reg_full70k.npy and reg_dev.npy successfully!")
