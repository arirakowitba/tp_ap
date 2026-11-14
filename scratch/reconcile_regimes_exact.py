import pickle
import pandas as pd
import numpy as np

# Load test in original file order
with open('data/splits/t98_parsed_test.pkl', 'rb') as f:
    df_test = pickle.load(f)

# The row order in df_test is 0..29999, which corresponds to ID 1..30000
print("df_test row 0..4 id:", df_test['id'].head().tolist())

# Load train
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

# KEEP ORIGINAL ROW ORDER OF test_jl (row 0 = ID 1, row 29999 = ID 30000)
reg_full70k = claude_regimes(train_jl, test_jl)
reg_dev = claude_regimes(dev_jl, test_jl)

# Load U36, U37, Clean
t30 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv')
u36 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U36_estricto.csv')
u37 = pd.read_csv('results/CANDIDATE_T30_PLUS_U37.csv')
clean = pd.read_csv('results/CANDIDATE_T30_PLUS_WARM_CLEAN.csv')

# Note: ID is 1..30000, so index 0..29999 directly corresponds to test_jl row 0..29999!
mask_u36 = (u36['condition'] != t30['condition']).values
mask_u37 = (u37['condition'] != t30['condition']).values
mask_clean = (clean['condition'] != t30['condition']).values

print("\n--- TEST REGIMES (30,000 rows) ---")
print("Full 70k fit:")
print(pd.Series(reg_full70k).value_counts())
print("\nDev 55.9k fit:")
print(pd.Series(reg_dev).value_counts())

print("\n--- U36 (36 flips) under Full 70k fit ---")
print(pd.Series(reg_full70k[mask_u36]).value_counts())

print("\n--- U36 (36 flips) under Dev 55.9k fit ---")
print(pd.Series(reg_dev[mask_u36]).value_counts())

print("\n--- U37 (37 flips) under Full 70k fit ---")
print(pd.Series(reg_full70k[mask_u37]).value_counts())

print("\n--- U37 (37 flips) under Dev 55.9k fit ---")
print(pd.Series(reg_dev[mask_u37]).value_counts())

print("\n--- Clean160 under Full 70k fit ---")
print(pd.Series(reg_full70k[mask_clean]).value_counts())

print("\n--- Clean160 under Dev 55.9k fit ---")
print(pd.Series(reg_dev[mask_clean]).value_counts())
