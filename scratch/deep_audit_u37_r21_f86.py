import pickle
import pandas as pd
import numpy as np

# Load files
s4 = pd.read_csv('submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv').sort_values('ID').reset_index(drop=True)
t30 = pd.read_csv('submissions/claude_20261114/claude_blendfix_posterior_top30_sobre_S4.csv').sort_values('ID').reset_index(drop=True)
t30_u58 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U58_blendfix_noprobado.csv').sort_values('ID').reset_index(drop=True)
t30_u58_f86 = pd.read_csv('submissions/claude_20261114/claude_T30_mas_U58_mas_F86_bfix_noprobado.csv').sort_values('ID').reset_index(drop=True)
b10 = pd.read_csv('results/round141_claude_baseline/full70k/submission_b10_full.csv').sort_values('ID').reset_index(drop=True)

# Also load our clean candidate on Top30
clean_t30 = pd.read_csv('results/CANDIDATE_T30_PLUS_WARM_CLEAN.csv').sort_values('ID').reset_index(drop=True)

with open('data/splits/t98_parsed_train.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open('data/splits/t98_parsed_test.pkl', 'rb') as f:
    df_test = pickle.load(f)

# Ensure df_test is sorted by id to align with submission dataframes
df_test = df_test.sort_values('id').reset_index(drop=True)

df_split = pd.read_csv('data/splits/canonical_3way_splits.csv')
df = df_train.merge(df_split[['id', 'role', 'regime']], on='id')
df_dev = df[df['role'] != 'confirm'].copy().reset_index(drop=True)

# Regimes of test relative to df_dev (55,925 rows)
dev_times = df_dev.groupby('seller_id')['start_time'].agg(['min', 'max'])
dev_all_stamps = df_dev.groupby('seller_id')['start_time'].apply(set).to_dict()

s_test = df_test['seller_id'].values
t_test = df_test['start_time'].values

in_dev_test = np.isin(s_test, dev_times.index)
bounds_t = dev_times.reindex(s_test)
t_min_t = bounds_t['min'].values
t_max_t = bounds_t['max'].values

r_cold_t = ~in_dev_test
has_same_time_t = np.zeros(len(df_test), dtype=bool)
for k in range(len(df_test)):
    s = s_test[k]
    t = t_test[k]
    if s in dev_all_stamps and t in dev_all_stamps[s]:
        has_same_time_t[k] = True

r_ties_t = in_dev_test & has_same_time_t
r_bracketed_t = in_dev_test & ~has_same_time_t & (t_test > t_min_t) & (t_test < t_max_t)
r_before_t = in_dev_test & ~has_same_time_t & (t_test < t_min_t)
r_after_t = in_dev_test & ~has_same_time_t & (t_test > t_max_t)

regimes_test_dev = {
    'bracketed (Warm)': r_bracketed_t,
    'same_timestamp (Warm)': r_ties_t,
    'before_train (Warm)': r_before_t,
    'after_train (Warm)': r_after_t,
    'new_seller (Cold dev)': r_cold_t
}

# Combine into master audit dataframe
audit = pd.DataFrame({
    'id': s4['ID'],
    'seller_id': df_test['seller_id'],
    's4': s4['condition'],
    't30': t30['condition'],
    't30_u58': t30_u58['condition'],
    't30_u58_f86': t30_u58_f86['condition'],
    'b10': b10['condition'],
    'clean_t30': clean_t30['condition'],
})

for rname, rmask in regimes_test_dev.items():
    audit[rname] = rmask

audit['is_u58'] = audit['t30_u58'] != audit['t30']
audit['is_u37'] = audit['is_u58'] & (audit['s4'] == audit['b10'])
audit['is_r21'] = audit['is_u58'] & (audit['s4'] != audit['b10'])
audit['is_f86'] = audit['t30_u58_f86'] != audit['t30_u58']
audit['is_clean160'] = audit['clean_t30'] != audit['t30']

print("=== COUNTS AND VERIFICATION ===")
print(f"Top30 flips vs S4: {(audit['t30'] != audit['s4']).sum()}")
print(f"U58 flips vs Top30: {audit['is_u58'].sum()}")
print(f"  - U37 (Concordant S4==B10, flip away from B10): {audit['is_u37'].sum()}")
print(f"  - R21 (Discordant S4!=B10, revert to B10): {audit['is_r21'].sum()}")
print(f"F86 flips vs T30_U58: {audit['is_f86'].sum()}")
print(f"Clean160 flips vs Top30: {audit['is_clean160'].sum()}")

print("\n=== 5 REGIMES DISTRIBUTION ===")
regime_cols = list(regimes_test_dev.keys())

reg_summary = []
for r in regime_cols:
    reg_summary.append({
        'Regime': r,
        'Test_Total': audit[r].sum(),
        'U58 (58)': (audit['is_u58'] & audit[r]).sum(),
        'U37 (37)': (audit['is_u37'] & audit[r]).sum(),
        'R21 (21)': (audit['is_r21'] & audit[r]).sum(),
        'F86 (86)': (audit['is_f86'] & audit[r]).sum(),
        'Clean160': (audit['is_clean160'] & audit[r]).sum(),
    })

df_reg = pd.DataFrame(reg_summary)
print(df_reg.to_string(index=False))

print("\n=== FLIP DIRECTIONS (NEW -> USED vs USED -> NEW) ===")
for name, col in [('U58', 'is_u58'), ('U37', 'is_u37'), ('R21', 'is_r21'), ('F86', 'is_f86'), ('Clean160', 'is_clean160')]:
    sub = audit[audit[col]]
    if name == 'F86':
        f_from = sub['t30_u58']
        f_to = sub['t30_u58_f86']
    elif name == 'Clean160':
        f_from = sub['t30']
        f_to = sub['clean_t30']
    else:
        f_from = sub['t30']
        f_to = sub['t30_u58']
    n2u = ((f_from == 'new') & (f_to == 'used')).sum()
    u2n = ((f_from == 'used') & (f_to == 'new')).sum()
    print(f"{name:10}: Total={len(sub):3d} | new->used: {n2u:3d} | used->new: {u2n:3d}")

# Let's save the exact IDs of U37 and R21
audit[audit['is_u37']][['id', 'seller_id', 's4', 't30', 't30_u58', 'b10'] + regime_cols].to_csv('scratch/u37_detail.csv', index=False)
audit[audit['is_r21']][['id', 'seller_id', 's4', 't30', 't30_u58', 'b10'] + regime_cols].to_csv('scratch/r21_detail.csv', index=False)
