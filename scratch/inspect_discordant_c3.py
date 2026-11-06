import pandas as pd
import numpy as np

c3 = pd.read_csv('submissions/today_20261105_portfolio_09480/submission_20261105_03_conservative_multimodal_top60.csv')
p_d1d2 = 'submissions/candidates_20261104/candidate_01_d1d2_plus_upper10.csv'
df_ref = pd.read_csv(p_d1d2)
UPPER10 = [4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453]
d1d2 = df_ref.copy()
d1d2.loc[d1d2['ID'].isin(UPPER10), 'condition'] = 'used'

df_cb = pd.read_csv('results/full_native_nb_70k_20261105/test_probabilities.csv')
df_lgb = pd.read_csv('results/full_lightgbm_enriched_70k_20261105/lgb_enriched_test_probabilities.csv')
p_sub = np.load('results/frente_c_subword_and_diagnostics_20261104/test_probs_subword_rich_text.npy')

cb_pred = np.where(df_cb['p_native_text'] >= 0.5, 'used', 'new')
nb_pred = np.where(df_cb['p_nested_nb'] >= 0.5, 'used', 'new')
lgb_pred = np.where(df_lgb['p_lgb_enriched'] >= 0.5, 'used', 'new')
sub_pred = np.where(p_sub >= 0.5, 'used', 'new')

unanimous_3 = (cb_pred == nb_pred) & (nb_pred == lgb_pred)

# Find the 7 discordant items in C3
c3_flips = c3[c3['condition'] != d1d2['condition']]['ID'].tolist()
c3_flips_no_u10 = [x for x in c3_flips if x not in UPPER10]
idx_c3 = [x - 1 for x in c3_flips_no_u10]

discordant_c3_ids = [c3_flips_no_u10[i] for i in range(len(c3_flips_no_u10)) if not unanimous_3[idx_c3[i]]]
print('The 7 discordant IDs in C3 that hurt us:', discordant_c3_ids)

for i in discordant_c3_ids:
    idx = i - 1
    d_c = d1d2.loc[idx, 'condition']
    c_c = c3.loc[idx, 'condition']
    p1 = df_cb.loc[idx, 'p_native_text']
    p2 = df_cb.loc[idx, 'p_nested_nb']
    p3 = df_lgb.loc[idx, 'p_lgb_enriched']
    p4 = p_sub[idx]
    print(f'ID {i}: d1d2={d_c}, c3={c_c} | CB_nat={p1:.3f}, CB_nb={p2:.3f}, LGB={p3:.3f}, Sub={p4:.3f}')
